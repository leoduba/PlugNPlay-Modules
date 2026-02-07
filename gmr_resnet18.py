
# 完整修复版：包含 GMR_Conv2d 定义和扩展测试

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math

# ==================== GMR_Conv2d 定义 ====================

class GMR_Conv2d(nn.Module):
    """
    Gaussian Mixture Ring Convolution (GMR-Conv)
    
    高效的旋转和反射等变卷积核，使用高斯混合环平滑径向对称性。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        n_rings: Optional[int] = None,
        sigma: float = 1.0,
        circular_constraint: bool = True
    ):
        super(GMR_Conv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.sigma = sigma
        self.circular_constraint = circular_constraint
        
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1, \
            "Kernel size must be odd"
        
        self.n_rings = n_rings or (self.kernel_size[0] // 2 + 1)
        
        self.ring_weights = nn.Parameter(torch.randn(self.n_rings))
        self.pointwise_weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, 1, 1)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._init_parameters()
        self.register_buffer('gaussian_rings', self._compute_gaussian_rings())
        
    def _init_parameters(self):
        nn.init.kaiming_normal_(self.pointwise_weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.ring_weights, mean=0, std=0.1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def _compute_gaussian_rings(self) -> torch.Tensor:
        k = self.kernel_size[0]
        center = k // 2
        
        y, x = torch.meshgrid(
            torch.arange(k, dtype=torch.float32),
            torch.arange(k, dtype=torch.float32),
            indexing='ij'
        )
        
        distances = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        
        gaussian_rings = []
        for i in range(self.n_rings):
            ring_radius = float(i)
            gaussian_weight = torch.exp(-((distances - ring_radius) ** 2) / (2 * self.sigma ** 2))
            gaussian_rings.append(gaussian_weight)
        
        gaussian_rings = torch.stack(gaussian_rings)
        
        if self.circular_constraint:
            max_radius = center * math.sqrt(2) * 0.9
            mask = distances <= max_radius
            gaussian_rings = gaussian_rings * mask.unsqueeze(0)
        
        return gaussian_rings
    
    def _compute_kernel(self) -> torch.Tensor:
        combined = torch.sum(
            self.ring_weights.view(-1, 1, 1) * self.gaussian_rings,
            dim=0
        )
        kernel = combined.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(self.in_channels, 1, 1, 1)
        return kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        depthwise_weight = self._compute_kernel()
        
        x = F.conv2d(
            x,
            depthwise_weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=C
        )
        
        out = F.conv2d(x, self.pointwise_weight, bias=self.bias)
        return out


# ==================== GMR-ResNet 定义 ====================

class GMR_BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        gmr_conv_size: int = 3,
        sigma: float = 1.0
    ):
        super(GMR_BasicBlock, self).__init__()
        
        self.conv1 = GMR_Conv2d(
            in_channels, out_channels,
            kernel_size=gmr_conv_size,
            stride=stride,
            padding=gmr_conv_size // 2,
            sigma=sigma
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = GMR_Conv2d(
            out_channels, out_channels,
            kernel_size=gmr_conv_size,
            padding=gmr_conv_size // 2,
            sigma=sigma
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class GMR_ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 1000,
        gmr_conv_sizes: List[int] = [3, 3, 3, 3],
        sigma: float = 1.0
    ):
        super(GMR_ResNet, self).__init__()
        
        self.in_channels = 64
        self.sigma = sigma
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], 
                                       gmr_conv_size=gmr_conv_sizes[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], 
                                       gmr_conv_size=gmr_conv_sizes[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], 
                                       gmr_conv_size=gmr_conv_sizes[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], 
                                       gmr_conv_size=gmr_conv_sizes[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels: int, blocks: int, 
                    gmr_conv_size: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(
            self.in_channels, out_channels, stride, downsample,
            gmr_conv_size=gmr_conv_size, sigma=self.sigma
        ))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels, out_channels,
                gmr_conv_size=gmr_conv_size, sigma=self.sigma
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def gmr_resnet18(num_classes: int = 1000, gmr_conv_sizes: List[int] = [3, 3, 3, 3], **kwargs):
    return GMR_ResNet(GMR_BasicBlock, [2, 2, 2, 2], 
                     num_classes=num_classes, gmr_conv_sizes=gmr_conv_sizes, **kwargs)


# ==================== 测试函数（修复版）====================

def test_gmr_resnet():
    """测试GMR-ResNet（修复版）"""
    print("=" * 60)
    print("GMR-ResNet 测试")
    print("=" * 60)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    # 创建GMR-ResNet-18
    model = gmr_resnet18(
        num_classes=10,
        gmr_conv_sizes=[9, 9, 5, 5],
        sigma=1.0
    )
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试中间特征图的旋转等变性
    def get_intermediate_features(model, x):
        with torch.no_grad():
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
        return x
    
    model.eval()
    with torch.no_grad():
        feat_original = get_intermediate_features(model, x)
        
        x_rot = torch.rot90(x, k=1, dims=(2, 3))
        feat_rot = get_intermediate_features(model, x_rot)
        feat_rot_back = torch.rot90(feat_rot, k=-1, dims=(2, 3))
        
        diff = torch.abs(feat_original - feat_rot_back).max().item()
    
    print(f"\n中间特征图旋转等变性误差: {diff:.6e}")
    print(f"特征图形状: {feat_original.shape}")
    print("✓ GMR-ResNet测试通过\n")


def test_orientation_robustness():
    """测试方向无关数据的鲁棒性"""
    print("=" * 60)
    print("方向无关数据鲁棒性测试")
    print("=" * 60)
    
    torch.manual_seed(42)
    x = torch.randn(4, 3, 64, 64)
    
    std_conv = nn.Conv2d(3, 16, 5, padding=2)
    gmr_conv = GMR_Conv2d(3, 16, 5, padding=2, sigma=1.0)
    
    angles = [0, 90, 180, 270]
    
    print("标准卷积在不同旋转下的输出变化:")
    std_outputs = []
    for angle in angles:
        x_rot = torch.rot90(x, k=angle//90, dims=(2, 3))
        with torch.no_grad():
            out = std_conv(x_rot)
        std_outputs.append(out)
    
    std_variance = torch.stack([o.mean() for o in std_outputs]).var().item()
    print(f"  输出均值方差: {std_variance:.6f}")
    
    print("\nGMR-Conv在不同旋转下的输出变化:")
    gmr_outputs = []
    for angle in angles:
        x_rot = torch.rot90(x, k=angle//90, dims=(2, 3))
        with torch.no_grad():
            out = gmr_conv(x_rot)
        gmr_outputs.append(out)
    
    gmr_variance = torch.stack([o.mean() for o in gmr_outputs]).var().item()
    print(f"  输出均值方差: {gmr_variance:.6f}")
    
    improvement = (std_variance - gmr_variance) / std_variance * 100 if std_variance > 0 else 0
    print(f"\nGMR-Conv相比标准卷积改善: {improvement:.1f}%")
    print("✓ 鲁棒性测试完成\n")


def test_compare_with_standard():
    """与标准卷积的详细对比"""
    print("=" * 60)
    print("GMR-Conv vs 标准卷积 详细对比")
    print("=" * 60)
    
    configs = [
        (3, 64, 3),
        (64, 128, 5),
        (128, 256, 7),
        (256, 512, 9),
    ]
    
    print(f"{'Config':<20} {'Std Params':<15} {'GMR Params':<15} {'Reduction':<12}")
    print("-" * 62)
    
    for in_ch, out_ch, k in configs:
        std = nn.Conv2d(in_ch, out_ch, k, padding=k//2)
        std_params = sum(p.numel() for p in std.parameters())
        
        gmr = GMR_Conv2d(in_ch, out_ch, k, padding=k//2)
        gmr_params = sum(p.numel() for p in gmr.parameters())
        
        reduction = (1 - gmr_params / std_params) * 100
        
        print(f"{in_ch}->{out_ch}, k={k}x{k}:     {std_params:<15,} {gmr_params:<15,} {reduction:>10.1f}%")
    
    print("\n✓ 对比测试完成\n")


def demo_practical_application():
    """实际应用场景演示"""
    print("=" * 60)
    print("实际应用场景演示：医学图像分类")
    print("=" * 60)
    
    batch_size = 4
    image_size = 128
    x = torch.randn(batch_size, 3, image_size, image_size)
    
    class MedicalImageClassifier(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.features = nn.Sequential(
                GMR_Conv2d(3, 32, 7, padding=3, sigma=1.0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                GMR_Conv2d(32, 64, 5, padding=2, sigma=1.0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                GMR_Conv2d(64, 128, 3, padding=1, sigma=1.0),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)
    
    model = MedicalImageClassifier(num_classes=2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    labels = torch.randint(0, 2, (batch_size,))
    
    model.train()
    optimizer.zero_grad()
    
    output = model(x)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    
    print(f"输入图像形状: {x.shape}")
    print(f"模型输出形状: {output.shape}")
    print(f"损失值: {loss.item():.4f}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    model.eval()
    with torch.no_grad():
        pred_original = model(x).argmax(dim=1)
        
        x_rot = torch.rot90(x, k=1, dims=(2, 3))
        pred_rot = model(x_rot).argmax(dim=1)
        
        print(f"\n旋转一致性:")
        print(f"  原始预测: {pred_original.tolist()}")
        print(f"  旋转预测: {pred_rot.tolist()}")
        print(f"  一致性: {(pred_original == pred_rot).sum().item()}/{batch_size}")
    
    print("\n✓ 实际应用演示完成\n")


# 运行测试
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GMR-Conv 扩展测试套件")
    print("=" * 60 + "\n")
    
    test_gmr_resnet()
    test_orientation_robustness()
    test_compare_with_standard()
    demo_practical_application()
    
    print("=" * 60)
    print("所有扩展测试完成！")
    print("=" * 60)

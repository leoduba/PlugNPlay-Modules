
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List
import math
#Github地址：https://arxiv.org/pdf/2504.02819
#论文地址：GMR-Conv: An Efficient Rotation and Reflection EquivariantConvolution Kernel Using Gaussian Mixture Rings
class GMR_Conv2d(nn.Module):
    """
    Gaussian Mixture Ring Convolution (GMR-Conv)
    
    高效的旋转和反射等变卷积核，使用高斯混合环平滑径向对称性。
    
    核心思想：
    1. 使用基于到中心欧氏距离的独立圆环参数化2D核
    2. 每个环对应一个可训练参数，实现局部旋转/反射不变性
    3. 使用高斯加权混合平滑离散化误差
    4. 通过深度可分离卷积优化计算效率
    
    复杂度：O(HWn(k² + C_in*C_out)) vs 标准卷积 O(HWk²C_in*C_out)
    其中n是环的数量，通常 n << C_in*C_out
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小（奇数）
        stride: 步长，默认1
        padding: 填充，默认0
        dilation: 空洞率，默认1
        groups: 分组数，默认1
        bias: 是否使用偏置，默认True
        n_rings: 环的数量，默认None（自动计算为kernel_size//2 + 1）
        sigma: 高斯标准差，默认1.0
        circular_constraint: 是否应用圆形约束（将角落设为零），默认True
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
        
        # 确保kernel size是奇数
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1, \
            "Kernel size must be odd"
        
        # 计算环的数量（从中心到角落的距离）
        self.n_rings = n_rings or (self.kernel_size[0] // 2 + 1)
        
        # 可学习的环权重 [n_rings]
        self.ring_weights = nn.Parameter(torch.randn(self.n_rings))
        
        # 1x1逐点卷积权重 [out_channels, in_channels, 1, 1]
        self.pointwise_weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, 1, 1)
        )
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 初始化
        self._init_parameters()
        
        # 预计算高斯环核（不学习，只作为结构）
        self.register_buffer('gaussian_rings', self._compute_gaussian_rings())
        
    def _init_parameters(self):
        """参数初始化"""
        nn.init.kaiming_normal_(self.pointwise_weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.ring_weights, mean=0, std=0.1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def _compute_gaussian_rings(self) -> torch.Tensor:
        """
        计算高斯混合环结构
        
        Returns:
            gaussian_rings: [n_rings, kernel_size, kernel_size]
        """
        k = self.kernel_size[0]
        center = k // 2
        
        # 创建坐标网格
        y, x = torch.meshgrid(
            torch.arange(k, dtype=torch.float32),
            torch.arange(k, dtype=torch.float32),
            indexing='ij'
        )
        
        # 计算每个位置到中心的距离
        distances = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
        
        gaussian_rings = []
        for i in range(self.n_rings):
            # 每个环对应一个半径距离
            ring_radius = float(i)
            # 高斯权重：距离越接近ring_radius，权重越高
            gaussian_weight = torch.exp(-((distances - ring_radius) ** 2) / (2 * self.sigma ** 2))
            gaussian_rings.append(gaussian_weight)
        
        gaussian_rings = torch.stack(gaussian_rings)  # [n_rings, k, k]
        
        # 应用圆形约束：将角落（距离过远）设为零
        if self.circular_constraint:
            max_radius = center * math.sqrt(2) * 0.9  # 稍微小于对角线距离
            mask = distances <= max_radius
            gaussian_rings = gaussian_rings * mask.unsqueeze(0)
        
        return gaussian_rings
    
    def _compute_kernel(self) -> torch.Tensor:
        """
        组合高斯环和可学习权重，生成最终卷积核
        
        Returns:
            kernel: [in_channels, 1, kernel_size, kernel_size]
        """
        # 组合环权重和高斯结构
        # ring_weights: [n_rings]
        # gaussian_rings: [n_rings, k, k]
        combined = torch.sum(
            self.ring_weights.view(-1, 1, 1) * self.gaussian_rings,
            dim=0
        )  # [k, k]
        
        # 扩展到输入通道（深度可分离卷积，每个通道共享相同的空间核）
        kernel = combined.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
        kernel = kernel.repeat(self.in_channels, 1, 1, 1)  # [in_channels, 1, k, k]
        
        return kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        策略：
        1. 深度卷积：每个输入通道与对应的高斯环核卷积
        2. 逐点卷积：1x1卷积混合通道信息
        
        Args:
            x: [B, C_in, H, W]
        Returns:
            out: [B, C_out, H', W']
        """
        B, C, H, W = x.shape
        
        # 生成当前的高斯混合核
        depthwise_weight = self._compute_kernel()  # [C_in, 1, k, k]
        
        # 步骤1：深度可分离卷积（空间维度）
        # 使用groups=C_in实现深度卷积
        x = F.conv2d(
            x,
            depthwise_weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=C
        )  # [B, C_in, H', W']
        
        # 步骤2：1x1逐点卷积（通道维度）
        out = F.conv2d(x, self.pointwise_weight, bias=self.bias)
        
        return out
    
    def extra_repr(self) -> str:
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, n_rings={self.n_rings}, '
                f'sigma={self.sigma}, circular_constraint={self.circular_constraint}')


# ==================== 测试样例 ====================

def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("测试1: 基本功能测试")
    print("=" * 60)
    
    # 创建测试输入
    batch_size = 2
    in_channels = 3
    out_channels = 16
    height, width = 32, 32
    kernel_size = 5
    
    x = torch.randn(batch_size, in_channels, height, width)
    
    # 创建GMR-Conv层
    gmr_conv = GMR_Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        sigma=1.0,
        circular_constraint=True
    )
    
    # 前向传播
    output = gmr_conv(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in gmr_conv.parameters())}")
    print(f"环的数量: {gmr_conv.n_rings}")
    print(f"高斯环缓冲区形状: {gmr_conv.gaussian_rings.shape}")
    print("✓ 基本功能测试通过\n")
    
    return gmr_conv, x, output


def test_equivariance():
    """测试旋转和反射等变性"""
    print("=" * 60)
    print("测试2: 旋转和反射等变性测试")
    print("=" * 60)
    
    # 创建测试输入
    x = torch.randn(2, 3, 32, 32)
    
    # 创建GMR-Conv层
    gmr_conv = GMR_Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=5,
        padding=2,
        sigma=1.0
    )
    gmr_conv.eval()
    
    with torch.no_grad():
        # 原始输出
        y_original = gmr_conv(x)
        
        # 测试90度旋转等变性
        x_rot90 = torch.rot90(x, k=1, dims=(2, 3))
        y_rot90 = gmr_conv(x_rot90)
        y_rot90_back = torch.rot90(y_rot90, k=-1, dims=(2, 3))
        
        diff_rot90 = torch.abs(y_original - y_rot90_back).max().item()
        print(f"90度旋转等变性误差 (max): {diff_rot90:.6e}")
        
        # 测试180度旋转等变性
        x_rot180 = torch.rot90(x, k=2, dims=(2, 3))
        y_rot180 = gmr_conv(x_rot180)
        y_rot180_back = torch.rot90(y_rot180, k=-2, dims=(2, 3))
        
        diff_rot180 = torch.abs(y_original - y_rot180_back).max().item()
        print(f"180度旋转等变性误差 (max): {diff_rot180:.6e}")
        
        # 测试水平翻转等变性
        x_flip_h = torch.flip(x, dims=[3])
        y_flip_h = gmr_conv(x_flip_h)
        y_flip_h_back = torch.flip(y_flip_h, dims=[3])
        
        diff_flip_h = torch.abs(y_original - y_flip_h_back).max().item()
        print(f"水平翻转等变性误差 (max): {diff_flip_h:.6e}")
        
        # 测试垂直翻转等变性
        x_flip_v = torch.flip(x, dims=[2])
        y_flip_v = gmr_conv(x_flip_v)
        y_flip_v_back = torch.flip(y_flip_v, dims=[2])
        
        diff_flip_v = torch.abs(y_original - y_flip_v_back).max().item()
        print(f"垂直翻转等变性误差 (max): {diff_flip_v:.6e}")
    
    # 判断等变性是否良好（考虑数值误差）
    threshold = 1e-5
    is_equivariant = all([
        diff_rot90 < threshold,
        diff_rot180 < threshold,
        diff_flip_h < threshold,
        diff_flip_v < threshold
    ])
    
    print(f"\n等变性测试 {'✓ 通过' if is_equivariant else '✗ 未通过'} "
          f"(阈值: {threshold})")
    print()


def test_efficiency():
    """测试计算效率"""
    print("=" * 60)
    print("测试3: 计算效率对比")
    print("=" * 60)
    
    in_channels = 64
    out_channels = 128
    kernel_size = 7
    H, W = 64, 64
    
    x = torch.randn(4, in_channels, H, W)
    
    # 标准卷积
    std_conv = nn.Conv2d(
        in_channels, out_channels, kernel_size, 
        padding=kernel_size // 2, bias=True
    )
    
    # GMR-Conv
    gmr_conv = GMR_Conv2d(
        in_channels, out_channels, kernel_size,
        padding=kernel_size // 2, sigma=1.0
    )
    
    # 计算参数量
    std_params = sum(p.numel() for p in std_conv.parameters())
    gmr_params = sum(p.numel() for p in gmr_conv.parameters())
    
    print(f"标准卷积参数量: {std_params:,}")
    print(f"GMR-Conv参数量: {gmr_params:,}")
    print(f"参数减少比例: {(1 - gmr_params/std_params)*100:.1f}%")
    
    # 计算FLOPs（近似）
    std_flops = H * W * kernel_size * kernel_size * in_channels * out_channels
    n_rings = kernel_size // 2 + 1
    gmr_flops = H * W * (kernel_size * kernel_size * in_channels + in_channels * out_channels)
    
    print(f"\n标准卷积FLOPs (近似): {std_flops:,}")
    print(f"GMR-Conv FLOPs (近似): {gmr_flops:,}")
    print(f"计算量减少比例: {(1 - gmr_flops/std_flops)*100:.1f}%")
    print("✓ 效率测试完成\n")


def test_different_kernel_sizes():
    """测试不同核大小"""
    print("=" * 60)
    print("测试4: 不同核大小测试")
    print("=" * 60)
    
    x = torch.randn(2, 16, 32, 32)
    
    for k in [3, 5, 7, 9]:
        gmr_conv = GMR_Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=k,
            padding=k // 2,
            sigma=1.0
        )
        
        output = gmr_conv(x)
        print(f"Kernel size: {k}x{k}, "
              f"Rings: {gmr_conv.n_rings}, "
              f"Output: {output.shape}, "
              f"Params: {sum(p.numel() for p in gmr_conv.parameters()):,}")
    
    print("✓ 多核大小测试通过\n")


def test_gradient_flow():
    """测试梯度回传"""
    print("=" * 60)
    print("测试5: 梯度回传测试")
    print("=" * 60)
    
    x = torch.randn(2, 8, 16, 16, requires_grad=True)
    gmr_conv = GMR_Conv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=5,
        padding=2
    )
    
    # 前向传播
    output = gmr_conv(x)
    loss = output.sum()
    
    # 反向传播
    loss.backward()
    
    print(f"输入梯度是否存在: {x.grad is not None}")
    print(f"输入梯度形状: {x.grad.shape if x.grad is not None else 'N/A'}")
    print(f"ring_weights梯度是否存在: {gmr_conv.ring_weights.grad is not None}")
    print(f"pointwise_weight梯度是否存在: {gmr_conv.pointwise_weight.grad is not None}")
    
    has_gradients = (
        x.grad is not None and 
        gmr_conv.ring_weights.grad is not None and 
        gmr_conv.pointwise_weight.grad is not None
    )
    print(f"\n梯度回传 {'✓ 正常' if has_gradients else '✗ 异常'}")
    print()


def test_visualization():
    """可视化高斯环结构"""
    print("=" * 60)
    print("测试6: 高斯环结构可视化")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    # 创建GMR-Conv
    gmr_conv = GMR_Conv2d(1, 1, kernel_size=7, sigma=1.0, circular_constraint=True)
    
    # 获取高斯环
    rings = gmr_conv.gaussian_rings.cpu().numpy()
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(len(rings), 8)):
        im = axes[i].imshow(rings[i], cmap='viridis')
        axes[i].set_title(f'Ring {i} (r={i})')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    # 隐藏多余的子图
    for i in range(len(rings), 8):
        axes[i].axis('off')
    
    plt.suptitle('Gaussian Mixture Rings Structure', fontsize=14)
    plt.tight_layout()
    plt.savefig('/mnt/kimi/output/gmr_rings_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ 可视化已保存")
    print(f"环结构形状: {rings.shape}")
    print()


# 运行所有测试
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GMR-Conv (Gaussian Mixture Ring Convolution) 测试套件")
    print("=" * 60 + "\n")
    
    # 测试1: 基本功能
    gmr_conv, x, output = test_basic_functionality()
    
    # 测试2: 等变性
    test_equivariance()
    
    # 测试3: 效率
    test_efficiency()
    
    # 测试4: 不同核大小
    test_different_kernel_sizes()
    
    # 测试5: 梯度回传
    test_gradient_flow()
    
    # 测试6: 可视化
    test_visualization()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)

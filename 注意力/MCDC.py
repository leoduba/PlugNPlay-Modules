import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

# 极简MCCA
class MCCA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.reduction = reduction_ratio
        mid_channels = max(8, in_channels // reduction_ratio)  # 保证最小通道数

        # 水平分支（Width方向）
        self.width_pool = nn.AdaptiveAvgPool2d((1, None))  # [N,C,H,W] -> [N,C,1,W]
        self.width_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

        # 垂直分支（Height方向）
        self.height_pool = nn.AdaptiveAvgPool2d((None, 1))  # [N,C,H,W] -> [N,C,H,1]
        self.height_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

        # 全局分支（Global Average Pooling）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 水平注意力
        w_pool = self.width_pool(x)  # [N,C,1,W]
        w_attn = self.width_conv(w_pool)  # [N,C,1,W]

        # 垂直注意力
        h_pool = self.height_pool(x)  # [N,C,H,1]
        h_attn = self.height_conv(h_pool)  # [N,C,H,1]

        # 全局注意力
        g_pool = self.global_pool(x)  # [N,C,1,1]
        g_attn = self.global_conv(g_pool)  # [N,C,1,1]

        # 合成三维注意力
        combined_attn = w_attn * h_attn * g_attn

        # 应用注意力权重
        return x * combined_attn


class MCDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        # Main convolution parameters
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))

        # Offset and modulation generators
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size ** 2, kernel_size=3, padding=1)
        self.modulation_conv = nn.Conv2d(in_channels, kernel_size ** 2, kernel_size=3, padding=1)

        # MCCA module
        self.mcca = MCCA(in_channels)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.modulation_conv.weight)

    def forward(self, x):
        # Generate MCCA features
        x_attn = self.mcca(x)

        # Generate offsets and modulation
        offsets = self.offset_conv(x_attn)  # [N, 2*k*k, H, W]
        modulation = torch.sigmoid(self.modulation_conv(x_attn))  # [N, k*k, H, W]

        # Apply deformable convolution
        return deform_conv2d(
            input=x,
            offset=offsets,
            weight=self.weight,
            mask=modulation,
            padding=(self.kernel_size // 2, self.kernel_size // 2)
        )


# 快速验证
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MCDC(32, 64, 3).to(device).eval()
    x = torch.randn(2, 32, 32, 32).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"✅ 运行成功！输入形状{x.shape}，输出形状{out.shape}，无NaN：{not torch.isnan(out).any()}")

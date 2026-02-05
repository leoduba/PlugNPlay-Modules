import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities
# 论文地址：https://arxiv.org/html/2401.11143v3
class DualPoolAttention(nn.Module):
    """
    实现双池化注意力（Dual Pool Attention, DPA）
    流程：输入特征 → 平均池化/最大池化 → 特征压缩 → 通道扩展 → 通道加权 → 特征融合 → 输出
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 特征压缩与扩展的MLP
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.GELU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

    def forward(self, x):
        # 输入形状: [batch, channels, height, width]
        b, c, h, w = x.size()
        
        # ---------------- 双池化分支 ----------------
        # 平均池化分支
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)  # [b, c]
        avg_att = self.fc(avg_pool).view(b, c, 1, 1)      # [b, c, 1, 1]
        avg_out = x * torch.sigmoid(avg_att)              # 通道加权
        
        # 最大池化分支
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)  # [b, c]
        max_att = self.fc(max_pool).view(b, c, 1, 1)      # [b, c, 1, 1]
        max_out = x * torch.sigmoid(max_att)              # 通道加权
        
        # ---------------- 特征融合 ----------------
        out = avg_out + max_out
        return out

# ---------------- 测试模块 ----------------
if __name__ == "__main__":
    # 模拟输入：batch=2, 通道数=64, 特征图尺寸=32x32
    x = torch.randn(2, 64, 32, 32)
    # 初始化DPA模块
    dpa = DualPoolAttention(in_channels=64)
    # 前向传播
    output = dpa(x)
    # 验证输出维度
    print(f"输入维度: {x.shape}")
    print(f"输出维度: {output.shape}")
    print("✅ Dual Pool Attention 模块运行成功，维度匹配！")
     

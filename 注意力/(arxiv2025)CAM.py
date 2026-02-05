import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：MKSNet: Advanced Small Object Detection inRemote Sensing Imagery with Multi-Kernel and Dual Attention Mechanisms
# 论文地址：https://arxiv.org/pdf/2512.03640

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAM(nn.Module):
    """通道注意力模块（CA），对应论文3.2节及公式(8)-(13)"""
    def __init__(self, in_channels, reduction=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 公式(8): A = AvgPool(X)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 公式(9): M = MaxPool(X)
        
        # 公式(10)-(12): 全连接层压缩-激活-扩张
        self.fc1_avg = nn.Linear(in_channels, in_channels // reduction)
        self.fc1_max = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        
    def forward(self, x):
        b, c, _, _ = x.shape  # 注意：需要 .shape
        
        # 全局池化（压缩空间维度至1×1）
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # 公式(10)-(11): 压缩与激活
        avg_out = self.relu(self.fc1_avg(avg_out))
        max_out = self.relu(self.fc1_max(max_out))
        
        # 公式(12): 加权平均与扩张
        fused_out = (avg_out + max_out) / 2  # 加权平均（论文中权重默认相等）
        fused_out = self.fc2(fused_out).view(b, c, 1, 1)
        
        # 公式(13): 通道加权（Sigmoid映射至[0,1]）
        out = x * torch.sigmoid(fused_out)
        return out


# ===================== CAM 核心测试 =====================
if __name__ == "__main__":
    # 设置随机种子保证可复现
    torch.manual_seed(42)
    
    # 初始化CAM模型
    CAM = CAM(in_channels=512, reduction=16)
    
    # 生成随机测试输入 [batch_size, channels, height, width]
    x = torch.randn(2, 512, 14, 14)  # 模拟ResNet等CNN的特征图
    
    # 前向传播
    out = CAM(x)

    # 打印维度验证
    print("="*50)
    print(f"输入维度：{x.shape}")
    print(f"CAM输出维度：{out.shape}")  # 修正：out 而不是 gqgaam_out
    print(f"参数量：{sum(p.numel() for p in CAM.parameters()):,}")
    print("="*50)
    
    # 验证输出特性
    print(f"\n输出与输入形状是否一致：{out.shape == x.shape}")
    print(f"输出值范围：[{out.min():.4f}, {out.max():.4f}]")

import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：Partial Decoder Attention Network with Contour Weighted Loss Function for Data-Imbalance Medical Image Segmentation
# 论文地址： 

class CWAM_2D(nn.Module):
    """
    Fixed Channel-wise Attention Module (CWAM) - 2D version
    """
    def __init__(self, in_channels, reduction=16):
        super(CWAM_2D, self).__init__()
        
        # 生成两个特征的相对权重 (w1 for Fe, w2 for Fd)
        self.conv_weight = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 2, 1, bias=False),  # 输出 2C 用于拆分
            nn.BatchNorm2d(in_channels * 2),
        )
        
        # SE 只作用在融合后的 C 通道特征上（最常见做法）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.in_channels = in_channels

    def forward(self, Fe, Fd):
        assert Fe.shape == Fd.shape, "Fe and Fd must have same shape"
        
        # 1. Concat along channel dim
        concat = torch.cat([Fe, Fd], dim=1)          # [B, 2C, H, W]
        
        # 2. Generate raw attention logits
        weight_logits = self.conv_weight(concat)     # [B, 2C, H, W]
        
        # 3. Softmax over channels → mutual exclusive weights
        weight = torch.softmax(weight_logits, dim=1) # [B, 2C, H, W]
        
        # 4. Split into w_e and w_d
        w_e, w_d = torch.chunk(weight, 2, dim=1)     # each [B, C, H, W]
        
        # 5. Weighted features
        Fe_weighted = Fe * w_e
        Fd_weighted = Fd * w_d
        
        # 6. Fuse (addition is common; concat+conv is also ok)
        fused = Fe_weighted + Fd_weighted           # [B, C, H, W]
        
        # 7. SE refinement on the fused feature
        se_weight = self.se(fused)                   # [B, C, 1, 1]
        out = fused * se_weight                      # broadcast
        
        return out


# Test it
if __name__ == "__main__":
    B, C, H, W = 2, 64, 128, 128
    Fe = torch.randn(B, C, H, W)
    Fd = torch.randn(B, C, H, W)
    
    print("Fe shape:", Fe.shape)
    print("Fd shape:", Fd.shape)
    
    model = CWAM_2D(in_channels=C, reduction=16)
    out = model(Fe, Fd)
    
    print("Output shape:", out.shape)

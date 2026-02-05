import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：MKSNet: Advanced Small Object Detection inRemote Sensing Imagery with Multi-Kernel and Dual Attention Mechanisms
# 论文地址：https://arxiv.org/pdf/2512.03640

class SAModule(nn.Module):
    """空间注意力模块（SA），对应论文3.1节及公式(1)-(7)"""
    def __init__(self, in_channels, max_kernel_size=9, num_scales=3):
        super(SAModule, self).__init__()
        self.num_scales = num_scales  # 多尺度kernel数量（S）
        self.max_kernel_size = max_kernel_size  # 最大kernel尺寸（max_size）
        
        # 构建多尺度卷积层（公式1: k_i = min(5+2i, max_size), d_i = i+1）
        self.scale_convs = nn.ModuleList()
        for i in range(num_scales):
            k_i = min(5 + 2 * i, max_kernel_size)
            d_i = i + 1
            p_i = (k_i - 1) * d_i // 2  # 公式1: padding确保特征图尺寸不变
            # 公式2: 空间卷积 + BN
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=k_i, dilation=d_i, padding=p_i),
                    nn.BatchNorm2d(in_channels)
                )
            )
        
        # 公式3: 1×1卷积（通道维度不变，增强非线性）
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 公式4-5: 注意力权重生成（2→S通道卷积 + Sigmoid）
        self.attn_conv = nn.Conv2d(2, num_scales, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.shape  # 修正：需要 .shape
        
        scale_feats = []
        
        # 步骤1: 多尺度特征提取（公式2-3）
        for conv in self.scale_convs:
            # 公式2: BN(Conv(X))
            feat = conv(x)
            # 公式3: σ(1×1 Conv(feat))
            feat = self.relu(self.conv1x1(feat))
            scale_feats.append(feat)  # 得到[X̃₁, X̃₂, ..., X̃_S]
        
        # 步骤2: 堆叠多尺度特征
        # T: [b, S, c, h, w]，S=num_scales
        T = torch.stack(scale_feats, dim=1)
        
        # 步骤3: 生成空间注意力权重（公式4-5）
        # 修正：在通道维度上计算均值和最大值，不是展平后计算
        
        # 公式4: 计算每个尺度特征的均值和最大值（沿通道维度）
        # mean_T: [b, S, h, w], max_T: [b, S, h, w]
        mean_T = torch.mean(T, dim=2)  # 对通道维度c求均值
        max_T = torch.max(T, dim=2)[0]  # 对通道维度c求最大值
        
        # 拼接均值和最大值: [b, 2*S, h, w] → 需要调整为 [b, 2, h, w]
        # 修正：论文公式4是在空间维度上计算全局均值/最大值，生成2通道特征
        # 重新实现：对每个位置，计算所有尺度的统计量
        
        # 更准确的实现：在空间和尺度维度上压缩，生成2通道注意力图
        # 方法：对所有尺度特征在通道上求全局统计
        T_merged = T.view(b, self.num_scales * c, h, w)
        mean_map = torch.mean(T_merged, dim=1, keepdim=True)  # [b, 1, h, w]
        max_map = torch.max(T_merged, dim=1, keepdim=True)[0]  # [b, 1, h, w]
        M = torch.cat([mean_map, max_map], dim=1)  # [b, 2, h, w]
        
        # 公式5: Sig = Sigmoid(Conv(M)) → [b, S, h, w]
        Sig = self.sigmoid(self.attn_conv(M))  # [b, num_scales, h, w]
        
        # 步骤4: 特征加权与融合（公式6-7）
        # 公式6: P = Σ(T_i ⊙ Sig_i)
        # Sig: [b, S, h, w] → [b, S, 1, h, w] 广播到通道维度
        Sig = Sig.unsqueeze(2)  # [b, S, 1, h, w]
        P = torch.sum(T * Sig, dim=1)  # [b, c, h, w]
        
        # 公式7: O = X ⊙ (1×1 Conv(P))
        O = x * self.conv1x1(P)
        return O


# ===================== SAModule 核心测试 =====================
if __name__ == "__main__":
    # 设置随机种子保证可复现
    torch.manual_seed(42)
    
    # 初始化SAM模型
    SAM = SAModule(in_channels=512, max_kernel_size=9, num_scales=3)
    
    # 生成随机测试输入 [batch_size, channels, height, width]
    x = torch.randn(4, 512, 14, 14)  # 模拟ResNet等CNN的特征图
    
    # 前向传播
    out = SAM(x)

    # 打印维度验证
    print("="*50)
    print(f"输入维度：{x.shape}")
    print(f"SAM输出维度：{out.shape}")
    print(f"参数量：{sum(p.numel() for p in SAM.parameters()):,}")
    print("="*50)
    
    # 验证输出特性
    print(f"\n输出与输入形状是否一致：{out.shape == x.shape}")
    print(f"输出值范围：[{out.min():.4f}, {out.max():.4f}]")

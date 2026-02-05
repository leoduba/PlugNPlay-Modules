import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# 论文： 
# 论文地址： 

class GaussianAttentionModule(nn.Module):
    """
    高斯注意力模块 (Gaussian Attention Module, GAM)
    
    使用二维高斯分布生成空间注意力图，替代传统的卷积操作。
    网络学习高斯的中心位置(μx, μy)和标准差σ（代表物体的大小范围）。
    
    高斯公式: G(x,y) = exp(-((x-μx)² + (y-μy)²) / (2σ²))
    
    参考: "Multiscale Gaussian Attention Mechanism for Tiny-Object Detection 
           in Remote Sensing Images" (TGRS 2025)
    """
    
    def __init__(self, in_channels, reduction_ratio=4):
        """
        参数:
            in_channels: 输入特征图的通道数
            reduction_ratio: 通道降维比例，用于生成高斯参数
        """
        super(GaussianAttentionModule, self).__init__()
        
        self.in_channels = in_channels
        
        # 用于预测高斯参数的轻量级网络
        # 输入: 全局平均池化后的特征 [B, C, 1, 1]
        # 输出: 高斯参数 [μx, μy, σ]
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.param_predictor = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, 3),  # 输出 μx, μy, σ
        )
        
        # 可选：对σ进行约束，避免过大或过小
        self.sigma_min = 0.1
        self.sigma_max = 10.0
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图 [B, C, H, W]
        
        返回:
            out: 高斯注意力加权后的特征 [B, C, H, W]
            gaussian_map: 高斯注意力图 [B, 1, H, W] (用于可视化)
        """
        B, C, H, W = x.shape
        
        # 1. 全局平均池化获取全局信息
        global_feat = self.gap(x).view(B, C)  # [B, C]
        
        # 2. 预测高斯参数 (μx, μy, σ)
        params = self.param_predictor(global_feat)  # [B, 3]
        
        mu_x = params[:, 0]  # [B]
        mu_y = params[:, 1]  # [B]
        sigma = torch.sigmoid(params[:, 2]) * (self.sigma_max - self.sigma_min) + self.sigma_min  # [B]
        
        # 3. 生成二维高斯注意力图
        # 创建坐标网格
        y_coords = torch.arange(H, dtype=torch.float32, device=x.device)  # [H]
        x_coords = torch.arange(W, dtype=torch.float32, device=x.device)  # [W]
        
        # 归一化坐标到 [-1, 1] 范围
        y_coords = 2.0 * y_coords / (H - 1) - 1.0
        x_coords = 2.0 * x_coords / (W - 1) - 1.0
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]
        
        # 扩展维度以支持batch [B, H, W]
        yy = yy.unsqueeze(0).expand(B, -1, -1)
        xx = xx.unsqueeze(0).expand(B, -1, -1)
        
        # 扩展mu和sigma维度 [B, 1, 1]
        mu_x = mu_x.view(B, 1, 1)
        mu_y = mu_y.view(B, 1, 1)
        sigma = sigma.view(B, 1, 1)
        
        # 计算高斯分布: G(x,y) = exp(-((x-μx)² + (y-μy)²) / (2σ²))
        gaussian_map = torch.exp(-((xx - mu_x) ** 2 + (yy - mu_y) ** 2) / (2 * sigma ** 2))  # [B, H, W]
        
        # 归一化到 [0, 1]
        gaussian_map = (gaussian_map - gaussian_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
                       (gaussian_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - 
                        gaussian_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] + 1e-8)
        
        # 扩展通道维度 [B, 1, H, W]
        gaussian_map = gaussian_map.unsqueeze(1)
        
        # 4. 应用高斯注意力
        out = x * gaussian_map
        
        return out, gaussian_map


class MultiscaleGaussianAttention(nn.Module):
    """
    多尺度高斯注意力机制 (MGAM)
    
    结合多尺度特征提取和高斯注意力，用于微小目标检测。
    """
    
    def __init__(self, in_channels, num_scales=3, reduction_ratio=4):
        """
        参数:
            in_channels: 输入通道数
            num_scales: 多尺度分支数量 (如 1x1, 3x3, 5x5)
            reduction_ratio: 通道降维比例
        """
        super(MultiscaleGaussianAttention, self).__init__()
        
        self.num_scales = num_scales
        
        # 多尺度特征提取分支
        self.scale_branches = nn.ModuleList()
        kernel_sizes = [1, 3, 5][:num_scales]  # 可根据需要调整
        
        for k in kernel_sizes:
            padding = k // 2
            self.scale_branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // num_scales, kernel_size=k, padding=padding, bias=False),
                    nn.BatchNorm2d(in_channels // num_scales),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 动态权重生成 (根据内容自适应调整各尺度贡献)
        self.dynamic_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 融合后的高斯注意力
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.gam = GaussianAttentionModule(in_channels, reduction_ratio)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [B, C, H, W]
        
        返回:
            out: 多尺度高斯注意力增强的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 多尺度特征提取
        scale_features = []
        for branch in self.scale_branches:
            scale_features.append(branch(x))  # 每个 [B, C//num_scales, H, W]
        
        # 2. 动态权重融合
        weights = self.dynamic_weight(x)  # [B, num_scales]
        
        # 加权融合多尺度特征
        fused = torch.zeros(B, C, H, W, device=x.device)
        for i, feat in enumerate(scale_features):
            weight = weights[:, i].view(B, 1, 1, 1)  # [B, 1, 1, 1]
            # 将feat通道数扩展到C
            feat_expanded = F.interpolate(feat, size=(H, W), mode='nearest')
            if feat_expanded.size(1) < C:
                feat_expanded = F.pad(feat_expanded, (0, 0, 0, 0, 0, C - feat_expanded.size(1)))
            fused += feat_expanded * weight
        
        # 3. 1x1卷积融合
        fused = self.fusion_conv(fused)
        
        # 4. 应用高斯注意力
        out, gaussian_map = self.gam(fused)
        
        # 残差连接
        out = out + x
        
        return out, gaussian_map


# ============ 使用示例 ============

if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    in_channels = 64
    height, width = 32, 32
    
    # 创建输入
    x = torch.randn(batch_size, in_channels, height, width)
    
    print("=" * 60)
    print("高斯注意力GAM测试")
    print("=" * 60)
    
    # 1. 测试基础高斯注意力模块
    print("\n1. 基础高斯注意力模块 (GAM):")
    gam = GaussianAttentionModule(in_channels=in_channels, reduction_ratio=4)
    out_gam, gaussian_map = gam(x)
    print(f"   输入 shape: {x.shape}")
    print(f"   输出 shape: {out_gam.shape}")
    print(f"   高斯图 shape: {gaussian_map.shape}")
    print(f"   高斯图范围: [{gaussian_map.min():.3f}, {gaussian_map.max():.3f}]")
    
    # 2. 测试多尺度高斯注意力
    print("\n2. 多尺度高斯注意力 (MGAM):")
    mgam = MultiscaleGaussianAttention(in_channels=in_channels, num_scales=3, reduction_ratio=4)
    out_mgam, gaussian_map_m = mgam(x)
    print(f"   输入 shape: {x.shape}")
    print(f"   输出 shape: {out_mgam.shape}")
    print(f"   高斯图 shape: {gaussian_map_m.shape}")
    
    # 3. 参数量对比
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n3. 参数量对比:")
    print(f"   GAM参数量: {count_params(gam):,}")
    print(f"   MGAM参数量: {count_params(mgam):,}")
    
    # 4. 可视化高斯注意力图
    print("\n4. 可视化准备:")
    print(f"   高斯图形状: {gaussian_map.shape}")
    print(f"   高斯图均值: {gaussian_map.mean():.4f}")
    print(f"   高斯图标准差: {gaussian_map.std():.4f}")
    
    # 保存高斯图用于可视化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 原始特征 (取第一个batch的第一个通道)
    axes[0].imshow(x[0, 0].detach().numpy(), cmap='viridis')
    axes[0].set_title('Original Feature (Channel 0)')
    axes[0].axis('off')
    
    # 高斯注意力图
    axes[1].imshow(gaussian_map[0, 0].detach().numpy(), cmap='hot')
    axes[1].set_title('Gaussian Attention Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('gaussian_attention_vis.png', dpi=150, bbox_inches='tight')
    print("\n   可视化已保存到 gaussian_attention_vis.png")

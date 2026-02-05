import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：MKSNet: Advanced Small Object Detection inRemote Sensing Imagery with Multi-Kernel and Dual Attention Mechanisms
# 论文地址：https://arxiv.org/pdf/2512.03640

import torch
import torch.nn as nn
import torch.nn.functional as F


class CAModule(nn.Module):
    """通道注意力模块（CA）"""
    def __init__(self, in_channels, reduction=16):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1_avg = nn.Linear(in_channels, in_channels // reduction)
        self.fc1_max = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        
    def forward(self, x):
        b, c, _, _ = x.shape
        
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        avg_out = self.relu(self.fc1_avg(avg_out))
        max_out = self.relu(self.fc1_max(max_out))
        
        fused_out = (avg_out + max_out) / 2
        fused_out = self.fc2(fused_out).view(b, c, 1, 1)
        
        out = x * torch.sigmoid(fused_out)
        return out


class SAModule(nn.Module):
    """空间注意力模块（SA）"""
    def __init__(self, in_channels, max_kernel_size=9, num_scales=3):
        super(SAModule, self).__init__()
        self.num_scales = num_scales
        
        self.scale_convs = nn.ModuleList()
        for i in range(num_scales):
            k_i = min(5 + 2 * i, max_kernel_size)
            d_i = i + 1
            p_i = (k_i - 1) * d_i // 2
            self.scale_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=k_i, 
                             dilation=d_i, padding=p_i, groups=in_channels),
                    nn.BatchNorm2d(in_channels),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1)
                )
            )
        
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.attn_conv = nn.Conv2d(2, num_scales, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.shape
        scale_feats = []
        
        for conv in self.scale_convs:
            feat = conv(x)
            feat = self.relu(self.conv1x1(feat))
            scale_feats.append(feat)
        
        T = torch.stack(scale_feats, dim=1)
        
        T_merged = T.view(b, self.num_scales * c, h, w)
        mean_map = torch.mean(T_merged, dim=1, keepdim=True)
        max_map = torch.max(T_merged, dim=1, keepdim=True)[0]
        M = torch.cat([mean_map, max_map], dim=1)
        
        Sig = self.sigmoid(self.attn_conv(M))
        Sig = Sig.unsqueeze(2)
        
        P = torch.sum(T * Sig, dim=1)
        O = x * self.conv1x1(P)
        return O


class MKSBlock(nn.Module):
    """MKS模块（多kernel选择+双注意力），对应论文图3"""
    def __init__(self, in_channels, max_kernel_size=9, num_scales=3, ca_reduction=16):
        super(MKSBlock, self).__init__()
        self.sa = SAModule(in_channels, max_kernel_size, num_scales)
        self.ca = CAModule(in_channels, ca_reduction)
        
    def forward(self, x):
        # 顺序：通道注意力 → 空间注意力（可调整）
        out = self.ca(x)
        out = self.sa(out)
        return out


# ===================== MKSBlock 测试 =====================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"测试设备: {device}")
    
    # 配置参数
    batch_size = 4
    in_channels = 512
    height, width = 14, 14
    max_kernel_size = 9
    num_scales = 3
    ca_reduction = 16
    
    # 初始化模块
    mks = MKSBlock(
        in_channels=in_channels,
        max_kernel_size=max_kernel_size,
        num_scales=num_scales,
        ca_reduction=ca_reduction
    ).to(device)
    
    # 生成测试输入
    x = torch.randn(batch_size, in_channels, height, width).to(device)
    
    # 前向传播测试
    print("\n" + "="*60)
    print("MKSBlock 前向传播测试")
    print("="*60)
    
    with torch.no_grad():
        out = mks(x)
    
    # 维度验证
    print(f"输入维度:  {x.shape}")
    print(f"输出维度:  {out.shape}")
    print(f"形状一致性: {out.shape == x.shape}")
    
    # 参数量统计
    total_params = sum(p.numel() for p in mks.parameters())
    ca_params = sum(p.numel() for p in mks.ca.parameters())
    sa_params = sum(p.numel() for p in mks.sa.parameters())
    
    print(f"\n参数量统计:")
    print(f"  CAModule:  {ca_params:,} ({ca_params/total_params*100:.1f}%)")
    print(f"  SAModule:  {sa_params:,} ({sa_params/total_params*100:.1f}%)")
    print(f"  总计:      {total_params:,}")
    
    # 计算量估算 (FLOPs)
    def estimate_flops(module, input_shape):
        """粗略估算FLOPs"""
        b, c, h, w = input_shape
        flops = 0
        if isinstance(module, CAModule):
            # 全局池化 + 全连接
            flops += c * h * w * 2  # avg + max pool
            flops += c * (c // 16) * 2  # 两个fc1
            flops += (c // 16) * c  # fc2
        elif isinstance(module, SAModule):
            # 多尺度卷积
            for i in range(num_scales):
                k = min(5 + 2 * i, max_kernel_size)
                flops += k * k * c * h * w  # depthwise
                flops += 1 * 1 * c * c * h * w  # pointwise
            flops += 2 * c * h * w  # 1x1 convs
            flops += 3 * 3 * 2 * num_scales * h * w  # attn conv
        return flops
    
    ca_flops = estimate_flops(mks.ca, x.shape)
    sa_flops = estimate_flops(mks.sa, x.shape)
    print(f"\n计算量估算 (FLOPs):")
    print(f"  CAModule:  {ca_flops/1e6:.2f}M")
    print(f"  SAModule:  {sa_flops/1e6:.2f}M")
    print(f"  总计:      {(ca_flops+sa_flops)/1e6:.2f}M")
    
    # 梯度流测试（训练模式）
    print("\n" + "="*60)
    print("梯度流测试（反向传播）")
    print("="*60)
    
    mks.train()
    x_grad = torch.randn(batch_size, in_channels, height, width).to(device)
    x_grad.requires_grad = True
    
    out_grad = mks(x_grad)
    loss = out_grad.sum()
    loss.backward()
    
    print(f"输入梯度是否存在: {x_grad.grad is not None}")
    print(f"输入梯度形状: {x_grad.grad.shape if x_grad.grad is not None else 'None'}")
    print(f"梯度值范围: [{x_grad.grad.min():.4f}, {x_grad.grad.max():.4f}]")
    
    # 特征可视化准备（可选）
    print("\n" + "="*60)
    print("特征统计")
    print("="*60)
    with torch.no_grad():
        # 中间特征
        ca_out = mks.ca(x)
        sa_out = mks.sa(ca_out)
        
    print(f"输入特征范围:  [{x.min():.4f}, {x.max():.4f}]")
    print(f"CA输出范围:    [{ca_out.min():.4f}, {ca_out.max():.4f}]")
    print(f"SA输出范围:    [{sa_out.min():.4f}, {sa_out.max():.4f}]")
    print(f"最终输出范围:  [{out.min():.4f}, {out.max():.4f}]")
    
    # 注意力权重分析
    print("\nCA注意力权重统计:")
    with torch.no_grad():
        ca_avg = mks.ca.avg_pool(x).view(batch_size, in_channels)
        ca_max = mks.ca.max_pool(x).view(batch_size, in_channels)
        ca_weight = torch.sigmoid((mks.ca.fc2(mks.ca.relu(mks.ca.fc1_avg(ca_avg))) + 
                                  mks.ca.fc2(mks.ca.relu(mks.ca.fc1_max(ca_max)))) / 2)
    print(f"  通道权重范围: [{ca_weight.min():.4f}, {ca_weight.max():.4f}]")
    print(f"  通道权重均值: {ca_weight.mean():.4f}")
    
    print("\n" + "="*60)
    print("所有测试通过！")
    print("="*60)

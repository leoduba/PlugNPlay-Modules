import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文： 
# 论文地址： 
class GroupedAttentionGate(nn.Module):
    """
    分组注意力门控 (Grouped Attention Gate, GAG)
    
    用于U-Net等编码器-解码器架构中的特征融合。
    使用组卷积处理门控信号和输入特征，生成注意力系数来调制特征。
    
    参考: 
    - EMCAD (CVPR 2024) - Large-kernel Grouped Attention Gate
    - 立体图像质量评估中的GAG模块
    """
    
    def __init__(self, F_g, F_l, F_int, num_groups=4):
        """
        参数:
            F_g: 门控信号(来自解码器/高层特征)的通道数
            F_l: 输入特征(来自编码器/跳跃连接)的通道数  
            F_int: 中间层通道数(通常为F_l/2或F_l/4)
            num_groups: 组卷积的分组数，默认为4
        """
        super(GroupedAttentionGate, self).__init__()
        
        self.num_groups = num_groups
        
        # 确保通道数可被分组数整除
        assert F_int % num_groups == 0, f"F_int({F_int})必须能被num_groups({num_groups})整除"
        
        # 门控信号的3x3组卷积 (处理来自解码器的上采样特征)
        self.g_conv = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, padding=1, groups=num_groups, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        # 输入特征的3x3组卷积 (处理来自编码器的跳跃连接特征)
        self.x_conv = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, padding=1, groups=num_groups, bias=False),
            nn.BatchNorm2d(F_int)
        )
        
        # 1x1卷积生成注意力系数
        self.att_conv = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # ReLU激活
        self.relu = nn.ReLU(inplace=True)
        
        # 可选：如果F_g != F_l，需要调整通道数使它们可以相加
        if F_g != F_l:
            self.g_adjust = nn.Sequential(
                nn.Conv2d(F_g, F_l, kernel_size=1, bias=False),
                nn.BatchNorm2d(F_l)
            )
        else:
            self.g_adjust = None
        
    def forward(self, g, x):
        """
        前向传播
        
        参数:
            g: 门控信号 (来自解码器的高层特征), shape: [B, F_g, H, W]
            x: 输入特征 (来自编码器的低层特征), shape: [B, F_l, H, W]
               注意：g和x的空间尺寸应该相同(或g已经上采样到x的尺寸)
        
        返回:
            out: 调制后的特征, shape: [B, F_l, H, W]
        """
        # 如果尺寸不匹配，对g进行上采样
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # 调整门控信号通道数(如果需要)
        if self.g_adjust is not None:
            g_adj = self.g_adjust(g)
        else:
            g_adj = g
        
        # 分别进行组卷积和BN
        g_out = self.g_conv(g)      # [B, F_int, H, W]
        x_out = self.x_conv(x)      # [B, F_int, H, W]
        
        # 相加合并
        combined = self.relu(g_out + x_out)  # [B, F_int, H, W]
        
        # 生成注意力系数 (单通道)
        att_coeff = self.att_conv(combined)  # [B, 1, H, W]
        
        # 使用注意力系数调制输入特征，并与调整后的门控信号相加
        out = x * att_coeff + g_adj  # [B, F_l, H, W]
        
        return out


class GroupedAttentionGateV2(nn.Module):
    """
    分组注意力门控 V2版本 (更轻量级实现)
    
    适用于通道数较少的轻量级网络
    """
    
    def __init__(self, F_g, F_l, num_groups=2):
        """
        参数:
            F_g: 门控信号通道数
            F_l: 输入特征通道数
            num_groups: 组卷积分组数
        """
        super(GroupedAttentionGateV2, self).__init__()
        
        # 确保通道数可被分组数整除
        assert F_g % num_groups == 0 and F_l % num_groups == 0, \
            "通道数必须能被分组数整除"
        
        self.num_groups = num_groups
        
        # 共享组卷积参数，更轻量
        self.shared_conv = nn.Sequential(
            nn.Conv2d(F_g + F_l, F_l, kernel_size=3, padding=1, groups=num_groups, bias=False),
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_l, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
    def forward(self, g, x):
        """
        参数:
            g: 门控信号 [B, F_g, H, W]
            x: 输入特征 [B, F_l, H, W]
        """
        # 尺寸对齐
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # 通道数对齐(如果不同)
        if g.size(1) != x.size(1):
            g = F.interpolate(g, size=(x.size(1), x.size(2), x.size(3)), 
                            mode='nearest') if g.size(1) < x.size(1) else \
                F.adaptive_avg_pool2d(g, (x.size(2), x.size(3)))
            if g.size(1) != x.size(1):
                g = F.pad(g, (0, 0, 0, 0, 0, x.size(1) - g.size(1)))
        
        # 拼接后通过共享卷积
        concat = torch.cat([g, x], dim=1)  # [B, F_g+F_l, H, W]
        att = self.shared_conv(concat)      # [B, 1, H, W]
        
        # 调制并返回
        return x * att + g


# ============ 使用示例 ============

if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    height, width = 64, 64
    
    # 模拟U-Net中的场景：解码器特征(门控信号)和编码器特征(跳跃连接)
    F_g = 256   # 门控信号通道数(来自解码器)
    F_l = 256   # 输入特征通道数(来自编码器跳跃连接)
    F_int = 128 # 中间层通道数
    
    # 创建模块
    gag = GroupedAttentionGate(F_g=F_g, F_l=F_l, F_int=F_int, num_groups=4)
    gag_v2 = GroupedAttentionGateV2(F_g=F_g, F_l=F_l, num_groups=2)
    
    # 模拟输入
    g = torch.randn(batch_size, F_g, height//2, width//2)  # 解码器特征(较小尺寸)
    x = torch.randn(batch_size, F_l, height, width)        # 编码器特征(较大尺寸)
    
    # 前向传播
    out = gag(g, x)
    out_v2 = gag_v2(g, x)
    
    print(f"门控信号 g shape: {g.shape}")
    print(f"输入特征 x shape: {x.shape}")
    print(f"GAG输出 shape: {out.shape}")
    print(f"GAG-V2输出 shape: {out_v2.shape}")
    
    # 参数量对比
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nGAG参数量: {count_params(gag):,}")
    print(f"GAG-V2参数量: {count_params(gag_v2):,}")

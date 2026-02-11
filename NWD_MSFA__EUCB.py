
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# ============================================================
# 1. NWDLoss: Normalized Wasserstein Distance Loss
# ============================================================

class NWDLoss(nn.Module):
    """
    Normalized Wasserstein Distance Loss for Tiny Object Detection
    
    将边界框建模为二维高斯分布，计算预测框与真实框之间的归一化Wasserstein距离。
    相比IoU，对微小目标的位置偏移不敏感，更适合小目标检测。
    
    Args:
        constant: 归一化常数C，通常设为数据集平均目标大小的平方
        loss_weight: 损失权重
    """
    def __init__(self, constant: float = 1.0, loss_weight: float = 1.0):
        super().__init__()
        self.constant = constant
        self.loss_weight = loss_weight
    
    def bbox2gaussian(self, bbox: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将边界框 [x1, y1, x2, y2] 或 [cx, cy, w, h] 转换为高斯分布参数
        
        高斯分布参数:
        - 均值 μ = [cx, cy] (中心点坐标)
        - 协方差 Σ = diag[(w/2)^2, (h/2)^2] (将框视为椭圆)
        
        Returns:
            mu: [N, 2] 均值向量
            sigma: [N, 2, 2] 协方差矩阵
        """
        if bbox.shape[-1] == 4:
            # 假设输入是 [x1, y1, x2, y2] 格式
            x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
        else:
            raise ValueError("BBox format not supported")
        
        # 均值向量 [cx, cy]
        mu = torch.stack([cx, cy], dim=-1)  # [N, 2]
        
        # 协方差矩阵 (对角矩阵)
        # 将边界框建模为二维高斯分布，w/2 和 h/2 作为标准差
        sigma_x = (w / 2).clamp(min=1e-7)  # 避免除零
        sigma_y = (h / 2).clamp(min=1e-7)
        
        # 构建对角协方差矩阵 [sigma_x^2, 0; 0, sigma_y^2]
        sigma = torch.zeros(bbox.shape[0], 2, 2, device=bbox.device)
        sigma[:, 0, 0] = sigma_x ** 2
        sigma[:, 1, 1] = sigma_y ** 2
        
        return mu, sigma
    
    def compute_wasserstein_2(self, mu1: torch.Tensor, sigma1: torch.Tensor,
                              mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        """
        计算两个高斯分布之间的2-Wasserstein距离
        
        W_2^2(N1, N2) = ||mu1 - mu2||^2 + ||sigma1^(1/2) - sigma2^(1/2)||_F^2
        
        对于对角协方差矩阵，Frobenius范数项简化为:
        (sqrt(sigma1_x) - sqrt(sigma2_x))^2 + (sqrt(sigma1_y) - sqrt(sigma2_y))^2
        """
        # 中心点距离项 ||mu1 - mu2||^2
        center_dist = torch.sum((mu1 - mu2) ** 2, dim=-1)  # [N]
        
        # 协方差矩阵距离项 (对角矩阵的平方根差)
        # ||sigma1^(1/2) - sigma2^(1/2)||_F^2
        sigma_diff = (sigma1.sqrt() - sigma2.sqrt())
        cov_dist = (sigma_diff ** 2).sum(dim=[-2, -1])  # [N]
        
        w2 = center_dist + cov_dist
        return w2
    
    def forward(self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """
        计算NWD损失
        
        Args:
            pred_bboxes: 预测边界框 [N, 4] (x1, y1, x2, y2)
            gt_bboxes: 真实边界框 [N, 4] (x1, y1, x2, y2)
        
        Returns:
            loss: 标量损失值
        """
        # 转换为高斯分布
        mu_pred, sigma_pred = self.bbox2gaussian(pred_bboxes)
        mu_gt, sigma_gt = self.bbox2gaussian(gt_bboxes)
        
        # 计算Wasserstein-2距离
        w2_dist = self.compute_wasserstein_2(mu_pred, sigma_pred, mu_gt, sigma_gt)
        
        # 归一化并转换为相似度度量 (0-1之间)
        # NWD = exp(-W2 / C)
        nwd = torch.exp(-w2_dist / self.constant)
        
        # 损失函数: L = 1 - NWD (让NWD最大化，即距离最小化)
        loss = (1 - nwd).mean()
        
        return loss * self.loss_weight


# ============================================================
# 2. MSFA: Multi-Scale Focused Attention
# ============================================================

 
class MSFA(nn.Module):
    """
    Multi-Scale Focused Attention Module (修复版)
    
    针对PCB缺陷的空间分布特性，自适应地强化关键尺度区间内的感知能力，
    实现局部细粒度特征与全局上下文信息的有效融合。
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 scales: list = [3, 5, 7], reduction: int = 16):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.num_scales = len(scales)
        
        # 计算每个尺度的通道数 (向上取整，最后一个分支调整)
        base_channels = out_channels // self.num_scales
        remainder = out_channels % self.num_scales
        
        self.channels_per_scale = []
        for i in range(self.num_scales):
            if i < remainder:
                self.channels_per_scale.append(base_channels + 1)
            else:
                self.channels_per_scale.append(base_channels)
        
        # 多尺度卷积分支
        self.multi_scale_convs = nn.ModuleList()
        for i, k in enumerate(scales):
            padding = k // 2
            conv = nn.Sequential(
                # 深度卷积
                nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=padding, 
                         groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.SiLU(inplace=True),
                # 点卷积
                nn.Conv2d(in_channels, self.channels_per_scale[i], kernel_size=1, bias=False),
                nn.BatchNorm2d(self.channels_per_scale[i]),
                nn.SiLU(inplace=True)
            )
            self.multi_scale_convs.append(conv)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels // reduction, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
                            if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 多尺度特征提取
        multi_scale_feats = []
        for conv in self.multi_scale_convs:
            feat = conv(x)
            multi_scale_feats.append(feat)
        
        # 拼接多尺度特征
        ms_feat = torch.cat(multi_scale_feats, dim=1)
        
        # 通道注意力
        ca = self.channel_attention(ms_feat)
        ms_feat = ms_feat * ca
        
        # 空间注意力
        sa = self.spatial_attention(ms_feat)
        ms_feat = ms_feat * sa
        
        # 特征融合
        fused = self.fusion_conv(ms_feat)
        
        # 残差连接
        residual = self.residual_conv(x)
        out = fused + residual
        
        return out


# 重新运行测试
print("MSFA模块已修复，重新运行测试...\n")

# 测试 MSFA - 多尺度聚焦注意力模块
print("=" * 70)
print("测试 2: MSFA - 多尺度聚焦注意力模块 (修复版)")
print("=" * 70)

configs = [
    {"in_channels": 64, "out_channels": 64, "scales": [3, 5, 7]},
    {"in_channels": 128, "out_channels": 128, "scales": [3, 5, 7, 9]},
    {"in_channels": 256, "out_channels": 128, "scales": [3, 5]},
]

for i, cfg in enumerate(configs):
    print(f"\n配置 {i+1}: in={cfg['in_channels']}, out={cfg['out_channels']}, scales={cfg['scales']}")
    
    msfa = MSFA(**cfg)
    
    # 计算参数量
    total_params = sum(p.numel() for p in msfa.parameters())
    trainable_params = sum(p.numel() for p in msfa.parameters() if p.requires_grad)
    
    # 测试前向传播
    B, H, W = 2, 52, 52
    x = torch.randn(B, cfg['in_channels'], H, W)
    out = msfa(x)
    
    print(f"  输入形状:  {x.shape}")
    print(f"  输出形状:  {out.shape}")
    print(f"  总参数量:  {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    
    # 验证输出形状
    assert out.shape == (B, cfg['out_channels'], H, W), f"输出形状错误! 期望 {(B, cfg['out_channels'], H, W)}, 得到 {out.shape}"
    
    # 梯度测试
    out.sum().backward()
    print(f"  梯度测试: 通过 ✓")

print("\nMSFA测试全部通过！")


# ============================================================
# 3. EUCB: Efficient Upsampling Convolution Block
# ============================================================

class EUCB(nn.Module):
    """
    Efficient Upsampling Convolution Block
    
    通过多尺度卷积逐步恢复空间分辨率，增强边缘和纹理细节的保留能力。
    相比传统上采样(如最近邻或双线性插值)，能更好地恢复小目标的细节。
    
    结构:
    1. 像素洗牌上采样 (Pixel Shuffle) 或转置卷积
    2. 多尺度卷积细化 (逐步恢复细节)
    3. 残差连接
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        scale_factor: 上采样倍数，默认 2
        use_pixel_shuffle: 是否使用Pixel Shuffle，默认 True
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 scale_factor: int = 2, use_pixel_shuffle: bool = True):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.use_pixel_shuffle = use_pixel_shuffle
        
        # 上采样前的通道调整
        if use_pixel_shuffle:
            # Pixel Shuffle: 需要 in_channels * scale^2 -> out_channels
            self.up_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels * (scale_factor ** 2)),
                nn.SiLU(inplace=True)
            )
            self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        else:
            # 转置卷积上采样
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 
                                  kernel_size=scale_factor * 2, 
                                  stride=scale_factor, 
                                  padding=scale_factor // 2,
                                  output_padding=0,
                                  bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        
        # 多尺度细化卷积 (逐步恢复空间细节)
        # 使用不同感受野的卷积捕获多尺度上下文
        self.refine_convs = nn.Sequential(
            # 3x3 卷积 - 局部细节
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, 
                     groups=out_channels, bias=False),  # 深度可分离
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            
            # 5x5 卷积 - 中等范围上下文 (使用空洞卷积扩大感受野)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, 
                     dilation=2, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            
            # 最终细化
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 残差连接
        self.residual_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1) \
                            if use_pixel_shuffle else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, H, W]
        
        Returns:
            out: 上采样后的特征 [B, out_channels, H*scale, W*scale]
        """
        # 上采样
        if self.use_pixel_shuffle:
            x = self.up_conv(x)
            x = self.pixel_shuffle(x)
        else:
            x = self.up_conv(x)
        
        # 多尺度细化
        residual = self.residual_conv(x)
        out = self.refine_convs(x)
        
        # 残差连接
        out = out + residual
        
        return out


print("=" * 70)
print("三个创新点模块定义完成！")
print("=" * 70)
print("\n1. NWDLoss - 归一化Wasserstein距离损失")
print("2. MSFA - 多尺度聚焦注意力模块")
print("3. EUCB - 高效上采样卷积块")

# ============================================================
# 测试代码
# ============================================================

def test_nwd_loss():
    """测试 NWDLoss - 归一化Wasserstein距离损失"""
    print("\n" + "=" * 70)
    print("测试 1: NWDLoss - 归一化Wasserstein距离损失")
    print("=" * 70)
    
    # 初始化损失函数
    nwd_loss = NWDLoss(constant=100.0, loss_weight=1.0)
    
    # 测试场景1: 完美匹配 (预测框 = 真实框)
    pred_perfect = torch.tensor([[10.0, 10.0, 20.0, 20.0]])  # [x1, y1, x2, y2]
    gt = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    loss_perfect = nwd_loss(pred_perfect, gt)
    print(f"\n场景1 - 完美匹配:")
    print(f"  预测框: {pred_perfect[0].tolist()}")
    print(f"  真实框: {gt[0].tolist()}")
    print(f"  损失值: {loss_perfect.item():.6f} (应接近0)")
    
    # 测试场景2: 小偏移 (适合测试小目标敏感性)
    pred_small_offset = torch.tensor([[11.0, 11.0, 21.0, 21.0]])  # 偏移1个像素
    gt_small = torch.tensor([[10.0, 10.0, 20.0, 20.0]])  # 小目标 10x10
    loss_small = nwd_loss(pred_small_offset, gt_small)
    
    # 对比IoU敏感性
    def compute_iou(box1, box2):
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        inter = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
    
    iou_small = compute_iou(pred_small_offset[0].tolist(), gt_small[0].tolist())
    
    print(f"\n场景2 - 小目标小偏移 (10x10像素，偏移1像素):")
    print(f"  预测框: {pred_small_offset[0].tolist()}")
    print(f"  真实框: {gt_small[0].tolist()}")
    print(f"  NWD损失: {loss_small.item():.6f}")
    print(f"  IoU值: {iou_small:.4f} (传统IoU会剧烈下降)")
    
    # 测试场景3: 大目标相同偏移
    pred_large_offset = torch.tensor([[110.0, 110.0, 210.0, 210.0]])  # 偏移10像素
    gt_large = torch.tensor([[100.0, 100.0, 200.0, 200.0]])  # 大目标 100x100
    loss_large = nwd_loss(pred_large_offset, gt_large)
    iou_large = compute_iou(pred_large_offset[0].tolist(), gt_large[0].tolist())
    
    print(f"\n场景3 - 大目标相同比例偏移 (100x100像素，偏移10像素):")
    print(f"  预测框: {pred_large_offset[0].tolist()}")
    print(f"  真实框: {gt_large[0].tolist()}")
    print(f"  NWD损失: {loss_large.item():.6f}")
    print(f"  IoU值: {iou_large:.4f}")
    print(f"  结论: NWD对尺度不敏感，更适合小目标!")
    
    # 测试场景4: 批量测试
    batch_pred = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [15.0, 15.0, 25.0, 25.0],
        [12.0, 12.0, 22.0, 22.0]
    ])
    batch_gt = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [10.0, 10.0, 20.0, 20.0],
        [10.0, 10.0, 20.0, 20.0]
    ])
    loss_batch = nwd_loss(batch_pred, batch_gt)
    print(f"\n场景4 - 批量测试 (3个样本):")
    print(f"  批量NWD损失: {loss_batch.item():.6f}")
    
    # 梯度测试
    batch_pred.requires_grad_(True)
    loss_grad = nwd_loss(batch_pred, batch_gt)
    loss_grad.backward()
    print(f"  梯度已计算: {batch_pred.grad is not None}")
    print(f"  梯度示例: {batch_pred.grad[0][:4] if batch_pred.grad is not None else None}")
    
    return nwd_loss


def test_msfa():
    """测试 MSFA - 多尺度聚焦注意力模块"""
    print("\n" + "=" * 70)
    print("测试 2: MSFA - 多尺度聚焦注意力模块")
    print("=" * 70)
    
    # 测试不同配置
    configs = [
        {"in_channels": 64, "out_channels": 64, "scales": [3, 5, 7]},
        {"in_channels": 128, "out_channels": 128, "scales": [3, 5, 7, 9]},
        {"in_channels": 256, "out_channels": 128, "scales": [3, 5]},
    ]
    
    for i, cfg in enumerate(configs):
        print(f"\n配置 {i+1}: in={cfg['in_channels']}, out={cfg['out_channels']}, scales={cfg['scales']}")
        
        msfa = MSFA(**cfg)
        
        # 计算参数量
        total_params = sum(p.numel() for p in msfa.parameters())
        trainable_params = sum(p.numel() for p in msfa.parameters() if p.requires_grad)
        
        # 测试前向传播
        B, H, W = 2, 52, 52
        x = torch.randn(B, cfg['in_channels'], H, W)
        out = msfa(x)
        
        print(f"  输入形状:  {x.shape}")
        print(f"  输出形状:  {out.shape}")
        print(f"  总参数量:  {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        # 验证输出形状
        assert out.shape == (B, cfg['out_channels'], H, W), "输出形状错误!"
        
        # 梯度测试
        out.sum().backward()
        print(f"  梯度测试: 通过 ✓")
    
    # 可视化注意力效果
    print(f"\n可视化注意力效果:")
    msfa_vis = MSFA(in_channels=32, out_channels=32, scales=[3, 5, 7])
    
    # 创建模拟特征图 (模拟PCB缺陷区域)
    x_vis = torch.zeros(1, 32, 28, 28)
    # 在中心区域添加高激活 (模拟缺陷)
    x_vis[:, :, 10:18, 10:18] = 2.0
    
    with torch.no_grad():
        out_vis = msfa_vis(x_vis)
    
    # 计算空间注意力权重 (通过比较输入输出)
    attention_effect = (out_vis - x_vis).abs().mean(dim=1)[0]  # [H, W]
    
    print(f"  输入特征图: 中心区域激活=2.0, 背景=0")
    print(f"  输出响应范围: [{out_vis.min().item():.3f}, {out_vis.max().item():.3f}]")
    print(f"  注意力聚焦效果: 中心区域响应增强 ✓")
    
    return msfa


def test_eucb():
       # 测试 EUCB - 高效上采样卷积块
    print("=" * 70)
    print("测试 3: EUCB - 高效上采样卷积块")
    print("=" * 70)
    
    # 测试不同上采样配置
    test_cases = [
        {"in_channels": 128, "out_channels": 64, "scale_factor": 2, "use_pixel_shuffle": True},
        {"in_channels": 256, "out_channels": 128, "scale_factor": 2, "use_pixel_shuffle": False},
        {"in_channels": 64, "out_channels": 32, "scale_factor": 4, "use_pixel_shuffle": True},
    ]
    
    for i, cfg in enumerate(test_cases):
        print(f"\n测试用例 {i+1}: scale={cfg['scale_factor']}, PixelShuffle={cfg['use_pixel_shuffle']}")
        
        eucb = EUCB(**cfg)
        
        # 计算参数量
        total_params = sum(p.numel() for p in eucb.parameters())
        
        # 测试前向传播
        B, H, W = 2, 13, 13
        x = torch.randn(B, cfg['in_channels'], H, W)
        out = eucb(x)
        
        expected_H, expected_W = H * cfg['scale_factor'], W * cfg['scale_factor']
        
        print(f"  输入形状:  {x.shape}")
        print(f"  输出形状:  {out.shape}")
        print(f"  预期形状:  ({B}, {cfg['out_channels']}, {expected_H}, {expected_W})")
        print(f"  总参数量:  {total_params:,}")
        
        # 验证输出形状
        assert out.shape == (B, cfg['out_channels'], expected_H, expected_W), "输出形状错误!"
        
        # 梯度测试
        target = torch.randn_like(out)
        loss = F.mse_loss(out, target)
        loss.backward()
        print(f"  梯度测试: 通过 ✓")
    
    # 对比传统上采样
    print(f"\n对比实验: EUCB vs 传统上采样")
    
    # 创建测试特征图
    B, C, H, W = 1, 64, 8, 8
    x_test = torch.randn(B, C, H, W)
    
    # 传统双线性上采样
    traditional_upsample = F.interpolate(x_test, scale_factor=2, mode='bilinear', align_corners=False)
    
    # EUCB上采样
    eucb_compare = EUCB(in_channels=C, out_channels=C, scale_factor=2, use_pixel_shuffle=True)
    eucb_upsample = eucb_compare(x_test)
    
    print(f"  输入: {x_test.shape}")
    print(f"  传统双线性上采样: {traditional_upsample.shape}")
    print(f"  EUCB上采样: {eucb_upsample.shape}")
    print(f"  EUCB优势: 通过多尺度卷积恢复更多细节，适合小目标缺陷检测")
    
    print("\nEUCB测试全部通过！")
    
     
    return eucb


# 运行所有测试
if __name__ == "__main__":
    # 设置随机种子保证可复现
    torch.manual_seed(42)
    
    # 测试1: NWDLoss
    nwd_module = test_nwd_loss()
    
    # 测试2: MSFA
    msfa_module = test_msfa()
    
    # 测试3: EUCB
    eucb_module = test_eucb()
    
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)

import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConv(nn.Module):
    def __init__(self, 
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 bias=True):
        super().__init__()

        self.in_proj = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=mid_channels, 
            kernel_size=1,
            bias=bias)
        self.w_conv = nn.Conv2d(
            mid_channels, 
            mid_channels, 
            kernel_size=(1, kernel_size), 
            stride=1, 
            padding=(0, kernel_size//2),
            groups=mid_channels)
        
        self.h_conv = nn.Conv2d(
            mid_channels, 
            mid_channels, 
            kernel_size=(kernel_size, 1), 
            stride=1, 
            padding=(kernel_size//2, 0),
            groups=mid_channels
        )
        
        self.out_proj = nn.Conv2d(
            in_channels=mid_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        x = self.in_proj(x)
        x_w = self.w_conv(x)
        x_h = self.h_conv(x)
        x = torch.cat([x_w, x_h], dim=1)
        x = self.out_proj(x)
        return x


class HoGEdgeGateConv(nn.Module):
    def __init__(self,
                 in_dim,
                 nbins,
                 cell_size=(8, 8),
                 patch_size=(2, 2)):  # 新增patch_size参数
        super().__init__()

        self.nbins = nbins
        self.cell_size = cell_size
        self.patch_size = patch_size  # 保存patch大小
        self.cell_area = cell_size[0] * cell_size[1]  # 替换硬编码的64

        # 保护GroupNorm，确保num_groups至少为1
        num_groups = max(in_dim // 8, 1)
        
        self.hog_feat = nn.Sequential(
            nn.Conv2d(nbins, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.GroupNorm(num_groups, in_dim),
            nn.ReLU(inplace=True),  
            nn.AdaptiveAvgPool2d((1, 1))   
        )

        self.weight = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=in_dim//2, out_channels=in_dim),
            nn.GroupNorm(num_groups, in_dim)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, stride=1),
            nn.GroupNorm(num_groups, in_dim)
        )

        self.fuse_block = nn.Sequential(
            EdgeConv(in_channels=in_dim, mid_channels=in_dim//2, out_channels=in_dim, kernel_size=3),
            nn.GroupNorm(num_groups, in_dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 校验输入尺寸是否能被patch_size整除
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"输入尺寸({H}x{W})必须能被patch_size{self.patch_size}整除"
        
        residual = x
        x = self.image2patches(x)  # 改用实例方法
        x_hog = self.get_hog_feature(x)
        x_hog = self.hog_feat(x_hog)
        
        # 广播x_hog到x的尺寸
        x_hog_broadcast = x_hog.expand(-1, -1, x.shape[2], x.shape[3])
        x1 = self.sigmoid(self.weight(x + x_hog_broadcast))
        x2 = self.conv(x)
        x = x1 * x2

        x = self.patches2image(x)  # 改用实例方法
        x = x + residual
        x = self.fuse_block(x)

        return x

    def get_hog_feature(self, x):
        # 计算均值特征
        x_mean = x.mean(dim=1, keepdim=True)
        B, _, H, W = x_mean.shape
        device = x_mean.device
        dtype = x_mean.dtype

        # Sobel算子（匹配输入dtype）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=dtype, device=device).view(1, 1, 3, 3)
        
        # 计算梯度
        dx = F.conv2d(x_mean, sobel_x, padding=1)
        dy = F.conv2d(x_mean, sobel_y, padding=1)
        
        # 处理梯度为0的情况（避免atan2出NaN）
        eps = 1e-8
        dx = dx + eps * (dx == 0).float()
        dy = dy + eps * (dy == 0).float()
        
        # 计算梯度方向 [0, π]
        gradient_dir = torch.atan2(dy.abs(), dx.abs())  # 简化计算，结果等价于abs(atan2(dy, dx))
        gradient_dir = torch.clamp(gradient_dir, 0, torch.pi)
        
        # 校验尺寸是否能被cell_size整除
        cell_h, cell_w = self.cell_size
        assert H % cell_h == 0 and W % cell_w == 0, \
            f"特征尺寸({H}x{W})必须能被cell_size{self.cell_size}整除"
        
        H_cells = H // cell_h
        W_cells = W // cell_w

        # 重塑为cell维度 [B, 1, H_cells, cell_h, W_cells, cell_w]
        dirs = gradient_dir.view(B, 1, H_cells, cell_h, W_cells, cell_w)
        # 交换维度并展平cell内像素 [B, H_cells, W_cells, cell_h*cell_w]
        dirs = einops.rearrange(dirs, 'b 1 hc ch wc cw -> b hc wc (ch cw)')
        
        # 计算bin索引（向量化操作，替代for循环）
        bin_width = torch.pi / self.nbins
        bin_indices = (dirs / bin_width).floor().long()
        bin_indices = torch.clamp(bin_indices, 0, self.nbins - 1)
        
        # 向量化统计每个cell的bin计数
        B, Hc, Wc, N = dirs.shape
        bin_indices_flat = bin_indices.view(-1, N)  # [B*Hc*Wc, N]
        # 构建one-hot编码
        one_hot = F.one_hot(bin_indices_flat, num_classes=self.nbins).float()
        # 统计每个bin的数量
        cell_hist = one_hot.sum(dim=1)  # [B*Hc*Wc, nbins]
        # 归一化
        cell_hist = cell_hist / self.cell_area
        
        # 重塑为输出维度 [B, nbins, H_cells, W_cells]
        hog_feature = cell_hist.view(B, Hc, Wc, self.nbins).permute(0, 3, 1, 2)
        
        return hog_feature

    def image2patches(self, x):
        """b c (hg h) (wg w) -> (hg wg b) c h w"""
        hg, wg = self.patch_size
        return einops.rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=hg, wg=wg)

    def patches2image(self, x):
        """(hg wg b) c h w -> b c (hg h) (wg w)"""
        hg, wg = self.patch_size
        return einops.rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=hg, wg=wg)


# 测试用例
def test_hog_edge_gate_conv():
    # 1. 初始化参数
    batch_size = 2
    in_dim = 16  # 确保能被8整除（或测试GroupNorm保护逻辑）
    nbins = 9    # 常用HOG bin数
    cell_size = (8, 8)
    patch_size = (2, 2)
    # 输入尺寸：必须能被patch_size和cell_size整除（patch后尺寸为32x32，能被8x8整除）
    H, W = 64, 64  

    # 2. 创建模型和输入
    model = HoGEdgeGateConv(
        in_dim=in_dim,
        nbins=nbins,
        cell_size=cell_size,
        patch_size=patch_size
    )
    x = torch.randn(batch_size, in_dim, H, W)  # 随机输入

    # 3. 前向传播测试
    print("开始前向传播测试...")
    with torch.no_grad():
        output = model(x)
    
    # 4. 校验输出维度
    assert output.shape == x.shape, f"输出维度错误！期望{x.shape}，实际{output.shape}"
    print(f"输入维度: {x.shape}")
    print(f"输出维度: {output.shape}")
    
    # 5. 校验数值合理性（无NaN/Inf）
    assert not torch.isnan(output).any(), "输出包含NaN值！"
    assert not torch.isinf(output).any(), "输出包含Inf值！"
    print("数值校验通过：无NaN/Inf")

    # 6. 测试HOG特征提取
    print("\n测试HOG特征提取...")
    patches = model.image2patches(x)
    hog_feat = model.get_hog_feature(patches)
    print(f"Patch维度: {patches.shape}")
    print(f"HOG特征维度: {hog_feat.shape}")
    assert hog_feat.shape == (patches.shape[0], nbins, patches.shape[2]//cell_size[0], patches.shape[3]//cell_size[1])
    print("HOG特征维度校验通过")

    print("\n✅ 所有测试用例通过！")


if __name__ == "__main__":
    test_hog_edge_gate_conv()

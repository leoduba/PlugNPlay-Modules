import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入einops的核心函数（原代码用到，必须加）
from einops import repeat, rearrange
# 论文：
# 论文地址：
## 获取轻量的自高斯注意力（修复后，保留原核心逻辑）
class LSGAttention(nn.Module):
    def __init__(self, dim, att_inputsize, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim  # 特征维度C
        self.att_inputsize = att_inputsize[0]  # 注意力处理的特征图尺寸（H=W）
        self.num_heads = num_heads  # 多头注意力头数
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子
        
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)  # 线性层（原逻辑：qkv共享）
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        
        # 生成二维高斯位置偏置（原逻辑不变）
        totalpixel = self.att_inputsize * self.att_inputsize
        gauss_coords_h = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        gauss_coords_w = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        gauss_x, gauss_y = torch.meshgrid(gauss_coords_h, gauss_coords_w, indexing='ij')  # 修复：加indexing='ij'避免警告
        sigma = 10
        gauss_pos_index = torch.exp(torch.true_divide(-(gauss_x ** 2 + gauss_y ** 2), (2 * sigma ** 2)))
        self.register_buffer("gauss_pos_index", gauss_pos_index)  # 注册为缓冲区，不参与训练
        
        # Token化可训练参数（原初始化逻辑不变）
        self.token_wA = nn.Parameter(torch.empty(1, self.att_inputsize * self.att_inputsize, dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.token_wV)

    def forward(self, x, mask=None):
        """
        输入x：[B, N, C]  N=H*W（展平后的序列特征）
        输出x：[B, N, C]  与输入维度一致
        """
        B_, N, C = x.shape
        # Token化A矩阵（原逻辑不变）
        wa = repeat(self.token_wA, '() n d -> b n d', b=B_)
        wa = rearrange(wa, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)
        # Token化V矩阵（原逻辑不变）
        VV = repeat(self.token_wV, '() n d -> b n d', b=B_)
        VV = torch.einsum('bij,bjk->bik', x, VV)
        x = torch.einsum('bij,bjk->bik', A, VV)

        # 融合高斯位置偏置的多头注意力（原逻辑不变）
        absolute_pos_bias = self.gauss_pos_index.unsqueeze(0)
        q = self.qkv(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = x.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + absolute_pos_bias.unsqueeze(0)  # 加高斯位置偏置
        
        # 掩码处理（原逻辑不变）
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        # 注意力加权+投影
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# -------------------------- 测试代码（完全修复+维度适配） --------------------------
if __name__ == "__main__":
    # 1. 超参数设置（匹配测试输入）
    torch.manual_seed(42)  # 固定随机种子，结果可复现
    d_model = 512          # 特征维度C
    num_heads = 4          # 多头注意力头数
    att_inputsize = [14]   # 注意力处理的特征图尺寸H=W=14（匹配测试输入的14x14）
    batch_size = 2         # 批次大小
    H, W = 14, 14          # 测试特征图的高宽

    # 2. 初始化LSGAttention模型（修复类名+补全参数+修正语法）
    lsga = LSGAttention(
        dim=d_model,
        att_inputsize=att_inputsize,
        num_heads=num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.
    )

    # 3. 生成随机测试输入 [batch_size, channels, height, width] → CNN标准特征图
    x = torch.randn(batch_size, d_model, H, W)
    print(f"原始CNN特征图输入维度：{x.shape}")

    # 4. 维度适配：CNN特征图[B,C,H,W] → 注意力输入[B,N,C]（N=H*W）
    B, C, H, W = x.shape
    x_flatten = rearrange(x, 'b c h w -> b (h w) c')  # 展平为序列
    print(f"注意力层输入维度（展平后）：{x_flatten.shape}")

    # 5. 前向传播（修复对象名：CAM → lsga）
    out_flatten = lsga(x_flatten)

    # 6. 维度还原：注意力输出[B,N,C] → CNN特征图[B,C,H,W]
    out = rearrange(out_flatten, 'b (h w) c -> b c h w', h=H, w=W)

    # 7. 打印维度验证+参数量统计（修复所有语法错误）
    print("="*50)
    print(f"LSGA输出维度（还原后）：{out.shape}")
    print(f"LSGA总参数量：{sum(p.numel() for p in lsga.parameters()):,}")  # 修复对象名：CAM → lsga
    print("="*50)

    # 8. 验证输出特性（修复语法错误：补充闭合引号）
    print(f"输出与输入形状是否一致：{out.shape == x.shape}")
    print(f"输出值范围：[{out.min():.4f}, {out.max():.4f}]")

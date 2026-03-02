import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 通用设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 测试输入（1个窗口×批量1，256个token，64通道）
x_test = torch.randn(1, 16*16, 64).to(device)

class FLattenAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.flatten = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # 深度卷积扁平化
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        h = w = int(math.sqrt(n))
        # 特征扁平化（空间维度融合）
        x_flat = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x_flat = self.flatten(x_flat).flatten(2).transpose(1, 2)  # [b, n, c]

        # 线性注意力计算
        qkv = self.qkv(x_flat).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        kv = k.transpose(-2, -1) @ v
        z = 1 / (q @ k.transpose(-2, -1).sum(dim=-1, keepdim=True) + 1e-6)
        x_attn = (q @ kv) * z

        x = x_attn.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 测试
flatten_attn = FLattenAttention(64, 4).to(device)
flatten_out = flatten_attn(x_test)
print(f"FLatten Attn输出维度：{flatten_out.shape}")
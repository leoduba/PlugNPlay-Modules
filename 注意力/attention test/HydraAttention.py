import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 通用设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 测试输入（1个窗口×批量1，256个token，64通道）
x_test = torch.randn(1, 16*16, 64).to(device)

class HydraAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., split=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads // split  # 多头拆分
        self.split = split
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.split, self.num_heads, self.head_dim).permute(2, 0, 3, 4, 1, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [b, split, heads, n, head_dim]

        # 多头解耦线性注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 测试
hydra_attn = HydraAttention(64, 4).to(device)
hydra_out = hydra_attn(x_test)
print(f"Hydra Attn输出维度：{hydra_out.shape}")
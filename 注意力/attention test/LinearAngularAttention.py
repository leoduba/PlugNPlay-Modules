import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 通用设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 测试输入（1个窗口×批量1，256个token，64通道）
x_test = torch.randn(1, 16*16, 64).to(device)

 
class LinearAngularAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = 1e-6

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 角度相似度归一化（L2归一化）
        q = F.normalize(q, dim=-1, eps=self.eps)
        k = F.normalize(k, dim=-1, eps=self.eps)
        v = F.normalize(v, dim=-1, eps=self.eps)

        # 线性注意力计算
        kv = k.transpose(-2, -1) @ v
        z = 1 / (q @ k.transpose(-2, -1).sum(dim=-1, keepdim=True) + self.eps)
        x_attn = (q @ kv) * z

        x = x_attn.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 测试
angular_attn = LinearAngularAttention(64, 4).to(device)
angular_out = angular_attn(x_test)
print(f"Linear Angular Attn输出维度：{angular_out.shape}")
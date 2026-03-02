import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 通用设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 测试输入（1个窗口×批量1，256个token，64通道）
x_test = torch.randn(1, 16*16, 64).to(device)

class SoftmaxAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子避免点积过大

        # QKV单线性层生成（ViT主流设计）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        # 1. QKV生成与拆分：[b, n, 3c] → [3, b, heads, n, head_dim]
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 缩放点积计算注意力分数：[b, heads, n, n]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3. 加权求和+投影：[b, heads, n, head_dim] → [b, n, c]
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 测试
sm_attn = SoftmaxAttention(64, 4).to(device)
sm_out = sm_attn(x_test)
print(f"Softmax Attention输出维度：{sm_out.shape}")  # 预期：[1,256,64]
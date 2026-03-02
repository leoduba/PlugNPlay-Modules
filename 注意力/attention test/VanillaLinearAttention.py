import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 通用设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 测试输入（1个窗口×批量1，256个token，64通道）
x_test = torch.randn(1, 16*16, 64).to(device)

class VanillaLinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., kernel=nn.ReLU()):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel = kernel  # 核函数（ReLU/LeakyReLU/Identity）
        self.eps = 1e-6  # 防止除零

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        # 1. QKV生成与拆分+多头维度调整
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 核函数映射（非单射性根源）
        q = self.kernel(q)
        k = self.kernel(k)
        v = self.kernel(v)

        # 3. 线性注意力核心：K/V预融合 + 除法归一化
        k_t = k.transpose(-2, -1)  # [b, heads, head_dim, n]
        kv = k_t @ v  # [b, heads, head_dim, head_dim]
        z = 1 / (q @ k_t.sum(dim=-1, keepdim=True) + self.eps)  # 归一化因子
        x_attn = (q @ kv) * z  # [b, heads, n, head_dim]

        # 4. 维度恢复+投影
        x = x_attn.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 测试（ReLU核函数，与原论文对比一致）
linear_attn = VanillaLinearAttention(64, 4, kernel=nn.ReLU()).to(device)
linear_out = linear_attn(x_test)
print(f"传统Linear Attention输出维度：{linear_out.shape}")  # 预期：[1,256,64]
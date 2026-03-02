import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 通用设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 测试输入（1个窗口×批量1，256个token，64通道）
x_test = torch.randn(1, 16*16, 64).to(device)

class InLineAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window=14):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window = window

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 残差卷积分支：强化局部建模（弥补线性注意力短板）
        self.residual = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv1d(dim, dim * 9, kernel_size=1, groups=num_heads)  # 生成3×3卷积核
        )

    def forward(self, x):
        b, n, c = x.shape
        h = w = int(math.sqrt(n))
        # 1. QKV生成与多头调整
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2. 生成残差卷积核（基于全局均值特征）
        res_weight = self.residual(x.mean(dim=1).unsqueeze(dim=-1)).reshape(b * c, 1, 3, 3)

        # 3. InLine核心：除法改减法 + K/V预融合（尺度平衡避免数值不稳定）
        kv = (k.transpose(-2, -1) * (self.scale / n) ** 0.5) @ (v * (self.scale / n) ** 0.5)
        x_attn = q @ kv + (1 - q @ k.mean(dim=2, keepdim=True).transpose(-2, -1) * self.scale) * v.mean(dim=2, keepdim=True)

        # 4. 维度恢复
        x = x_attn.transpose(1, 2).reshape(b, n, c)

        # 5. 残差卷积增强局部特征（排除首个token，适配窗口化）
        hw = n - 1
        h_new = int(math.sqrt(hw))
        w_new = hw // h_new
        if h_new * w_new != hw:
            h_new = int(math.sqrt(hw)) + 1
            w_new = hw // h_new
        # 维度转换适配分组卷积
        v_ = v[:, :, 1:, :].transpose(1, 2).reshape(b, h_new, w_new, c).permute(0, 3, 1, 2).reshape(1, b * c, h_new, w_new)
        residual = F.conv2d(v_, res_weight, None, padding=(1, 1), groups=b * c)
        x[:, 1:, :] = x[:, 1:, :] + residual.reshape(b, c, hw).permute(0, 2, 1)

        # 6. 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 测试（与原论文开源代码一致）
inline_attn = InLineAttention(64, 4).to(device)
inline_out = inline_attn(x_test)
print(f"InLine Attention输出维度：{inline_out.shape}")  # 预期：[1,256,64]
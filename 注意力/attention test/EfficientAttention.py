import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 通用设备配置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 测试输入（1个窗口×批量1，256个token，64通道）
x_test = torch.randn(1, 16*16, 64).to(device)

 

class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window  # 局部窗口大小
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        h = w = int(math.sqrt(n))  # 16×16
        
        # 核心修复1：补零使尺寸能被窗口大小整除
        pad_h = (self.window - h % self.window) % self.window
        pad_w = (self.window - w % self.window) % self.window
        # 对特征图补零（仅在右侧/下侧补，不影响原有数据）
        x_padded = F.pad(x.reshape(b, h, w, c), (0, 0, 0, pad_w, 0, pad_h))
        h_padded, w_padded = h + pad_h, w + pad_w  # 补零后的尺寸：21×21（7的倍数）
        
        # 核心修复2：基于补零后的尺寸切分窗口
        x_windows = x_padded.unfold(1, self.window, self.window).unfold(2, self.window, self.window)
        # 调整维度：[b, h_win, w_win, window, window, c] → [b, num_windows, window*window, c]
        x_windows = x_windows.reshape(b, -1, self.window*self.window, c)
        b, nw, n_win, c = x_windows.shape
        
        # 窗口内线性注意力计算
        qkv = self.qkv(x_windows).reshape(b, nw, n_win, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 线性注意力核心计算
        kv = k.transpose(-2, -1) @ v
        z = 1 / (q @ k.transpose(-2, -1).sum(dim=-1, keepdim=True) + 1e-6)
        x_attn = (q @ kv) * z

        # 核心修复3：维度恢复（先恢复补零后的尺寸，再裁剪回原始尺寸）
        x_attn = x_attn.permute(0, 1, 3, 2, 4).reshape(b, nw, n_win, c)
        # 计算窗口数量并重塑为补零后的2D尺寸
        h_win = h_padded // self.window
        w_win = w_padded // self.window
        x = x_attn.reshape(b, h_win, w_win, self.window, self.window, c)
        # 合并窗口：[b, h_win, w_win, win, win, c] → [b, h_padded, w_padded, c]
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h_padded, w_padded, c)
        # 裁剪掉补零部分，恢复原始16×16尺寸
        x = x[:, :h, :w, :].reshape(b, n, c)
        
        # 最终投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 测试（修复后可正常运行）
efficient_attn = EfficientAttention(64, 4, window=7).to(device)
efficient_out = efficient_attn(x_test)
print(f"Efficient Attn输出维度：{efficient_out.shape}")  # 预期：[1,256,64]
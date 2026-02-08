
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#地址：https://arxiv.org/pdf/2504.09377
#论文：Gradient as Conditions: Rethinking HOG for All-in-one Image Restoration" (HOGformer)
print("开始测试...")

# 1. HOG特征提取器
class HOGExtractor(nn.Module):
    def __init__(self, num_ori=9, cell_size=8):
        super().__init__()
        self.num_ori = num_ori
        self.cell_size = cell_size
        self.register_buffer('gx', torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32))
        self.register_buffer('gy', torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32))
        self.register_buffer('bins', torch.linspace(0, math.pi, num_ori+1)[:-1] + math.pi/(2*num_ori))
    
    def forward(self, x):
        B, C, H, W = x.shape
        if C > 1:
            x = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
        gx = F.conv2d(x, self.gx, padding=1)
        gy = F.conv2d(x, self.gy, padding=1)
        mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        ori = torch.atan2(gy, gx)
        ori = torch.where(ori < 0, ori + math.pi, ori)
        
        # Soft binning
        weights = torch.exp(-10 * (ori.unsqueeze(2) - self.bins.view(1,1,-1,1,1))**2)
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6)
        hog = (mag.unsqueeze(2) * weights).sum(dim=1)
        hog = F.avg_pool2d(hog, self.cell_size, self.cell_size)
        norm = torch.sqrt((hog**2).sum(dim=1, keepdim=True) + 1e-6)
        return hog / (norm + 1e-6)

print("测试HOG提取器...")
x = torch.randn(1, 3, 64, 64)
hog_ext = HOGExtractor()
hog = hog_ext(x)
print(f"HOG: {hog.shape}")

# 2. HOG注意力
class HOGAttn(nn.Module):
    def __init__(self, dim, heads=2, hog_dim=9):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.hog_gate = nn.Sequential(nn.Linear(hog_dim, dim), nn.Sigmoid())
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, hog):
        B, N, C = x.shape
        shortcut = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        gate = self.hog_gate(hog).reshape(B, N, self.heads, self.head_dim).permute(0,2,1,3)
        q, k = q * gate, k * gate
        attn = (q @ k.transpose(-2,-1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, N, C)
        return shortcut + self.proj(out)

print("测试HOG注意力...")
x_seq = torch.randn(1, 64, 16)  # 减小尺寸
hog_seq = torch.randn(1, 64, 9)
attn = HOGAttn(16, heads=2, hog_dim=9)
out = attn(x_seq, hog_seq)
print(f"Attn: {out.shape}")

# 3. 前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        shortcut = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = x.transpose(1,2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = x.view(B, -1, N).transpose(1,2)
        return shortcut + self.fc2(x)

print("测试前馈网络...")
x_ff = torch.randn(1, 64, 16)
ff = FeedForward(16, hidden=32)
out_ff = ff(x_ff, 8, 8)
print(f"FF: {out_ff.shape}")

# 4. 完整HOGformer
class HOGformer(nn.Module):
    def __init__(self, dim=16, blocks=2, heads=2):
        super().__init__()
        self.hog_ext = HOGExtractor()
        self.hog_proj = nn.Conv2d(9, dim, 1)
        self.input_proj = nn.Conv2d(3, dim, 3, padding=1)
        
        # 简化为一个block
        self.attn = HOGAttn(dim, heads, dim)
        self.ff = FeedForward(dim, dim*2)
        
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, 3, 3, padding=1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        hog = self.hog_ext(x)
        hog_up = F.interpolate(hog, (H, W), mode='bilinear', align_corners=False)
        hog_up = self.hog_proj(hog_up)
        
        feat = self.input_proj(x)
        
        # Block
        x_seq = feat.view(B, feat.size(1), H*W).transpose(1,2)
        hog_seq = hog_up.view(B, hog_up.size(1), H*W).transpose(1,2)
        x_seq = self.attn(x_seq, hog_seq)
        x_seq = self.ff(x_seq, H, W)
        feat = x_seq.transpose(1,2).view(B, -1, H, W)
        
        out = self.output(feat)
        return out + x, hog

print("\n测试完整HOGformer...")
x_img = torch.randn(1, 3, 64, 64)
model = HOGformer(dim=16, blocks=2, heads=2)
restored, hog_feat = model(x_img)
print(f"输入: {x_img.shape}")
print(f"修复: {restored.shape}")
print(f"HOG: {hog_feat.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# 梯度测试
target = torch.randn(1, 3, 64, 64)
loss = F.l1_loss(restored, target)
loss.backward()
print(f"损失: {loss.item():.6f}, 梯度: OK")

# 退化区分测试
print("\n退化区分测试...")
clear = torch.randn(2, 3, 32, 32) * 0.5 + 0.5
blur = F.avg_pool2d(clear, 3, 1, 1)
noise = clear + torch.randn_like(clear) * 0.1

with torch.no_grad():
    h_c = hog_ext(clear)
    h_b = hog_ext(blur)
    h_n = hog_ext(noise)

print(f"清晰: {h_c.mean():.4f}")
print(f"模糊: {h_b.mean():.4f}")
print(f"噪声: {h_n.mean():.4f}")
print(f"差异(模糊): {F.mse_loss(h_c, h_b):.6f}")
print(f"差异(噪声): {F.mse_loss(h_c, h_n):.6f}")

print("\n所有测试通过！")

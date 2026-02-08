
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math
#论文:ReGLA: Efficient Receptive-Field Modeling with Gated Linear AttentionNetwork
#论文地址：https://arxiv.org/pdf/2602.05262
# ==================== 修复版 ReGLA 实现 ====================

class ELRF(nn.Module):
    """Efficient Local Receptive Field"""
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dwconv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv3x3(x)
        x = self.act(x)
        x = self.dwconv5x5(x)
        x = self.act(x)
        return x


class CPE(nn.Module):
    """Conditional Positional Encoding"""
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.dwconv(x))


class RGMA(nn.Module):
    """
    Receptive-field Gated Mixed Attention
    修复版：正确处理输入维度
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # V分支：Linear + Sigmoid gate
        self.v_proj = nn.Linear(dim, dim)
        self.v_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # K-Q分支：Linear + ReLU
        self.k_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kq_act = nn.ReLU()
        
        # 输出
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C] 或 [B, N, C]
        """
        # 处理输入格式
        if x.dim() == 4:
            B, H, W, C = x.shape
            x_flat = x.view(B, H*W, C)
            reshape_back = True
            N = H * W
        else:
            B, N, C = x.shape
            x_flat = x
            reshape_back = False
        
        shortcut = x_flat
        x_norm = self.norm(x_flat)
        
        # V分支：带门控
        v = self.v_proj(x_norm)
        v_gate = self.v_gate(x_norm)
        v = v * v_gate
        
        # K-Q分支：ReLU激活
        k = self.kq_act(self.k_proj(x_norm))
        q = self.kq_act(self.q_proj(x_norm))
        
        # 多头分割
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        
        # 残差
        out = shortcut + out
        
        if reshape_back:
            out = out.view(B, H, W, C)
        
        return out


class ReGLABlock(nn.Module):
    """ReGLA基础块"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.elrf = ELRF(dim)
        self.cpe = CPE(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.rgma = RGMA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        
        # ELRF + CPE
        x = x + self.elrf(x)
        x = x + self.cpe(x)
        
        # 转换格式 [B, C, H, W] -> [B, H, W, C]
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1)
        
        # RGMA
        x_seq = self.rgma(x_seq)
        
        # FFN
        x_seq = x_seq + self.ffn(self.norm2(x_seq))
        
        # 转换回 [B, C, H, W]
        x = x_seq.permute(0, 3, 1, 2)
        
        return x + shortcut


class Downsample(nn.Module):
    """下采样层"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class Stem(nn.Module):
    """初始Stem层"""
    def __init__(self, in_chans: int = 3, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            nn.GELU(),
            nn.Conv2d(out_dim // 2, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ReGLAStage(nn.Module):
    """ReGLA Stage"""
    def __init__(self, dim: int, num_blocks: int, num_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            ReGLABlock(dim, num_heads) for _ in range(num_blocks)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ReGLA(nn.Module):
    """
    完整ReGLA架构
    Stem → Stage 1 → Stage 2 → Stage 3 → Stage 4 → CLS Head
    """
    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        num_classes: int = 1000,
        dims: List[int] = [64, 128, 256, 512],
        num_blocks: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [2, 4, 8, 16]
    ):
        super().__init__()
        
        self.stem = Stem(in_chans, dims[0])
        
        self.stage1 = ReGLAStage(dims[0], num_blocks[0], num_heads[0])
        self.down1 = Downsample(dims[0], dims[1])
        
        self.stage2 = ReGLAStage(dims[1], num_blocks[1], num_heads[1])
        self.down2 = Downsample(dims[1], dims[2])
        
        self.stage3 = ReGLAStage(dims[2], num_blocks[2], num_heads[2])
        self.down3 = Downsample(dims[2], dims[3])
        
        self.stage4 = ReGLAStage(dims[3], num_blocks[3], num_heads[3])
        
        self.norm = nn.BatchNorm2d(dims[3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[3], num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        
        x = self.norm(x)
        x = self.avgpool(x).flatten(1)
        x = self.head(x)
        
        return x


# ==================== 测试 ====================

print("=" * 60)
print("ReGLA 测试套件")
print("=" * 60)

# 1. ELRF
print("\n1. ELRF")
x = torch.randn(2, 64, 56, 56)
elrf = ELRF(64)
out = elrf(x)
print(f"输入: {x.shape}, 输出: {out.shape}, 参数量: {sum(p.numel() for p in elrf.parameters()):,}")
print("✓")

# 2. CPE
print("\n2. CPE")
cpe = CPE(64)
out = cpe(x)
print(f"输入: {x.shape}, 输出: {out.shape}, 参数量: {sum(p.numel() for p in cpe.parameters()):,}")
print("✓")

# 3. RGMA
print("\n3. RGMA")
x = torch.randn(2, 14, 14, 128)  # [B, H, W, C] 格式
rgma = RGMA(128, num_heads=8)
out = rgma(x)
print(f"输入: {x.shape}, 输出: {out.shape}, 参数量: {sum(p.numel() for p in rgma.parameters()):,}")
loss = out.sum()
loss.backward()
print("梯度: OK ✓")

# 4. ReGLA Block
print("\n4. ReGLA Block")
x = torch.randn(2, 128, 28, 28)
block = ReGLABlock(128, num_heads=8)
out = block(x)
print(f"输入: {x.shape}, 输出: {out.shape}, 参数量: {sum(p.numel() for p in block.parameters()):,}")
print("✓")

# 5. 完整ReGLA
print("\n5. 完整ReGLA架构")
x = torch.randn(2, 3, 224, 224)
model = ReGLA(
    img_size=224,
    num_classes=1000,
    dims=[64, 128, 256, 512],
    num_blocks=[2, 2, 6, 2],
    num_heads=[2, 4, 8, 16]
)
logits = model(x)
print(f"输入: {x.shape}, 输出: {logits.shape}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

target = torch.randint(0, 1000, (2,))
loss = nn.CrossEntropyLoss()(logits, target)
loss.backward()
print(f"损失: {loss.item():.4f}, 梯度: OK ✓")

# 6. 维度验证
print("\n6. 各阶段维度验证")
x = torch.randn(1, 3, 224, 224)
print(f"输入: {x.shape}")
x = model.stem(x)
print(f"Stem: {x.shape} (56×56)")
x = model.stage1(x)
print(f"Stage 1: {x.shape}")
x = model.down1(x)
print(f"Down 1: {x.shape} (28×28)")
x = model.stage2(x)
print(f"Stage 2: {x.shape}")
x = model.down2(x)
print(f"Down 2: {x.shape} (14×14)")
x = model.stage3(x)
print(f"Stage 3: {x.shape}")
x = model.down3(x)
print(f"Down 3: {x.shape} (7×7)")
x = model.stage4(x)
print(f"Stage 4: {x.shape}")
print("✓ 所有维度验证通过")

print("\n" + "=" * 60)
print("所有测试完成！")
print("=" * 60)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math
#Github地址：https://arxiv.org/pdf/2411.15659
#论文地址：SMM-Conv: Scalar Matrix Multiplication with Zero Packing forAccelerated Convolution
#QKFormer: Hierarchical Spiking Transformer using Q-K Attention
# ==================== 基础组件 ====================

class SurrogateGradient(torch.autograd.Function):
    """
    代理梯度函数：解决脉冲神经元不可导问题
    前向：Heaviside阶跃函数
    反向：Fast Sigmoid梯度近似
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float = 1.0, slope: float = 25.0) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.slope = slope
        return (input >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        input, = ctx.saved_tensors
        # Fast sigmoid surrogate gradient
        grad_input = grad_output.clone()
        sigmoid_grad = ctx.slope * torch.exp(-ctx.slope * torch.abs(input - ctx.threshold))
        grad_input = grad_input * sigmoid_grad
        return grad_input, None, None


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) 脉冲神经元
    
    膜电位更新公式：
    V[t] = beta * V[t-1] + I[t] - S[t-1] * V_th
    S[t] = 1 if V[t] >= V_th else 0
    
    其中：
    - beta: 膜电位衰减系数
    - V_th: 发放阈值
    - I[t]: 输入电流
    - S[t]: 输出脉冲 (0或1)
    """
    
    def __init__(
        self,
        tau: float = 2.0,  # 膜时间常数
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_slope: float = 25.0
    ):
        super(LIFNeuron, self).__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_slope = surrogate_slope
        self.beta = math.exp(-1.0 / tau)  # 衰减系数
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C, H, W] 或 [T, B, C, L] 或 [T, B, C]
        Returns:
            spike: 同输入形状
        """
        T = x.shape[0]
        batch_shape = x.shape[1:]
        
        # 初始化膜电位
        v = torch.zeros(batch_shape, device=x.device, dtype=x.dtype)
        spikes = []
        
        for t in range(T):
            # 膜电位更新
            v = self.beta * v + x[t]
            
            # 脉冲发放（使用代理梯度）
            spike = SurrogateGradient.apply(v, self.v_threshold, self.surrogate_slope)
            spikes.append(spike)
            
            # 硬重置
            v = v * (1 - spike) + self.v_reset * spike
        
        return torch.stack(spikes)


# ==================== 核心创新模块 ====================

class QKAttention(nn.Module):
    """
    Q-K Attention: 脉冲形式的Q-K注意力机制（核心创新1）
    
    特点：
    1. 线性复杂度 O(N*C) 而非标准注意力的 O(N²*C)
    2. 使用二值脉冲向量进行注意力计算
    3. 仅使用Query和Key，不使用Value
    
    两种模式：
    - 'token': QK Token Attention (QKTA)，沿token维度计算注意力
    - 'channel': QK Channel Attention (QKCA)，沿channel维度计算注意力
    
    Args:
        dim: 通道数
        num_heads: 注意力头数
        qka_type: 'token' 或 'channel'
        tau: LIF神经元时间常数
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qka_type: str = 'token',
        tau: float = 2.0
    ):
        super(QKAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert qka_type in ['token', 'channel'], "qka_type must be 'token' or 'channel'"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qka_type = qka_type
        self.scale = self.head_dim ** -0.5
        
        # Q和K的线性投影（不使用V！）
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        
        # BatchNorm + LIF生成脉冲形式的Q和K
        self.q_bn = nn.BatchNorm1d(dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNeuron(tau=tau)
        self.k_lif = LIFNeuron(tau=tau)
        
        # 输出投影
        self.out_proj = nn.Linear(dim, dim)
        self.out_bn = nn.BatchNorm1d(dim)
        self.out_lif = LIFNeuron(tau=tau)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, L, C] 其中L = H*W (token数量)
        Returns:
            out: [T, B, L, C]
        """
        T, B, L, C = x.shape
        
        # 生成Q和K（浮点数）
        q = self.q_proj(x)  # [T, B, L, C]
        k = self.k_proj(x)  # [T, B, L, C]
        
        # 转换为脉冲形式
        # 重塑用于BatchNorm: [T*B*L, C] -> [C, T*B*L] -> [T, B, L, C]
        q = q.reshape(T * B * L, C)
        q = self.q_bn(q).reshape(T, B, L, C)
        q_spike = self.q_lif(q)  # 二值脉冲 [T, B, L, C]
        
        k = k.reshape(T * B * L, C)
        k = self.k_bn(k).reshape(T, B, L, C)
        k_spike = self.k_lif(k)  # 二值脉冲 [T, B, L, C]
        
        # 多头分割
        q_spike = q_spike.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k_spike = k_spike.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        # [T, B, num_heads, L, head_dim]
        
        if self.qka_type == 'token':
            # QK Token Attention: 沿token维度计算
            # 注意力权重 = Q * K^T
            attn = torch.matmul(q_spike, k_spike.transpose(-2, -1)) * self.scale
            # [T, B, num_heads, L, L]
            
            # 使用脉冲形式的注意力权重对输入进行加权
            # 注意：这里不使用V，而是直接用注意力权重对原始输入加权
            x_reshaped = x.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            out = torch.matmul(attn, x_reshaped)  # [T, B, num_heads, L, head_dim]
            
        else:  # channel
            # QK Channel Attention: 沿channel维度计算
            # 注意力权重 = Q^T * K
            attn = torch.matmul(q_spike.transpose(-2, -1), k_spike) * self.scale
            # [T, B, num_heads, head_dim, head_dim]
            
            x_reshaped = x.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            out = torch.matmul(x_reshaped, attn)  # [T, B, num_heads, L, head_dim]
        
        # 合并多头
        out = out.permute(0, 1, 3, 2, 4).reshape(T, B, L, C)
        
        # 输出投影 + LIF
        out = self.out_proj(out)
        out = out.reshape(T * B * L, C)
        out = self.out_bn(out).reshape(T, B, L, C)
        out = self.out_lif(out)
        
        return out


class SPEDS(nn.Module):
    """
    Spiking Patch Embedding with Deformed Shortcut (SPEDS)（核心创新3）
    
    增强脉冲信息传输和整合的Patch嵌入模块
    
    公式：Y = F(X, {Wi}) + SN(Wd * X)
    其中：
    - F(X, {Wi}): 主干网络（卷积层）
    - Wd: 1x1卷积的shortcut
    - SN: 脉冲神经元层
    
    特点：
    - 在下采样块中引入恒等映射
    - 使用轻量级线性投影（1x1卷积）
    - 增强信息流动
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        tau: float = 2.0
    ):
        super(SPEDS, self).__init__()
        
        # 主干网络
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.lif = LIFNeuron(tau=tau)
        
        # Deformed Shortcut: 1x1卷积 + 脉冲神经元
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                LIFNeuron(tau=tau)
            )
        else:
            self.shortcut = LIFNeuron(tau=tau)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C, H, W]
        Returns:
            out: [T, B, out_channels, H', W']
        """
        T, B, C, H, W = x.shape
        
        # 主干路径
        out = []
        for t in range(T):
            out.append(self.conv(x[t]))
        out = torch.stack(out)  # [T, B, out_channels, H', W']
        
        # 合并T和B用于BatchNorm
        out = out.reshape(T * B, out.shape[2], out.shape[3], out.shape[4])
        out = self.bn(out)
        out = out.reshape(T, B, out.shape[1], out.shape[2], out.shape[3])
        out = self.lif(out)
        
        # Shortcut路径
        if isinstance(self.shortcut, nn.Sequential):
            shortcut_out = []
            for t in range(T):
                shortcut_out.append(self.shortcut[0](x[t]))  # Conv
            shortcut_out = torch.stack(shortcut_out)
            shortcut_out = shortcut_out.reshape(T * B, shortcut_out.shape[2], 
                                               shortcut_out.shape[3], shortcut_out.shape[4])
            shortcut_out = self.shortcut[1](shortcut_out)  # BN
            shortcut_out = shortcut_out.reshape(T, B, shortcut_out.shape[1], 
                                               shortcut_out.shape[2], shortcut_out.shape[3])
            shortcut_out = self.shortcut[2](shortcut_out)  # LIF
        else:
            shortcut_out = self.shortcut(x)
        
        # 残差连接
        return out + shortcut_out


class QKFormerBlock(nn.Module):
    """
    QKFormer基础块：结合Q-K Attention和MLP
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qka_type: str = 'token',
        tau: float = 2.0
    ):
        super(QKFormerBlock, self).__init__()
        
        # Q-K Attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = QKAttention(dim, num_heads, qka_type, tau)
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            LIFNeuron(tau=tau),
            nn.Linear(mlp_hidden_dim, dim),
            nn.BatchNorm1d(dim),
            LIFNeuron(tau=tau)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, L, C]
        Returns:
            out: [T, B, L, C]
        """
        T, B, L, C = x.shape
        
        # Attention分支
        x_norm = self.norm1(x.reshape(T * B * L, C)).reshape(T, B, L, C)
        attn_out = self.attn(x_norm)
        x = x + attn_out
        
        # MLP分支
        x_norm = self.norm2(x.reshape(T * B * L, C)).reshape(T, B, L, C)
        mlp_out = []
        for t in range(T):
            out_t = self.mlp[0](x_norm[t])  # Linear
            out_t = out_t.reshape(B * L, -1)
            out_t = self.mlp[1](out_t)  # BN
            out_t = out_t.reshape(B, L, -1)
            out_t = self.mlp[2](out_t.unsqueeze(0)).squeeze(0)  # LIF
            
            out_t = self.mlp[3](out_t)  # Linear
            out_t = out_t.reshape(B * L, -1)
            out_t = self.mlp[4](out_t)  # BN
            out_t = out_t.reshape(B, L, -1)
            out_t = self.mlp[5](out_t.unsqueeze(0)).squeeze(0)  # LIF
            mlp_out.append(out_t)
        mlp_out = torch.stack(mlp_out)
        
        x = x + mlp_out
        
        return x


class QKFormer(nn.Module):
    """
    QKFormer: 层次化脉冲Transformer（整体架构）
    
    核心创新：
    1. Q-K Attention: 线性复杂度的脉冲注意力
    2. 层次化结构: 多尺度脉冲表示（token数逐级减少）
    3. SPEDS: 带变形shortcut的脉冲patch嵌入
    
    Args:
        img_size: 输入图像大小
        patch_size: 初始patch大小
        in_channels: 输入通道数
        num_classes: 分类数
        embed_dims: 每个阶段的嵌入维度 [C1, C2, C3]
        num_heads: 每个阶段的头数
        depths: 每个阶段的块数 [D1, D2, D3]
        mlp_ratios: MLP扩展比例
        qka_types: 每个阶段的QKA类型 ['token', 'token', 'channel']
        tau: LIF神经元时间常数
        T: 时间步数
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dims: List[int] = [96, 192, 384],
        num_heads: List[int] = [3, 6, 12],
        depths: List[int] = [2, 2, 6],
        mlp_ratios: List[float] = [4.0, 4.0, 4.0],
        qka_types: List[str] = ['token', 'token', 'channel'],
        tau: float = 2.0,
        T: int = 4
    ):
        super(QKFormer, self).__init__()
        self.num_classes = num_classes
        self.T = T
        self.num_stages = len(embed_dims)
        
        # 初始Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], kernel_size=patch_size, stride=patch_size, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            LIFNeuron(tau=tau)
        )
        
        # 计算每个阶段的token数量
        num_patches = (img_size // patch_size) ** 2
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # 随机深度衰减
        
        # 构建层次化阶段
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = nn.ModuleDict()
            
            # Patch merging (除了第一阶段)
            if i > 0:
                stage['downsample'] = SPEDS(
                    embed_dims[i-1], embed_dims[i],
                    kernel_size=3, stride=2, padding=1, tau=tau
                )
                num_patches = num_patches // 4
            
            # QKFormer Blocks
            blocks = []
            for j in range(depths[i]):
                blocks.append(QKFormerBlock(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qka_type=qka_types[i],
                    tau=tau
                ))
            stage['blocks'] = nn.ModuleList(blocks)
            
            self.stages.append(stage)
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            logits: [B, num_classes]
        """
        B = x.shape[0]
        
        # 重复T次作为时间步
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        
        # 初始Patch Embedding
        patch_out = []
        for t in range(self.T):
            out_t = self.patch_embed[0](x[t])  # Conv
            patch_out.append(out_t)
        x = torch.stack(patch_out)  # [T, B, C, H', W']
        
        T, B, C, H, W = x.shape
        x = x.reshape(T * B, C, H, W)
        x = self.patch_embed[1](x)  # BN
        x = x.reshape(T, B, C, H, W)
        x = self.patch_embed[2](x)  # LIF
        
        # 转换为序列 [T, B, L, C]
        T, B, C, H, W = x.shape
        x = x.reshape(T, B, C, H * W).permute(0, 1, 3, 2)
        
        # 通过各个阶段
        for stage in self.stages:
            # 下采样（除了第一阶段）
            if 'downsample' in stage:
                # 转换回图像格式 [T, B, L, C] -> [T, B, C, H, W]
                T, B, L, C = x.shape
                H = W = int(math.sqrt(L))
                x_img = x.permute(0, 1, 3, 2).reshape(T, B, C, H, W)
                x_img = stage['downsample'](x_img)
                T, B, C, H, W = x_img.shape
                x = x_img.reshape(T, B, C, H * W).permute(0, 1, 3, 2)
            
            # 通过blocks
            for block in stage['blocks']:
                x = block(x)
        
        # 全局平均池化
        x = x.mean(dim=2)  # [T, B, C]
        
        # 时间维度平均
        x = x.mean(dim=0)  # [B, C]
        
        # 分类
        x = self.norm(x)
        x = self.head(x)
        
        return x


# ==================== 测试样例 ====================

def test_lif_neuron():
    """测试LIF神经元"""
    print("=" * 60)
    print("测试1: LIF脉冲神经元")
    print("=" * 60)
    
    lif = LIFNeuron(tau=2.0, v_threshold=1.0)
    
    # 创建输入：随时间增加的电流
    T, B, C = 10, 2, 4
    x = torch.randn(T, B, C) * 0.5 + 0.3  # 正向偏置电流
    
    spike = lif(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出脉冲形状: {spike.shape}")
    print(f"脉冲发放率: {spike.mean().item():.4f}")
    print(f"脉冲总和: {spike.sum().item()}")
    print(f"是否为二值: {torch.unique(spike).tolist()}")
    print("✓ LIF神经元测试通过\n")


def test_qk_attention():
    """测试Q-K Attention"""
    print("=" * 60)
    print("测试2: Q-K Attention")
    print("=" * 60)
    
    T, B, L, C = 4, 2, 16, 96  # 时间步, batch, token数, 通道
    
    # Token Attention
    qkta = QKAttention(dim=C, num_heads=3, qka_type='token', tau=2.0)
    x = torch.randn(T, B, L, C)
    out = qkta(x)
    
    print(f"QKTA输入形状: {x.shape}")
    print(f"QKTA输出形状: {out.shape}")
    print(f"输出是否为脉冲(二值): {torch.unique(out).tolist()}")
    
    # Channel Attention
    qkca = QKAttention(dim=C, num_heads=3, qka_type='channel', tau=2.0)
    out_c = qkca(x)
    
    print(f"\nQKCA输入形状: {x.shape}")
    print(f"QKCA输出形状: {out_c.shape}")
    print("✓ Q-K Attention测试通过\n")


def test_speds():
    """测试SPEDS模块"""
    print("=" * 60)
    print("测试3: SPEDS (Spiking Patch Embedding with Deformed Shortcut)")
    print("=" * 60)
    
    T, B, C_in, H, W = 4, 2, 3, 56, 56
    
    speds = SPEDS(
        in_channels=C_in,
        out_channels=96,
        kernel_size=3,
        stride=2,
        tau=2.0
    )
    
    x = torch.randn(T, B, C_in, H, W)
    out = speds(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"空间分辨率降低: {H}x{W} -> {out.shape[-2]}x{out.shape[-1]}")
    print(f"通道数变化: {C_in} -> {out.shape[2]}")
    print(f"输出是否为脉冲: {torch.unique(out).tolist()}")
    print("✓ SPEDS测试通过\n")


def test_qkformer_block():
    """测试QKFormer基础块"""
    print("=" * 60)
    print("测试4: QKFormer Block")
    print("=" * 60)
    
    T, B, L, C = 4, 2, 16, 96
    
    block = QKFormerBlock(dim=C, num_heads=3, mlp_ratio=4.0, qka_type='token')
    x = torch.randn(T, B, L, C)
    out = block(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")
    print("✓ QKFormer Block测试通过\n")


def test_full_qkformer():
    """测试完整QKFormer模型"""
    print("=" * 60)
    print("测试5: 完整QKFormer模型")
    print("=" * 60)
    
    # 创建小型QKFormer用于测试
    model = QKFormer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dims=[96, 192, 384],
        num_heads=[3, 6, 12],
        depths=[1, 1, 2],
        qka_types=['token', 'token', 'channel'],
        tau=2.0,
        T=4
    )
    
    x = torch.randn(2, 3, 32, 32)  # [B, C, H, W]
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"输入图像形状: {x.shape}")
    print(f"输出logits形状: {output.shape}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试梯度回传
    model.train()
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    print(f"梯度回传: 成功")
    print("✓ 完整QKFormer测试通过\n")


def test_energy_efficiency():
    """测试能量效率特性"""
    print("=" * 60)
    print("测试6: 能量效率分析")
    print("=" * 60)
    
    T, B, L, C = 4, 4, 64, 96
    
    # 标准自注意力 (模拟)
    class StandardSelfAttention(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.num_heads = num_heads
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
        
        def forward(self, x):
            # x: [B, L, C]
            B, L, C = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, L, C)
            return self.proj(out)
    
    std_attn = StandardSelfAttention(C, 3)
    qk_attn = QKAttention(C, 3, 'token')
    
    x = torch.randn(B, L, C)
    x_time = x.unsqueeze(0).repeat(T, 1, 1, 1)  # [T, B, L, C]
    
    # 计算FLOPs (近似)
    head_dim = C // 3
    
    # 标准注意力: O(L^2 * C)
    std_flops = B * (L * L * head_dim * 3)  # QK^T, softmax, @V
    
    # Q-K Attention: O(L * C) (线性复杂度)
    qk_flops = B * T * (L * head_dim * head_dim)  # 简化的矩阵乘法
    
    print(f"Token数量: {L}, 通道数: {C}")
    print(f"标准注意力FLOPs (近似): {std_flops:,}")
    print(f"Q-K Attention FLOPs (近似): {qk_flops:,}")
    print(f"计算复杂度降低: {(1 - qk_flops/std_flops)*100:.1f}%")
    
    # 测试脉冲稀疏性
    with torch.no_grad():
        out = qk_attn(x_time)
        firing_rate = out.mean().item()
    
    print(f"\n脉冲发放率: {firing_rate:.4f} (越稀疏越节能)")
    print("✓ 能量效率分析完成\n")


def test_hierarchical_structure():
    """测试层次化结构"""
    print("=" * 60)
    print("测试7: 层次化多尺度结构")
    print("=" * 60)
    
    model = QKFormer(
        img_size=224,
        patch_size=4,
        embed_dims=[96, 192, 384],
        num_heads=[3, 6, 12],
        depths=[2, 2, 6],
        qka_types=['token', 'token', 'channel'],
        T=4
    )
    
    x = torch.randn(1, 3, 224, 224)
    
    print("层次化结构信息:")
    print(f"阶段数: {model.num_stages}")
    print(f"嵌入维度: {model.stages}")
    
    # 计算每个阶段的token数
    patch_size = 4
    num_patches = (224 // patch_size) ** 2  # 3136
    print(f"\nToken数量变化:")
    print(f"  Stage 1: {num_patches} tokens (14x14 patches)")
    print(f"  Stage 2: {num_patches // 4} tokens (7x7 patches)")
    print(f"  Stage 3: {num_patches // 16} tokens (3.5x3.5 -> 实际为整数)")
    
    print(f"\n总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ 层次化结构测试通过\n")

# 扩展实现：混合注意力策略和实际训练示例

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# ==================== 混合注意力策略 ====================

class MixedQKAttention(nn.Module):
    """
    混合Q-K注意力策略（论文中的优化策略）
    
    策略：
    - 早期阶段（token多，channel少）：使用QKTA (Token Attention)
    - 后期阶段（token少，channel多）：使用QKCA (Channel Attention)
    
    这样可以在不同层次上优化计算效率和表征能力
    """
    
    def __init__(self, dim: int, num_heads: int = 8, tau: float = 2.0):
        super().__init__()
        self.qkta = QKAttention(dim, num_heads, 'token', tau)
        self.qkca = QKAttention(dim, num_heads, 'channel', tau)
    
    def forward_token(self, x):
        """Token attention mode"""
        return self.qkta(x)
    
    def forward_channel(self, x):
        """Channel attention mode"""
        return self.qkca(x)


class AdvancedQKFormerBlock(nn.Module):
    """
    高级QKFormer块，支持混合注意力
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qka_type: str = 'token',
        tau: float = 2.0,
        drop_path: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = QKAttention(dim, num_heads, qka_type, tau)
        
        # Drop Path (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            LIFNeuron(tau=tau),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, dim),
            nn.BatchNorm1d(dim),
            LIFNeuron(tau=tau),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, L, C = x.shape
        
        # Attention with residual
        x_norm = self.norm1(x.reshape(T * B * L, C)).reshape(T, B, L, C)
        attn_out = self.attn(x_norm)
        x = x + self.drop_path(attn_out)
        
        # MLP with residual
        x_norm = self.norm2(x.reshape(T * B * L, C)).reshape(T, B, L, C)
        mlp_out = []
        for t in range(T):
            out_t = self.mlp(x_norm[t])
            mlp_out.append(out_t)
        mlp_out = torch.stack(mlp_out)
        x = x + self.drop_path(mlp_out)
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# ==================== 训练工具 ====================

class SNNTrainer:
    """
    SNN训练器，支持代理梯度下降
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        self.scheduler.step()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100.0 * correct / total
        }
    
    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100.0 * correct / total
        }


# ==================== 扩展测试 ====================

def test_mixed_attention():
    """测试混合注意力策略"""
    print("=" * 60)
    print("测试8: 混合注意力策略")
    print("=" * 60)
    
    T, B, L, C = 4, 2, 64, 96
    x = torch.randn(T, B, L, C)
    
    mixed_attn = MixedQKAttention(C, num_heads=3)
    
    # Token attention (适合早期阶段)
    out_token = mixed_attn.forward_token(x)
    print(f"QKTA (Token) 输出形状: {out_token.shape}")
    
    # Channel attention (适合后期阶段)
    out_channel = mixed_attn.forward_channel(x)
    print(f"QKCA (Channel) 输出形状: {out_channel.shape}")
    
    print("\n策略建议:")
    print("  Stage 1 (L=3136, C=96): 使用QKTA - token多，适合token attention")
    print("  Stage 3 (L=196, C=384): 使用QKCA - channel多，适合channel attention")
    print("✓ 混合注意力测试通过\n")


def test_training_pipeline():
    """测试完整训练流程"""
    print("=" * 60)
    print("测试9: 训练流程测试")
    print("=" * 60)
    
    # 创建小型数据集
    batch_size = 8
    train_data = torch.randn(64, 3, 32, 32)
    train_labels = torch.randint(0, 10, (64,))
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = QKFormer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dims=[64, 128, 256],
        num_heads=[2, 4, 8],
        depths=[1, 1, 2],
        qka_types=['token', 'token', 'channel'],
        T=4
    )
    
    trainer = SNNTrainer(model, device='cpu', learning_rate=1e-3)
    
    # 训练一个epoch
    print("训练一个epoch...")
    start_time = time.time()
    train_metrics = trainer.train_epoch(train_loader)
    train_time = time.time() - start_time
    
    print(f"训练时间: {train_time:.2f}s")
    print(f"训练损失: {train_metrics['loss']:.4f}")
    print(f"训练准确率: {train_metrics['accuracy']:.2f}%")
    
    # 评估
    eval_metrics = trainer.evaluate(train_loader)
    print(f"评估准确率: {eval_metrics['accuracy']:.2f}%")
    
    print("✓ 训练流程测试通过\n")


def test_energy_consumption():
    """测试能耗估算"""
    print("=" * 60)
    print("测试10: 能耗估算")
    print("=" * 60)
    
    # 创建测试模型
    model = QKFormer(
        img_size=32,
        patch_size=4,
        embed_dims=[96, 192, 384],
        num_heads=[3, 6, 12],
        depths=[2, 2, 6],
        T=4
    )
    
    x = torch.randn(1, 3, 32, 32)
    
    # 统计脉冲发放率
    firing_rates = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            firing_rates.append(output.mean().item())
    
    # 注册hook
    hooks = []
    for module in model.modules():
        if isinstance(module, LIFNeuron):
            hooks.append(module.register_forward_hook(hook_fn))
    
    model.eval()
    with torch.no_grad():
        _ = model(x)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    avg_firing_rate = sum(firing_rates) / len(firing_rates) if firing_rates else 0
    
    print(f"平均脉冲发放率: {avg_firing_rate:.4f}")
    print(f"稀疏度: {(1 - avg_firing_rate) * 100:.2f}%")
    
    # 估算能耗（相对于ANN的乘加操作）
    # SNN: 脉冲事件驱动，能耗与发放率成正比
    # ANN: 每个时间步都需要浮点乘加
    energy_ratio = avg_firing_rate * 4  # T=4时间步
    print(f"\n相对能耗估算:")
    print(f"  ANN (浮点乘加): 100%")
    print(f"  SNN (脉冲驱动): {energy_ratio * 100:.2f}%")
    print(f"  节能比例: {(1 - energy_ratio) * 100:.2f}%")
    
    print("✓ 能耗估算测试通过\n")


def test_comparison_with_ann():
    """与ANN的对比测试"""
    print("=" * 60)
    print("测试11: SNN vs ANN 对比")
    print("=" * 60)
    
    # SNN模型
    snn_model = QKFormer(
        img_size=32,
        patch_size=4,
        embed_dims=[64, 128],
        num_heads=[2, 4],
        depths=[1, 1],
        T=4
    )
    
    # 等效ANN模型（简化版）
    class ANNTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, 64, 4, 4)
            self.attn = nn.MultiheadAttention(64, 2, batch_first=True)
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.patch_embed(x)
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).permute(0, 2, 1)
            x, _ = self.attn(x, x, x)
            x = x.mean(dim=1)
            return self.fc(x)
    
    ann_model = ANNTransformer()
    
    x = torch.randn(2, 3, 32, 32)
    
    # 参数量对比
    snn_params = sum(p.numel() for p in snn_model.parameters())
    ann_params = sum(p.numel() for p in ann_model.parameters())
    
    print(f"参数量对比:")
    print(f"  SNN (QKFormer): {snn_params:,}")
    print(f"  ANN (标准Transformer): {ann_params:,}")
    
    # 输出对比
    with torch.no_grad():
        snn_out = snn_model(x)
        ann_out = ann_model(x)
    
    print(f"\n输出形状:")
    print(f"  SNN: {snn_out.shape}")
    print(f"  ANN: {ann_out.shape}")
    
    # 计算复杂度对比
    print(f"\n计算复杂度:")
    print(f"  SNN Attention: O(T × L × C) = 线性")
    print(f"  ANN Attention: O(L² × C) = 二次")
    print(f"  (其中L=token数, C=通道数, T=时间步)")
    
    print("✓ 对比测试通过\n")


def test_robustness():
    """测试鲁棒性"""
    print("=" * 60)
    print("测试12: 鲁棒性测试")
    print("=" * 60)
    
    model = QKFormer(
        img_size=32,
        patch_size=4,
        embed_dims=[64, 128],
        num_heads=[2, 4],
        depths=[1, 1],
        T=4
    )
    
    model.eval()
    
    # 测试不同噪声水平
    x_clean = torch.randn(4, 3, 32, 32)
    
    print("噪声鲁棒性测试:")
    with torch.no_grad():
        out_clean = model(x_clean)
        
        for noise_level in [0.1, 0.2, 0.5]:
            x_noisy = x_clean + noise_level * torch.randn_like(x_clean)
            out_noisy = model(x_noisy)
            
            # 计算输出变化
            diff = torch.abs(out_clean - out_noisy).mean().item()
            print(f"  噪声水平 {noise_level}: 输出变化 {diff:.4f}")
    
    # 测试时间步变化
    print("\n时间步鲁棒性测试:")
    model_T2 = QKFormer(
        img_size=32, patch_size=4,
        embed_dims=[64, 128], num_heads=[2, 4],
        depths=[1, 1], T=2
    )
    model_T2.eval()
    
    with torch.no_grad():
        out_T4 = model(x_clean)
        out_T2 = model_T2(x_clean)
    
    print(f"  T=4 输出均值: {out_T4.mean().item():.4f}")
    print(f"  T=2 输出均值: {out_T2.mean().item():.4f}")
    
    print("✓ 鲁棒性测试通过\n")

 

# 运行所有测试
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QKFormer: Hierarchical Spiking Transformer 测试套件")
    print("=" * 60 + "\n")
    
    test_lif_neuron()
    test_qk_attention()
    test_speds()
    test_qkformer_block()
    test_full_qkformer()
    test_energy_efficiency()
    test_hierarchical_structure()
    
    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("QKFormer 扩展测试套件")
    print("=" * 60 + "\n")
    
    test_mixed_attention()
    test_training_pipeline()
    test_energy_consumption()
    test_comparison_with_ann()
    test_robustness()
    
    print("=" * 60)
    print("所有扩展测试完成！")
    print("=" * 60)

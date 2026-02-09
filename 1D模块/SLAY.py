
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math
#论文：SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel
#h论文地址：ttps://arxiv.org/pdf/2602.04915 
class GaussLaguerreQuadrature:
    """
    Gauss-Laguerre 求积节点和权重计算
    用于近似积分 ∫_0^∞ e^{-t} f(t) dt ≈ Σ w_i f(t_i)
    """
    def __init__(self, num_points: int = 8):
        self.num_points = num_points
        # 使用 numpy 计算 Gauss-Laguerre 节点和权重
        self.nodes, self.weights = self._compute_nodes_weights()
    
    def _compute_nodes_weights(self):
        """计算 Gauss-Laguerre 求积节点和权重"""
        # 使用 numpy 的专用函数
        nodes, weights = np.polynomial.laguerre.laggauss(self.num_points)
        return torch.tensor(nodes, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32)
    
    def get_scaled_params(self, C: float):
        """
        根据论文进行变量替换 t = C*s, s = t/C
        返回缩放后的节点 s_r 和权重 w_r = α_r/C
        """
        # s_r = t_r / C, w_r = α_r / C
        s_r = self.nodes / C
        w_r = self.weights / C
        return s_r, w_r


class SphericalYatKernel(nn.Module):
    """
    球面 Yat-Kernel 实现
    T_sph(q_hat, k_hat) = (q_hat^T k_hat)^2 / (C - 2*q_hat^T k_hat)
    """
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.C = 2.0 + eps
    
    def forward(self, q_hat: torch.Tensor, k_hat: torch.Tensor) -> torch.Tensor:
        """
        计算精确的球面 Yat-Kernel（用于验证）
        Args:
            q_hat: [..., d] 单位球面上的查询向量
            k_hat: [..., d] 单位球面上的键向量
        Returns:
            kernel_values: [...] 核函数值
        """
        # 计算余弦相似度 x = q^T k
        x = torch.sum(q_hat * k_hat, dim=-1)
        
        # T_sph = x^2 / (C - 2x)
        numerator = x ** 2
        denominator = self.C - 2.0 * x
        
        # 确保数值稳定性
        denominator = torch.clamp(denominator, min=self.eps)
        
        return numerator / denominator


class PolynomialAnchorFeatures(nn.Module):
    """
    多项式锚点特征 (Anchor Features)
    φ_anc(x) = 1/√P [(x^T a_i)^2]_{i=1}^P
    保证非负性，比精确的 vec(xx^T) 更稳定
    """
    def __init__(self, dim: int, num_anchors: int = 64):
        super().__init__()
        self.dim = dim
        self.num_anchors = num_anchors
        
        # 随机初始化锚点向量并归一化
        anchors = torch.randn(num_anchors, dim)
        anchors = F.normalize(anchors, p=2, dim=-1)
        self.register_buffer('anchors', anchors)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d] 输入向量（应在单位球面上）
        Returns:
            features: [..., P] 锚点特征
        """
        # x^T a_i: [..., P]
        projections = torch.matmul(x, self.anchors.T)
        
        # (x^T a_i)^2 保证非负
        squared = projections ** 2
        
        # 归一化
        features = squared / math.sqrt(self.num_anchors)
        
        return features


class ExponentialPRF(nn.Module):
    """
    指数正随机特征 (Positive Random Features)
    φ_PRF(u; s) = 1/√D [exp(√(2s) ω_i^T u - s)]_{i=1}^D
    用于近似 exp(2s q^T k)
    """
    def __init__(self, dim: int, num_features: int = 64):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        
        # 随机采样 ω_i ~ N(0, I_d)
        omega = torch.randn(num_features, dim)
        self.register_buffer('omega', omega)
    
    def forward(self, u: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: [..., d] 输入向量
            s: 标量或 [...] 尺度参数
        Returns:
            features: [..., D] 指数随机特征
        """
        # ω^T u: [..., D]
        projections = torch.matmul(u, self.omega.T)
        
        # √(2s) ω^T u - s
        if isinstance(s, torch.Tensor) and s.dim() > 0:
            # s 是向量，需要广播
            scale = torch.sqrt(2.0 * s).unsqueeze(-1)  # [..., 1]
            bias = s.unsqueeze(-1)  # [..., 1]
        else:
            scale = math.sqrt(2.0 * s)
            bias = s
        
        scaled_proj = scale * projections - bias
        
        # exp(·) 保证正性
        exp_features = torch.exp(scaled_proj)
        
        # 归一化
        features = exp_features / math.sqrt(self.num_features)
        
        return features


class SketchedTensorProduct(nn.Module):
    """
    草图张量积 (Sketched Tensor Product)
    高效近似高维 Kronecker 积，避免显式计算大的张量积
    """
    def __init__(self, poly_dim: int, exp_dim: int, out_dim: int):
        super().__init__()
        self.poly_dim = poly_dim
        self.exp_dim = exp_dim
        self.out_dim = out_dim
        
        # 草图矩阵 S: [poly_dim * exp_dim, out_dim]
        # 使用 Count-sketch 或随机投影
        total_dim = poly_dim * exp_dim
        sketch_matrix = torch.randn(total_dim, out_dim) / math.sqrt(out_dim)
        self.register_buffer('sketch_matrix', sketch_matrix)
    
    def forward(self, poly_features: torch.Tensor, exp_features: torch.Tensor) -> torch.Tensor:
        """
        计算 S(φ_poly ⊗ φ_exp)
        Args:
            poly_features: [..., poly_dim]
            exp_features: [..., exp_dim]
        Returns:
            sketched: [..., out_dim]
        """
        # 外积 (Kronecker 积): [..., poly_dim * exp_dim]
        # 使用 einsum 高效计算 batch 外积
        outer = torch.einsum('...i,...j->...ij', poly_features, exp_features)
        outer_flat = outer.reshape(*outer.shape[:-2], -1)  # [..., poly_dim * exp_dim]
        
        # 草图投影
        sketched = torch.matmul(outer_flat, self.sketch_matrix)  # [..., out_dim]
        
        return sketched


class SLAYAttention(nn.Module):
    """
    SLAY: Spherical Linearized Attention with Yat-Kernel
    几何感知的球面线性化注意力机制
    
    核心公式:
    Y = Ψ(Q)(Ψ(K)^T V) / Ψ(Q)(Ψ(K)^T 1)
    
    其中 Ψ 是融合特征映射:
    Ψ_r(u) = √w_r S(φ_poly(u) ⊗ φ_PRF(u; s_r))
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_quadrature_points: int = 8,
        num_anchor_features: int = 64,
        num_exp_features: int = 64,
        sketch_dim: int = 256,
        eps: float = 1e-4,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.C = 2.0 + eps
        
        # 求积参数
        self.quadrature = GaussLaguerreQuadrature(num_quadrature_points)
        self.num_quadrature_points = num_quadrature_points
        
        # 特征维度
        self.num_anchor_features = num_anchor_features
        self.num_exp_features = num_exp_features
        self.sketch_dim = sketch_dim
        
        # 每个头独立的特征映射参数
        self.poly_features = nn.ModuleList([
            PolynomialAnchorFeatures(self.head_dim, num_anchor_features)
            for _ in range(num_heads)
        ])
        
        self.exp_features = nn.ModuleList([
            ExponentialPRF(self.head_dim, num_exp_features)
            for _ in range(num_heads)
        ])
        
        self.sketch_projections = nn.ModuleList([
            SketchedTensorProduct(num_anchor_features, num_exp_features, sketch_dim)
            for _ in range(num_heads)
        ])
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缓存求积参数
        self.register_buffer('quadrature_nodes', None)
        self.register_buffer('quadrature_weights', None)
        self._init_quadrature()
    
    def _init_quadrature(self):
        """初始化并缓存求积参数"""
        s_r, w_r = self.quadrature.get_scaled_params(self.C)
        self.register_buffer('quadrature_nodes', s_r)  # [R]
        self.register_buffer('quadrature_weights', w_r)  # [R]
    
    def _compute_feature_map(self, x: torch.Tensor, head_idx: int) -> torch.Tensor:
        """
        计算融合特征映射 Ψ(x)
        Args:
            x: [batch, seq_len, head_dim] 单头的输入 (已归一化到球面)
            head_idx: 头索引
        Returns:
            features: [batch, seq_len, R * sketch_dim] 融合特征
        """
        batch_size, seq_len, _ = x.shape
        R = self.num_quadrature_points
        
        # 计算多项式锚点特征 [batch, seq, P]
        poly_feat = self.poly_features[head_idx](x)
        
        # 为每个求积点计算特征并融合
        all_features = []
        for r in range(R):
            s_r = self.quadrature_nodes[r]
            w_r = self.quadrature_weights[r]
            
            # 指数特征 [batch, seq, D]
            exp_feat = self.exp_features[head_idx](x, s_r.item())
            
            # 草图张量积 [batch, seq, sketch_dim]
            sketched = self.sketch_projections[head_idx](poly_feat, exp_feat)
            
            # 乘以 √w_r
            scaled = sketched * torch.sqrt(w_r)
            all_features.append(scaled)
        
        # 拼接所有求积点的特征 [batch, seq, R * sketch_dim]
        fused_features = torch.cat(all_features, dim=-1)
        
        return fused_features
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim] 输入序列
            mask: [batch, seq_len] 可选的掩码
            return_attention: 是否返回注意力权重（仅用于分析，会破坏线性复杂度）
        Returns:
            output: [batch, seq_len, dim] 输出序列
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影到 Q, K, V
        Q = self.q_proj(x)  # [batch, seq, dim]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 分头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 球面归一化 (L2归一化)
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        
        # 对每个头应用 SLAY 注意力
        outputs = []
        for h in range(self.num_heads):
            q_h = Q[:, h, :, :]  # [batch, seq, head_dim]
            k_h = K[:, h, :, :]  # [batch, seq, head_dim]
            v_h = V[:, h, :, :]  # [batch, seq, head_dim]
            
            # 计算特征映射
            psi_q = self._compute_feature_map(q_h, h)  # [batch, seq, m]
            psi_k = self._compute_feature_map(k_h, h)  # [batch, seq, m]
            
            m = psi_q.shape[-1]
            
            # 线性注意力计算
            # 分子: Ψ(Q)(Ψ(K)^T V) 
            # KV_product: [batch, m, head_dim]
            KV_product = torch.matmul(psi_k.transpose(-2, -1), v_h)
            numerator = torch.matmul(psi_q, KV_product)  # [batch, seq, head_dim]
            
            # 分母: Ψ(Q)(Ψ(K)^T 1)
            # K_sum: [batch, m, 1]
            K_sum = psi_k.sum(dim=-2, keepdim=True).transpose(-2, -1)  # [batch, m, 1]
            denominator = torch.matmul(psi_q, K_sum).squeeze(-1)  # [batch, seq]
            
            # 应用掩码（如果提供）
            if mask is not None:
                # mask: [batch, seq] -> [batch, seq, 1]
                mask_expanded = mask.unsqueeze(-1).float()
                v_masked = v_h * mask_expanded
                # 重新计算带掩码的分子分母
                KV_product = torch.matmul(psi_k.transpose(-2, -1), v_masked)
                numerator = torch.matmul(psi_q, KV_product)
                
                K_sum_masked = (psi_k * mask.unsqueeze(-2)).sum(dim=-2, keepdim=True).transpose(-2, -1)
                denominator = torch.matmul(psi_q, K_sum_masked).squeeze(-1)
            
            # 归一化
            denominator = torch.clamp(denominator, min=self.eps)
            head_output = numerator / denominator.unsqueeze(-1)  # [batch, seq, head_dim]
            
            outputs.append(head_output)
        
        # 合并所有头 [batch, seq, dim]
        output = torch.stack(outputs, dim=1).transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        
        # 输出投影
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output


class SLAYTransformerLayer(nn.Module):
    """
    完整的 SLAY Transformer 层
    包含 SLAY Attention + FFN + 残差连接 + LayerNorm
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        **slay_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SLAYAttention(dim, num_heads, dropout=dropout, **slay_kwargs)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 注意力子层
        x = x + self.attn(self.norm1(x), mask)
        # FFN 子层
        x = x + self.mlp(self.norm2(x))
        return x


print("✅ SLAY 模块实现完成！")
print(f"核心组件:")
print(f"  - GaussLaguerreQuadrature: Gauss-Laguerre 求积")
print(f"  - SphericalYatKernel: 球面 Yat-Kernel")
print(f"  - PolynomialAnchorFeatures: 多项式锚点特征")
print(f"  - ExponentialPRF: 指数正随机特征")
print(f"  - SketchedTensorProduct: 草图张量积")
print(f"  - SLAYAttention: 主注意力模块")
print(f"  - SLAYTransformerLayer: 完整 Transformer 层")
# 1. 基础使用
import torch
 

attn = SLAYAttention(
    dim=512, 
    num_heads=8,
    num_quadrature_points=4,    # 求积点数（精度 vs 速度权衡）
    num_anchor_features=64,     # 多项式特征维度
    num_exp_features=64,        # 指数特征维度  
    sketch_dim=128              # 草图投影维度
)

x = torch.randn(2, 1024, 512)   # [batch, seq, dim]
output = attn(x)                # [batch, seq, dim]

# 2. 构建 Transformer
from slay import SLAYTransformerLayer

layer = SLAYTransformerLayer(dim=512, num_heads=8)
x = layer(x)

# 3. 处理超长序列（128K tokens）
long_seq = torch.randn(1, 131072, 512)
output = attn(long_seq)  # 线性内存，不会 OOM

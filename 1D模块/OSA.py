
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Literal
import math

# 论文：Orthogonal Self-Attention
# 论文：https://arxiv.org/pdf/2602.05996
class NewtonSchulzIteration(nn.Module):
    """
    Newton-Schulz 迭代计算正交基
    M_{k+1} = 1/2 * M_k (3I - M_k^T M_k)
    用于从 [Q,K] 计算近似正交矩阵 B(X)
    """
    def __init__(self, num_iters: int = 5, eps: float = 1e-8):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
    
    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: [N, 2d_v] 输入矩阵 (Q 和 K 的拼接)
        Returns:
            B: [N, r] 近似正交矩阵，满足 B^T B ≈ I_r
        """
        N, d = M.shape
        
        # 初始化 M_0 = M / (||M||_F + eps)
        norm = torch.norm(M, p='fro') + self.eps
        M_iter = M / norm
        
        # Newton-Schulz 迭代
        for _ in range(self.num_iters):
            M_t_M = torch.matmul(M_iter.T, M_iter)  # [2d_v, 2d_v]
            M_iter = 0.5 * torch.matmul(M_iter, (3.0 * torch.eye(d, device=M.device) - M_t_M))
        
        return M_iter  # [N, 2d_v]，近似正交


class OrthogonalBasisQR(nn.Module):
    """
    使用 Reduced QR 分解构造正交基 B(X)
    对 [Q,K] 进行 QR 分解，取前 r 列
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: [N, 2d_v] 输入矩阵
        Returns:
            B: [N, r] 正交矩阵，满足 B^T B = I_r
        """
        # QR 分解: M = QR
        Q, R = torch.linalg.qr(M, mode='reduced')  # Q: [N, min(N, 2d_v)], R: [min(N, 2d_v), 2d_v]
        
        # Q 的列已经是正交基
        return Q  # [N, r] where r = min(N, 2d_v)


class MatrixExponentialLowRank(nn.Module):
    """
    低秩矩阵指数计算
    利用定理 2.1: exp(S) = I_N + B (exp(tilde_S) - I_r) B^T
    其中 tilde_S = B^T S B
    """
    def __init__(self, method: Literal['qr', 'newton_schulz'] = 'newton_schulz', 
                 num_iters: int = 5):
        super().__init__()
        self.method = method
        if method == 'newton_schulz':
            self.basis_module = NewtonSchulzIteration(num_iters=num_iters)
        else:
            self.basis_module = OrthogonalBasisQR()
    
    def forward(self, S: torch.Tensor, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        计算 exp(S) 的低秩近似
        Args:
            S: [N, N] 斜对称矩阵 (理论上，但这里用 Q,K 重新构造低秩版本)
            Q: [N, d_v] Query 矩阵
            K: [N, d_v] Key 矩阵
        Returns:
            exp_S: [N, N] 正交矩阵
        """
        N, d_v = Q.shape
        
        # 构造 M = [Q, K] ∈ R^{N × 2d_v}
        M = torch.cat([Q, K], dim=-1)  # [N, 2d_v]
        
        # 计算正交基 B(X) ∈ R^{N × r}
        B = self.basis_module(M)  # [N, r], r <= 2d_v
        r = B.shape[1]
        
        # 计算低秩 tilde_S = B^T S B ∈ R^{r × r}
        # S = alpha/sqrt(d_v) * (QK^T - KQ^T)
        # 直接计算 B^T S B 避免构造完整的 S
        # B^T (QK^T - KQ^T) B = B^T Q K^T B - B^T K Q^T B
        
        BT_Q = torch.matmul(B.T, Q)  # [r, d_v]
        BT_K = torch.matmul(B.T, K)  # [r, d_v]
        
        # (B^T Q)(K^T B) - (B^T K)(Q^T B) = BT_Q @ BT_K.T - BT_K @ BT_Q.T
        tilde_S = torch.matmul(BT_Q, BT_K.T) - torch.matmul(BT_K, BT_Q.T)  # [r, r]
        
        # 计算 exp(tilde_S) 使用矩阵指数
        # 对于小矩阵 (r <= 2d_v, 通常很小)，可以直接计算
        exp_tilde_S = torch.linalg.matrix_exp(tilde_S)  # [r, r]
        
        # 重构完整的 exp(S) = I_N + B (exp(tilde_S) - I_r) B^T
        I_N = torch.eye(N, device=S.device, dtype=S.dtype)
        I_r = torch.eye(r, device=S.device, dtype=S.dtype)
        
        diff = exp_tilde_S - I_r  # [r, r]
        
        # B @ diff @ B^T
        B_diff = torch.matmul(B, diff)  # [N, r]
        B_diff_BT = torch.matmul(B_diff, B.T)  # [N, N]
        
        exp_S = I_N + B_diff_BT  # [N, N]
        
        return exp_S


class OrthogonalSelfAttention(nn.Module):
    """
    正交自注意力 (OSA) 单头实现
    OSA(X) = A(X) X W_V W_O
    其中 A(X) = exp(S) ∈ SO(N), S = alpha/sqrt(d_v) * (QK^T - KQ^T)
    """
    def __init__(
        self, 
        d_model: int, 
        d_v: int,
        basis_method: Literal['qr', 'newton_schulz'] = 'newton_schulz',
        num_iters: int = 5,
        init_alpha: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_v = d_v
        
        # 投影矩阵
        self.W_Q = nn.Linear(d_model, d_v, bias=False)
        self.W_K = nn.Linear(d_model, d_v, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        
        # 可学习的缩放参数 alpha
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        
        # 低秩矩阵指数计算模块
        self.matrix_exp = MatrixExponentialLowRank(method=basis_method, num_iters=num_iters)
        
        self._init_weights()
    
    def _init_weights(self):
        """
        特殊初始化确保 Jacobian 条件良好
        1. W_Q, W_K 使得 [W_Q, W_K] 正交 (Stiefel 流形)
        2. W_V 正交初始化
        3. alpha 初始较小，使 A(X) ≈ I
        """
        # 对 W_Q 和 W_K 进行正交初始化
        # 将 [W_Q, W_K] 视为一个整体进行正交初始化
        W_Q_weight = self.W_Q.weight.data  # [d_v, d_model]
        W_K_weight = self.W_K.weight.data  # [d_v, d_model]
        
        # 拼接并正交化
        W_cat = torch.cat([W_Q_weight, W_K_weight], dim=0)  # [2d_v, d_model]
        if W_cat.shape[0] <= W_cat.shape[1]:
            # 使用 QR 分解进行正交初始化
            Q, _ = torch.linalg.qr(W_cat.T, mode='reduced')
            W_cat_orth = Q.T[:2*self.d_v, :]
        else:
            # 使用 SVD
            U, S, Vh = torch.linalg.svd(W_cat, full_matrices=False)
            W_cat_orth = U @ Vh
        
        # 分割回 W_Q 和 W_K
        self.W_Q.weight.data = W_cat_orth[:self.d_v, :]
        self.W_K.weight.data = W_cat_orth[self.d_v:2*self.d_v, :]
        
        # W_V 正交初始化
        nn.init.orthogonal_(self.W_V.weight)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [batch, N, d_model] 输入序列
        Returns:
            output: [batch, N, d_v] 输出序列
        """
        batch_size, N, _ = X.shape
        
        # 计算 Q, K, V
        Q = self.W_Q(X)  # [batch, N, d_v]
        K = self.W_K(X)  # [batch, N, d_v]
        V = self.W_V(X)  # [batch, N, d_v]
        
        # 对每个 batch 分别计算
        outputs = []
        for b in range(batch_size):
            Q_b = Q[b]  # [N, d_v]
            K_b = K[b]  # [N, d_v]
            V_b = V[b]  # [N, d_v]
            
            # 构造斜对称矩阵 S (理论上)
            # 实际通过低秩方法计算 exp(S)
            # 这里传递 Q, K 用于构造低秩基
            dummy_S = torch.zeros(N, N, device=X.device, dtype=X.dtype)  # 占位符
            
            # 计算正交注意力矩阵 A(X) = exp(S) ∈ SO(N)
            A = self.matrix_exp(dummy_S, Q_b, K_b)  # [N, N]
            
            # 应用注意力: A(X) @ V
            attended = torch.matmul(A, V_b)  # [N, d_v]
            outputs.append(attended)
        
        output = torch.stack(outputs, dim=0)  # [batch, N, d_v]
        
        # 缩放
        output = output * (self.alpha / math.sqrt(self.d_v))
        
        return output


class MultiHeadOSA(nn.Module):
    """
    多头正交自注意力 (M-OSA)
    M-OSA(X) = Concat(OSA_1, ..., OSA_h) W_O
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        basis_method: Literal['qr', 'newton_schulz'] = 'newton_schulz',
        num_iters: int = 5,
        init_alpha: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_v = d_model // num_heads  # 每个头的维度
        
        # 创建 h 个 OSA 头
        self.heads = nn.ModuleList([
            OrthogonalSelfAttention(
                d_model=d_model,
                d_v=self.d_v,
                basis_method=basis_method,
                num_iters=num_iters,
                init_alpha=init_alpha
            )
            for _ in range(num_heads)
        ])
        
        # 输出投影 W_O
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        # 正交初始化 W_O
        nn.init.orthogonal_(self.W_O.weight)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [batch, N, d_model]
        Returns:
            output: [batch, N, d_model]
        """
        # 每个头的输出 [batch, N, d_v]
        head_outputs = [head(X) for head in self.heads]
        
        # 拼接 [batch, N, d_model]
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # 最终投影
        output = self.W_O(concatenated)
        
        return output


class OSATransformerLayer(nn.Module):
    """
    OSA Transformer 层 (无跳跃连接版本，论文核心)
    X_l = MLP(M-OSA(X_{l-1}))
    注意：没有残差连接和层归一化！
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        basis_method: Literal['qr', 'newton_schulz'] = 'newton_schulz',
        num_iters: int = 5,
        init_alpha: float = 0.1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.mosa = MultiHeadOSA(
            d_model=d_model,
            num_heads=num_heads,
            basis_method=basis_method,
            num_iters=num_iters,
            init_alpha=init_alpha
        )
        
        # MLP
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # 注意：OSA 论文中明确去除了 LayerNorm 和跳跃连接
        # 这是其核心创新：无需这些稳定技巧即可训练深层网络
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [batch, N, d_model]
        Returns:
            X_l: [batch, N, d_model]
        """
        # OSA 子层 (无跳跃连接，无 LayerNorm)
        X_hat = self.mosa(X)
        
        # MLP 子层 (无跳跃连接，无 LayerNorm)
        X_l = self.mlp(X_hat)
        
        return X_l


class OSATransformer(nn.Module):
    """
    完整的 OSA Transformer (Skipless 版本)
    用于验证：无需跳跃连接和层归一化即可稳定训练
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        basis_method: Literal['qr', 'newton_schulz'] = 'newton_schulz',
        num_iters: int = 5,
        init_alpha: float = 0.1,
        dropout: float = 0.0,
        num_classes: int = 10,
        patch_size: int = 16,
        img_size: int = 224,
        in_channels: int = 3
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch 嵌入
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 位置嵌入 (可学习)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        
        # OSA Transformer 层 (Skipless!)
        self.layers = nn.ModuleList([
            OSATransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                basis_method=basis_method,
                num_iters=num_iters,
                init_alpha=init_alpha,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(d_model)  # 只在最后使用一次
        self.head = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            logits: [batch, num_classes]
        """
        # Patch 嵌入
        x = self.patch_embed(x)  # [batch, d_model, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [batch, N, d_model]
        
        # 加位置嵌入
        x = x + self.pos_embed
        
        # 通过 OSA 层 (无跳跃连接!)
        for layer in self.layers:
            x = layer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 分类
        x = self.norm(x)
        logits = self.head(x)
        
        return logits


print("✅ Orthogonal Self-Attention (OSA) 模块实现完成！")
print("\n核心组件:")
print("  - NewtonSchulzIteration: Newton-Schulz 迭代计算正交基")
print("  - OrthogonalBasisQR: QR 分解构造正交基")
print("  - MatrixExponentialLowRank: 低秩矩阵指数计算 (定理 2.1)")
print("  - OrthogonalSelfAttention: 单头 OSA")
print("  - MultiHeadOSA: 多头 OSA (M-OSA)")
print("  - OSATransformerLayer: Skipless Transformer 层")
print("  - OSATransformer: 完整 ViT 架构")

# 修复实现中的问题

class ImprovedNewtonSchulz(nn.Module):
    """
    改进的 Newton-Schulz 迭代
    增加迭代次数，改进初始化
    """
    def __init__(self, num_iters: int = 10, eps: float = 1e-8):
        super().__init__()
        self.num_iters = num_iters
        self.eps = eps
    
    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        计算 M 的近似正交基
        目标: B^T B = I
        """
        # 如果 N < 2d_v，需要调整
        N, d = M.shape
        
        # 更好的初始化：使用 SVD 的近似
        # M = U S V^T, 取 U 的前 r 列
        try:
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            # U 已经是正交的，但我们需要维度匹配
            # 取前 min(N, d) 个左奇异向量
            r = min(N, d)
            B_init = U[:, :r]
            
            # 如果 N > d，需要填充
            if N > d:
                # 使用 Newton-Schulz 细化
                M_iter = B_init
                for _ in range(self.num_iters):
                    M_t_M = torch.matmul(M_iter.T, M_iter)
                    M_iter = 0.5 * torch.matmul(M_iter, (3.0 * torch.eye(r, device=M.device) - M_t_M))
                return M_iter
            else:
                return B_init
        except:
            # 回退到标准 Newton-Schulz
            norm = torch.norm(M, p='fro') + self.eps
            M_iter = M / norm
            
            for _ in range(self.num_iters):
                M_t_M = torch.matmul(M_iter.T, M_iter)
                M_iter = 0.5 * torch.matmul(M_iter, (3.0 * torch.eye(d, device=M.device) - M_t_M))
            
            return M_iter


class ImprovedMatrixExponential(nn.Module):
    """
    改进的低秩矩阵指数计算
    更稳定的数值实现
    """
    def __init__(self, method: str = 'newton_schulz', num_iters: int = 10):
        super().__init__()
        self.method = method
        self.num_iters = num_iters
        if method == 'newton_schulz':
            self.basis_module = ImprovedNewtonSchulz(num_iters=num_iters)
        else:
            self.basis_module = OrthogonalBasisQR()
    
    def forward(self, S: torch.Tensor, Q: torch.Tensor, K: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """
        计算 exp(S) 的低秩近似
        S = alpha/sqrt(d_v) * (QK^T - KQ^T)
        """
        N, d_v = Q.shape
        
        # 构造 M = [Q, K]
        M = torch.cat([Q, K], dim=-1)  # [N, 2d_v]
        
        # 计算正交基 B(X)
        B = self.basis_module(M)  # [N, r]
        r = B.shape[1]
        
        # 确保 r <= 2d_v
        r = min(r, 2 * d_v)
        B = B[:, :r]
        
        # 计算低秩 tilde_S = B^T S B
        BT_Q = torch.matmul(B.T, Q)  # [r, d_v]
        BT_K = torch.matmul(B.T, K)  # [r, d_v]
        
        # tilde_S = alpha/sqrt(d_v) * (BT_Q @ BT_K.T - BT_K @ BT_Q.T)
        scale = alpha / math.sqrt(d_v)
        tilde_S = scale * (torch.matmul(BT_Q, BT_K.T) - torch.matmul(BT_K, BT_Q.T))
        
        # 确保 tilde_S 是斜对称的
        tilde_S = 0.5 * (tilde_S - tilde_S.T)
        
        # 计算 exp(tilde_S)
        exp_tilde_S = torch.linalg.matrix_exp(tilde_S)
        
        # 重构 exp(S)
        I_N = torch.eye(N, device=S.device, dtype=S.dtype)
        I_r = torch.eye(r, device=S.device, dtype=S.dtype)
        
        diff = exp_tilde_S - I_r
        B_diff = torch.matmul(B, diff)
        B_diff_BT = torch.matmul(B_diff, B.T)
        
        exp_S = I_N + B_diff_BT
        
        return exp_S


def test_improved_implementation():
    """测试改进后的实现"""
    print("=" * 60)
    print("改进实现测试")
    print("=" * 60)
    
    # 测试 1: Newton-Schulz 正交化
    print("\n1. Newton-Schulz 正交化 (10 次迭代):")
    N, d = 100, 64
    M = torch.randn(N, 2*d)
    
    ns = ImprovedNewtonSchulz(num_iters=10)
    B = ns(M)
    BT_B = torch.matmul(B.T, B)
    I = torch.eye(BT_B.shape[0])
    error = torch.norm(BT_B - I, p='fro').item()
    print(f"   正交误差: {error:.6f} {'✅' if error < 1.0 else '❌'}")
    
    # 测试 2: 矩阵指数
    print("\n2. 低秩矩阵指数:")
    N, d_v = 50, 32
    Q = torch.randn(N, d_v)
    K = torch.randn(N, d_v)
    alpha = 0.1
    
    matrix_exp = ImprovedMatrixExponential(method='newton_schulz', num_iters=10)
    exp_S = matrix_exp(torch.zeros(N, N), Q, K, alpha)
    
    # 验证正交性
    I = torch.eye(N)
    ortho_error = torch.norm(torch.matmul(exp_S.T, exp_S) - I, p='fro').item()
    det = torch.det(exp_S).item()
    print(f"   正交误差: {ortho_error:.6f} {'✅' if ortho_error < 1.0 else '❌'}")
    print(f"   行列式: {det:.6f} {'✅' if abs(abs(det) - 1.0) < 0.1 else '❌'}")
    
    # 测试 3: 与直接计算的对比
    S = (alpha / math.sqrt(d_v)) * (torch.matmul(Q, K.T) - torch.matmul(K, Q.T))
    exp_S_direct = torch.linalg.matrix_exp(S)
    diff = torch.norm(exp_S - exp_S_direct, p='fro').item()
    print(f"   与直接计算差异: {diff:.6f}")
    
    print("\n✅ 改进实现测试完成")

# 运行改进测试
test_improved_implementation()

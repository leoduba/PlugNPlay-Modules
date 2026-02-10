
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple

# 设置随机种子确保可重复性
torch.manual_seed(42)

print("=" * 80)
print("四种注意力机制的PyTorch实现")
print("=" * 80)

# ==============================================================================
# A. Vanilla Softmax Attention (标准注意力)
# ==============================================================================
class VanillaSoftmaxAttention(nn.Module):
    """
    标准Softmax注意力机制
    计算复杂度: O(n²d)
    内存复杂度: O(n²)
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
            mask: [batch_size, seq_len, seq_len] or None
        Returns:
            output: [batch_size, seq_len, dim]
        """
        B, N, D = x.shape
        
        # 投影到Q, K, V
        Q = self.q_proj(x)  # [B, N, D]
        K = self.k_proj(x)  # [B, N, D]
        V = self.v_proj(x)  # [B, N, D]
        
        # 重塑为多头: [B, N, H, d] -> [B, H, N, d]
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: [B, H, N, d] @ [B, H, d, N] = [B, H, N, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和: [B, H, N, N] @ [B, H, N, d] = [B, H, N, d]
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头: [B, H, N, d] -> [B, N, H, d] -> [B, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.out_proj(attn_output)

print("\n✅ A. Vanilla Softmax Attention 定义完成")


# ==============================================================================
# B. Nyström Attention (Nyström近似注意力)
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class NystromAttention(nn.Module):
    """
    Nyström Attention - 使用Nyström方法近似Softmax注意力
    通过 landmarks (landmark点) 将复杂度从 O(n²) 降低到 O(nm)
    其中 m << n 是landmark数量
    
    参考: "Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention" (2021)
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.residual = residual
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def iterative_pinv(self, x: torch.Tensor, iterations: int) -> torch.Tensor:
        """
        使用迭代方法计算Moore-Penrose伪逆
        基于: https://arxiv.org/abs/2102.03902
        """
        # 初始化
        device = x.device
        dtype = x.dtype
        
        # 转置用于计算
        x = x.transpose(-2, -1)  # [B, H, m, n] -> [B, H, n, m]
        
        # 使用迭代法近似伪逆
        # 初始近似
        x_t = x.transpose(-2, -1)
        identity = torch.eye(x.size(-1), device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        
        # 迭代改进
        curr = x_t @ torch.inverse(x @ x_t + 1e-6 * identity)
        
        for _ in range(iterations):
            # Newton-Schulz迭代
            xt_x = x_t @ x
            curr = 2 * curr - curr @ xt_x @ curr
            
        return curr
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            output: [batch_size, seq_len, dim]
        """
        # ===================== 核心修复：强制转为整数 =====================
        # 错误根源：B/N/D可能是张量类型，需转为普通整数
        B = int(x.shape[0])
        N = int(x.shape[1])
        D = int(x.shape[2])
        
        # 投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 重塑为多头（全程使用整数，避免张量）
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        
        # 缩放因子（纯浮点数，无张量）
        scale = float(torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        Q = Q / scale
        
        # 选择landmarks (均匀采样)
        if N <= self.num_landmarks:
            # 如果序列长度小于landmark数，直接使用全部
            Q_land = Q
            K_land = K
            V_land = V
            m = N  # m是整数
        else:
            # 均匀采样landmarks
            indices = torch.linspace(0, N-1, self.num_landmarks, device=x.device).long()
            Q_land = Q[:, :, indices, :]  # [B, H, m, d]
            K_land = K[:, :, indices, :]  # [B, H, m, d]
            V_land = V[:, :, indices, :]  # [B, H, m, d]
            m = self.num_landmarks  # m是整数
        
        # 正确计算注意力得分矩阵
        A = torch.matmul(Q_land, K_land.transpose(-2, -1))
        A = F.softmax(A, dim=-1)  # [B, H, m, m]
        
        B_mat = torch.matmul(Q, K_land.transpose(-2, -1))  # 变量名改为B_mat，避免和batch_size的B冲突
        B_mat = F.softmax(B_mat, dim=-1)  # [B, H, N, m]
        
        C = torch.matmul(Q_land, K.transpose(-2, -1))
        C = F.softmax(C, dim=-1)  # [B, H, m, N]
        
        # 计算A的伪逆
        A_pinv = self.iterative_pinv(A, self.pinv_iterations)  # [B, H, m, m]
        
        # 正确的Nyström近似公式
        C_V = torch.matmul(C, V)  # [B, H, m, d]
        A_pinv_C_V = torch.matmul(A_pinv, C_V)  # [B, H, m, d]
        attn_output = torch.matmul(B_mat, A_pinv_C_V)  # [B, H, N, d]
        
        # 可选的残差连接
        if self.residual:
            local_attn = torch.matmul(F.softmax(torch.matmul(Q, K.transpose(-2, -1)), dim=-1), V)
            attn_output = attn_output + local_attn
        
        # Dropout
        attn_output = self.dropout(attn_output)
        
        # 合并多头（所有参数都是整数）
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, N, D)  # 现在参数都是纯整数，不会报错
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


print("✅ B. Nyström Attention 定义完成")


# ==============================================================================
# C. Dilated + Flash Attention (稀疏+分块注意力)
# ==============================================================================

# 修复 Dilated Flash Attention
class DilatedFlashAttention(nn.Module):
    """
    修复后的分块稀疏注意力
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        block_size: int = 64,
        dilation: int = 4,
        window_size: int = 256,
        dropout: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.dilation = dilation
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def create_sparse_mask(self, N: int, device: torch.device) -> torch.Tensor:
        """创建稀疏掩码矩阵"""
        # 使用块稀疏模式
        num_blocks = (N + self.block_size - 1) // self.block_size
        mask = torch.zeros(N, N, device=device, dtype=torch.bool)
        
        for i in range(N):
            # 1. 局部窗口
            start = max(0, i - self.window_size)
            end = min(N, i + self.window_size + 1)
            mask[i, start:end] = True
            
            # 2. 同一块内的连接
            block_id = i // self.block_size
            block_start = block_id * self.block_size
            block_end = min(block_start + self.block_size, N)
            mask[i, block_start:block_end] = True
            
            # 3. 空洞采样 (全局)
            if i % self.dilation == 0:
                dilated_indices = torch.arange(0, N, self.dilation, device=device)
                mask[i, dilated_indices] = True
            
            # 4. 自身
            mask[i, i] = True
            
        return mask
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        
        # 投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 重塑为多头
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 创建稀疏掩码
        sparse_mask = self.create_sparse_mask(N, x.device)
        
        # 标准注意力计算，但应用稀疏掩码
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用稀疏掩码
        scores = scores.masked_fill(~sparse_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # 处理可能的NaN
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.out_proj(attn_output)

 


print("✅ C. Dilated + Flash Attention 定义完成")


# ==============================================================================
# D. Linear Attention (线性注意力)
# ==============================================================================
class LinearAttention(nn.Module):
    """
    Linear Attention - 将复杂度从 O(n²) 降低到 O(n)
    通过核技巧将softmax分解，改变矩阵乘法顺序
    
    标准注意力: softmax(QK^T)V
    线性注意力: φ(Q)(φ(K)^T V)  -> 先算 (φ(K)^T V) 得到 d×d 矩阵
    
    参考: "Efficient Attention: Attention with Linear Complexities" (Shen et al., 2021)
          "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (Katharopoulos et al., 2020)
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        feature_dim: Optional[int] = None,
        kernel_fn: str = "elu+1",  # 核函数: "relu", "elu+1", "softmax"
        dropout: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.feature_dim = feature_dim or self.head_dim
        self.kernel_fn = kernel_fn
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # 可选的特征映射层 (用于softmax核近似)
        if kernel_fn == "softmax":
            self.feature_map = nn.Linear(self.head_dim, self.feature_dim)
        
    def apply_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用核函数将Q, K映射到特征空间
        """
        if self.kernel_fn == "relu":
            # ReLU核: φ(x) = ReLU(x)
            return F.relu(x)
        elif self.kernel_fn == "elu+1":
            # ELU+1核: φ(x) = ELU(x) + 1 (保证正值)
            return F.elu(x) + 1
        elif self.kernel_fn == "softmax":
            # Softmax核近似: 使用随机特征映射
            return F.softmax(self.feature_map(x), dim=-1)
        else:
            return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
        Returns:
            output: [batch_size, seq_len, dim]
        """
        B, N, D = x.shape
        
        # 投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 重塑为多头
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d]
        
        # 应用核函数
        Q_prime = self.apply_kernel(Q)  # [B, H, N, d] or [B, H, N, feature_dim]
        K_prime = self.apply_kernel(K)  # [B, H, N, d] or [B, H, N, feature_dim]
        
        # 线性注意力核心: 改变计算顺序
        # 标准: QK^T @ V  -> O(n²d)
        # 线性: Q @ (K^T @ V)  -> O(nd²)
        
        # 步骤1: 计算 K^T @ V: [B, H, d, N] @ [B, H, N, d] = [B, H, d, d]
        KV = torch.matmul(K_prime.transpose(-2, -1), V)
        
        # 步骤2: 计算 Q @ KV: [B, H, N, d] @ [B, H, d, d] = [B, H, N, d]
        Z = torch.matmul(Q_prime, KV)
        
        # 步骤3: 归一化 (关键!)
        # 计算分母: Q @ (K^T @ 1)  [B, H, N, d] @ [B, H, d, 1] = [B, H, N, 1]
        K_sum = K_prime.sum(dim=-2, keepdim=True).transpose(-2, -1)  # [B, H, d, 1]
        normalizer = torch.matmul(Q_prime, K_sum) + 1e-6  # [B, H, N, 1]
        
        attn_output = Z / normalizer
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.out_proj(attn_output)

print("✅ D. Linear Attention 定义完成")
print("\n" + "=" * 80)


# ==============================================================================
# 测试与对比
# ==============================================================================

def test_attention_module(
    name: str, 
    module: nn.Module, 
    batch_size: int = 2, 
    seq_len: int = 512, 
    dim: int = 512,
    num_heads: int = 8
):
    """测试单个注意力模块的性能和输出"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, dim, device=device)
    
    # 前向传播
    module.eval()
    with torch.no_grad():
        # 预热
        _ = module(x)
        
        # 计时
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        output = module(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
    
    # 统计信息
    elapsed_time = (end_time - start_time) * 1000  # ms
    memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
    
    print(f"输入形状:  {x.shape}")
    print(f"输出形状:  {output.shape}")
    print(f"前向时间:  {elapsed_time:.2f} ms")
    if torch.cuda.is_available():
        print(f"显存使用:  {memory_mb:.2f} MB")
    print(f"参数数量:  {sum(p.numel() for p in module.parameters()):,}")
    
    # 检查输出有效性
    assert output.shape == x.shape, f"输出形状错误: {output.shape} != {x.shape}"
    assert not torch.isnan(output).any(), "输出包含NaN"
    assert not torch.isinf(output).any(), "输出包含Inf"
    
    print("✅ 测试通过")
    
    return {
        'name': name,
        'time_ms': elapsed_time,
        'memory_mb': memory_mb,
        'output': output
    }

# 运行所有测试
print("\n开始测试四种注意力机制...")
print("设备:", "CUDA" if torch.cuda.is_available() else "CPU")

results = []

# 测试配置
test_configs = [
    ("Vanilla Softmax Attention", VanillaSoftmaxAttention(512, num_heads=8)),
    ("Nyström Attention", NystromAttention(dim=512, num_heads=8, num_landmarks=64)),
    ("Dilated + Flash Attention", DilatedFlashAttention(512, num_heads=8, block_size=64, dilation=4)),
    ("Linear Attention", LinearAttention(512, num_heads=8, kernel_fn="elu+1")),
]

for name, module in test_configs:
    try:
        result = test_attention_module(name, module, batch_size=2, seq_len=512, dim=512)
        results.append(result)
    except Exception as e:
        print(f"❌ {name} 测试失败: {e}")
        import traceback
        traceback.print_exc()


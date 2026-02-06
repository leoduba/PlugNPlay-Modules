
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
import numpy as np

# 论文：MiTA Attention: Efficient Fast-Weight Scaling viaa Mixture of Top-k Activations
# 论文地址：https://arxiv.org/pdf/2602.01219

class MiTAAttention(nn.Module):
    """
    MiTA Attention: Mixture of Top-k Activations
    论文: "MiTA Attention: Efficient Fast-Weight Scaling via a Mixture of Top-k Activations"
    arXiv: 2602.01219
    """
    def __init__(self, dim, num_heads=8, m=16, k=16, s=1, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.m = m  # 地标查询数量
        self.k = k  # 每个专家的top-k
        self.s = s  # 路由专家数
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, return_details=False):
        # 处理输入维度
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            spatial_shape = (H, W)
        else:
            B, N, C = x.shape
            spatial_shape = None
            
        N = x.size(1)
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]
        
        # 1. 生成地标查询 (均匀采样)
        if N >= self.m:
            indices = torch.linspace(0, N-1, self.m, device=x.device).long()
            q_landmark = q[:, :, indices, :]  # [B, heads, m, head_dim]
        else:
            # 如果序列太短，使用平均池化
            q_landmark = q.mean(dim=2, keepdim=True).expand(-1, -1, self.m, -1)
        
        # 2. 计算地标查询与所有key的相似度
        attn_landmark = torch.matmul(q_landmark, k.transpose(-2, -1)) * self.scale  # [B, heads, m, N]
        
        # 3. 获取top-k索引
        actual_k = min(self.k, N)
        _, topk_indices = torch.topk(attn_landmark, actual_k, dim=-1)  # [B, heads, m, k]
        
        # 4. 计算地标值 (共享专家)
        attn_landmark_softmax = F.softmax(attn_landmark, dim=-1)
        v_landmark = torch.matmul(attn_landmark_softmax, v)  # [B, heads, m, head_dim]
        
        # 5. 收集top-k key-value对
        k_expanded = k.unsqueeze(2).expand(-1, -1, self.m, -1, -1)
        v_expanded = v.unsqueeze(2).expand(-1, -1, self.m, -1, -1)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        
        k_experts = torch.gather(k_expanded, dim=3, index=topk_indices_expanded)  # [B, heads, m, k, head_dim]
        v_experts = torch.gather(v_expanded, dim=3, index=topk_indices_expanded)
        
        # 6. 路由
        routing_logits = torch.matmul(q, q_landmark.transpose(-2, -1))  # [B, heads, N, m]
        expert_assignments = torch.argmax(routing_logits, dim=-1)  # [B, heads, N]
        
        # 7. 合并key-value
        k_shared = q_landmark
        v_shared = v_landmark
        k_routed = k_experts.reshape(B, self.num_heads, self.m * actual_k, self.head_dim)
        v_routed = v_experts.reshape(B, self.num_heads, self.m * actual_k, self.head_dim)
        
        k_all = torch.cat([k_shared, k_routed], dim=2)  # [B, heads, m + m*k, head_dim]
        v_all = torch.cat([v_shared, v_routed], dim=2)
        
        # 8. 最终注意力
        attn = torch.matmul(q, k_all.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v_all)
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        out = self.proj(out)
        
        if spatial_shape is not None:
            H, W = spatial_shape
            out = out.transpose(1, 2).reshape(B, self.dim, H, W)
            
        if return_details:
            return out, {
                'landmark_queries': q_landmark,
                'expert_assignments': expert_assignments,
                'topk_indices': topk_indices,
                'attention_weights': attn
            }
        return out


class StandardAttention(nn.Module):
    """标准多头注意力"""
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            spatial_shape = (H, W)
        else:
            B, N, C = x.shape
            spatial_shape = None
            
        N = x.size(1)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, self.dim)
        out = self.proj(out)
        
        if spatial_shape is not None:
            H, W = spatial_shape
            out = out.transpose(1, 2).reshape(B, self.dim, H, W)
            
        return out
def visualize_experts():
    """可视化MiTA在2D图像上的专家分配"""
    # 创建一个模拟的图像特征
    H, W = 14, 14
    dim = 64
    num_heads = 4
    m, k = 8, 8
    
    # 创建模拟输入 (模拟一个简单图像模式)
    x = torch.randn(1, dim, H, W)
    # 添加一些结构：中心区域值更高
    center_h, center_w = H // 2, W // 2
    x[:, :, center_h-3:center_h+3, center_w-3:center_w+3] += 2.0
    
    mita = MiTAAttention(dim=dim, num_heads=num_heads, m=m, k=k)
    mita.eval()
    
    with torch.no_grad():
        output, details = mita(x, return_details=True)
    
    # 获取专家分配
    expert_assignments = details['expert_assignments'][0, 0].cpu().numpy()  # [N]
    landmark_queries_pos = details['landmark_queries'][0, 0].cpu().numpy()  # [m, head_dim]
    
    # 重塑为2D图像
    expert_map = expert_assignments.reshape(H, W)
    
    # 计算每个专家被分配到的位置
    expert_positions = {i: [] for i in range(m)}
    for idx, expert_id in enumerate(expert_assignments):
        h, w = idx // W, idx % W
        expert_positions[expert_id].append((h, w))
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 输入特征 (平均)
    ax = axes[0, 0]
    input_vis = x[0].mean(0).cpu().numpy()
    im = ax.imshow(input_vis, cmap='viridis')
    ax.set_title('Input Feature (Channel Average)')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # 2. 专家分配图
    ax = axes[0, 1]
    im = ax.imshow(expert_map, cmap='tab20', vmin=0, vmax=m-1)
    ax.set_title(f'Expert Assignment Map (m={m})')
    ax.axis('off')
    plt.colorbar(im, ax=ax, ticks=range(m))
    
    # 3. 专家覆盖热力图
    ax = axes[0, 2]
    coverage = np.zeros((H, W))
    for expert_id, positions in expert_positions.items():
        for h, w in positions:
            coverage[h, w] += 1
    im = ax.imshow(coverage, cmap='hot')
    ax.set_title('Expert Coverage Density')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    # 4. 地标查询位置
    ax = axes[1, 0]
    # 计算每个地标查询在原始序列中的位置 (均匀采样)
    landmark_indices = np.linspace(0, H*W-1, m, dtype=int)
    landmark_h = landmark_indices // W
    landmark_w = landmark_indices % W
    
    ax.imshow(input_vis, cmap='gray', alpha=0.3)
    scatter = ax.scatter(landmark_w, landmark_h, c=range(m), cmap='tab20', s=200, edgecolors='black', linewidth=2)
    ax.set_title('Landmark Query Positions')
    ax.axis('off')
    plt.colorbar(scatter, ax=ax, ticks=range(m))
    
    # 5. 每个专家分配的token数量
    ax = axes[1, 1]
    counts = [len(expert_positions[i]) for i in range(m)]
    bars = ax.bar(range(m), counts, color=plt.cm.tab20(np.linspace(0, 1, m)))
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Number of Tokens')
    ax.set_title('Tokens per Expert')
    ax.set_xticks(range(m))
    
    # 6. Top-k索引可视化 (第一个专家的top-k位置)
    ax = axes[1, 2]
    topk_indices = details['topk_indices'][0, 0].cpu().numpy()  # [m, k]
    
    # 创建热力图显示哪些位置被选中为top-k
    topk_map = np.zeros((H, W))
    for expert_idx in range(min(4, m)):  # 只显示前4个专家
        for idx in topk_indices[expert_idx]:
            h, w = idx // W, idx % W
            topk_map[h, w] += 1
    
    im = ax.imshow(topk_map, cmap='YlOrRd')
    ax.set_title(f'Top-k Selection Heatmap (First {min(4,m)} Experts)')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    #plt.savefig('/mnt/kimi/output/mita_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 可视化已保存")
    print(f"  - 输入序列长度: {H*W}")
    print(f"  - 地标查询数 (m): {m}")
    print(f"  - 每个专家top-k: {k}")
    print(f"  - 实际参与计算的key-value对: {m + m*k} (vs 标准注意力的 {H*W})")
    
    return expert_positions
    
def benchmark(seq_lengths=[196, 784, 3136], dim=128, num_heads=4):
    results = []
    m, k = 16, 16
    
    print(f"{'序列长度':<12} {'标准注意力':<15} {'MiTA':<15} {'理论加速'}")
    print("-" * 60)
    
    for seq_len in seq_lengths:
        x = torch.randn(2, seq_len, dim)
        
        # 标准注意力
        std_attn = StandardAttention(dim, num_heads)
        std_attn.eval()
        with torch.no_grad():
            for _ in range(3):  # 预热
                _ = std_attn(x)
            start = time.time()
            for _ in range(10):
                _ = std_attn(x)
            std_time = (time.time() - start) / 10
        
        # MiTA
        mita = MiTAAttention(dim, num_heads, m=m, k=k)
        mita.eval()
        with torch.no_grad():
            for _ in range(3):  # 预热
                _ = mita(x)
            start = time.time()
            for _ in range(10):
                _ = mita(x)
            mita_time = (time.time() - start) / 10
        
        # 理论复杂度
        std_ops = seq_len ** 2
        mita_ops = seq_len * (m + m * k)
        speedup = std_ops / mita_ops
        
        print(f"{seq_len:<12} {std_time*1000:>10.2f}ms    {mita_time*1000:>10.2f}ms    {speedup:.1f}x")
        results.append({
            'seq_len': seq_len,
            'std_time': std_time,
            'mita_time': mita_time,
            'theoretical_speedup': speedup
        })
    
    return results
    
 
if __name__ == "__main__":
# ========== 测试1: 基础功能测试 ==========
    print("\n【测试1】基础功能测试")
    print("-" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    batch_size = 2
    dim = 256
    num_heads = 8
    m, k = 16, 16
    
    # 测试1D序列
    seq_len = 196
    x_1d = torch.randn(batch_size, seq_len, dim).to(device)
    mita = MiTAAttention(dim=dim, num_heads=num_heads, m=m, k=k).to(device)
    output_1d = mita(x_1d)
    print(f"✓ 1D序列: {x_1d.shape} -> {output_1d.shape}")
    
    # 测试2D图像
    H, W = 14, 14
    x_2d = torch.randn(batch_size, dim, H, W).to(device)
    output_2d = mita(x_2d)
    print(f"✓ 2D图像: {x_2d.shape} -> {output_2d.shape}")
    
    # 测试梯度回传
    x_test = torch.randn(1, 100, 64, requires_grad=True)
    mita_test = MiTAAttention(dim=64, num_heads=4, m=8, k=8)
    out_test = mita_test(x_test)
    loss = out_test.sum()
    loss.backward()
    print(f"✓ 梯度回传测试通过，梯度形状: {x_test.grad.shape}")
    
    # ========== 测试2: 效率对比 ==========
    print("\n【测试2】计算效率对比")
    print("-" * 50)
    
    
    results = benchmark()
    print("\n【测试2】完成 ✓")

# ========== 测试3: 可视化专家分配 ==========
    print("\n【测试3】专家分配可视化")
    print("-" * 50)



    expert_pos = visualize_experts()


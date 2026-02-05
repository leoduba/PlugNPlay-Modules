import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
# 论文地址：https://arxiv.org/abs/2305.13245
class GQA(nn.Module):
    """
    标准GQA（Grouped Query Attention）
    核心：n_q_heads个查询头 → 分为n_groups组 → 每组共享1个KV头（共n_kv_heads=n_q_heads/n_groups个）
    输入输出维度与MHA一致，效率高于MHA、性能优于MQA
    """
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_k, d_v, dropout=0.1, eps=1e-8):
        super().__init__()
        self.d_model = d_model
        self.n_q_heads = n_q_heads  # 查询头数
        self.n_kv_heads = n_kv_heads# 键/值头数
        self.n_groups = n_q_heads // n_kv_heads  # 分组数（必须整除）
        self.d_k = d_k
        self.d_v = d_v
        self.eps = eps
        self.dropout = nn.Dropout(dropout)

        # 线性投影层：Q投影为n_q_heads个，K/V投影为n_kv_heads个（GQA核心差异）
        self.w_q = nn.Linear(d_model, n_q_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * d_v, bias=False)
        self.w_o = nn.Linear(n_q_heads * d_v, d_model, bias=False)

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(self, Q, K, V, mask=None):
        """
        GQA前向：Q/K/V投影 → 分组适配 → 点积注意力 → 合并输出
        Input: Q/K/V [b, seq, d_model], mask [b, 1, seq, seq]
        Output: gqa_feat [b, seq, d_model] 分组查询注意力特征
        """
        b, seq_q, _ = Q.shape
        _, seq_kv, _ = K.shape

        # 1. Q/K/V线性投影
        q = self.w_q(Q).view(b, seq_q, self.n_q_heads, self.d_k)  # [b, seq_q, n_qh, d_k]
        k = self.w_k(K).view(b, seq_kv, self.n_kv_heads, self.d_k)  # [b, seq_kv, n_kvh, d_k]
        v = self.w_v(V).view(b, seq_kv, self.n_kv_heads, self.d_v)  # [b, seq_kv, n_kvh, d_v]

        # 2. GQA核心：查询头分组，KV头扩展以匹配分组（每组共享1个KV头）
        # Q分组：[b, seq_q, n_kvh, n_groups, d_k] → 合并为[n_qh, d_k]
        q = q.view(b, seq_q, self.n_kv_heads, self.n_groups, self.d_k).transpose(2, 3)
        q = q.contiguous().view(b, seq_q, self.n_q_heads, self.d_k)
        # K/V扩展：[b, seq_kv, n_kvh, d_k] → [b, seq_kv, n_qh, d_k]（每组复制1次）
        k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).contiguous().view(b, seq_kv, self.n_q_heads, self.d_k)
        v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).contiguous().view(b, seq_kv, self.n_q_heads, self.d_v)

        # 3. 维度转置：适配点积注意力 [b, n_heads, seq, d_k/d_v]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. 点积注意力计算（标准缩放点积）
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        # 掩码
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, -1e9)
        # 归一化+Dropout
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        # 5. 加权求和+合并输出
        gqa_feat = torch.matmul(attn_weight, v)  # [b, n_qh, seq_q, d_v]
        gqa_feat = gqa_feat.transpose(1, 2).contiguous().view(b, seq_q, -1)  # [b, seq_q, n_qh*d_v]
        gqa_feat = self.dropout(self.w_o(gqa_feat))  # [b, seq_q, d_model]

        return gqa_feat

# -------------------------- 测试代码 --------------------------
if __name__ == "__main__":
    # 超参数设置
    d_model = 512    # 模型特征维度
    n_q_heads = 8    # GQA查询头数（与MHA一致）
    n_kv_heads = 4   # GQA键/值头数（需为n_q_heads的约数，8/4=2组，每组2个Q头共享1个KV头）
    n_groups = n_q_heads // n_kv_heads  # 分组数，核心GQA参数
    d_k = d_model // n_q_heads          # 每个查询头的维度
    d_v = d_model // n_q_heads          # 每个键/值头的维度
    batch_size = 4
    seq_len = 10
    eps = 1e-8       # 数值稳定性
    dropout = 0.1

    # 初始化模型
    gqa = GQA(d_model, n_q_heads, n_kv_heads, d_k, d_v)
    q_gqa = torch.randn(batch_size, seq_len, d_model)
    k_gqa = torch.randn(batch_size, seq_len, d_model)
    v_gqa = torch.randn(batch_size, seq_len, d_model)
    gqa_out = gqa(q_gqa, k_gqa, v_gqa)
    print(f"GQA输入维度：{q_gqa.shape}, 输出维度：{gqa_out.shape}")  # 维度一致
# GQA测试

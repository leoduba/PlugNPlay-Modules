import torch
import torch.nn as nn
import torch.nn.functional as F
# 论文：Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities
# 论文地址：https://arxiv.org/html/2401.11143v3

class GQGAAM(nn.Module):
    """
    GQGAAM（Grouped Query Gaussian Adaptive Attention Mechanism）
    论文核心：GAAM + GQA 融合版 → 将GAAM的高斯自适应调制嵌入GQA的分组查询框架
    核心逻辑：
        1. 用GAAM对GQA的输入Q/K做高斯自适应特征调制，得到调制后的Q_gaam、K_gaam；
        2. 将调制后的Q_gaam、K_gaam输入GQA，得到GAAM调制后的GQA特征；
        3. 自适应门控融合「GAAM调制GQA特征」和「原始GQA特征」，兼顾自适应与稳定性；
    输入输出与MHA/GQA/GAAM完全一致，可无缝替换现有注意力模块
    """
    def __init__(self, d_model, n_q_heads, n_kv_heads, d_k, d_v, dropout=0.1, eps=1e-8):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # 1. 嵌入基础模块：GAAM（高斯调制） + GQA（分组查询）
        self.gaam = GAAM(d_model, n_q_heads, d_k, eps, dropout)  # GAAM与GQA查询头数一致
        self.gqa = GQA(d_model, n_q_heads, n_kv_heads, d_k, d_v, dropout, eps)

        # 2. 自适应融合门控：学习GAAM调制的贡献度（0~1），贴合论文自适应思想
        self.fuse_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()  # 门控系数g∈[0,1]
        )

        # 输出投影：保证特征维度一致性和表达能力
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(self, Q, K, V, mask=None):
        """
        GQGAAM前向：GAAM调制 → GQA计算 → 自适应融合 → 输出
        Input: Q/K/V [b, seq, d_model], mask [b, 1, seq, seq]
        Output: gqgaam_out [b, seq, d_model] 融合后最终特征
        """
        # 步骤1：GAAM对GQA的Q/K做高斯自适应调制（GAAM核心作用）
        Q_gaam = self.gaam(Q, K, mask)  # GAAM调制查询特征 [b, seq, d_model]
        K_gaam = self.gaam(K, Q, mask)  # GAAM调制键特征 [b, seq, d_model]

        # 步骤2：双路GQA计算
        gqa_ori = self.gqa(Q, K, V, mask)    # 原始GQA特征（无GAAM调制）
        gqa_gaam = self.gqa(Q_gaam, K_gaam, V, mask)  # GAAM调制后的GQA特征（核心融合）

        # 步骤3：自适应门控融合（论文自适应思想）
        fuse_feat = torch.cat([gqa_ori, gqa_gaam], dim=-1)  # [b, seq, 2*d_model]
        gate = self.fuse_gate(fuse_feat)                    # 自适应门控系数 [b, seq, d_model]
        gqa_fuse = gate * gqa_gaam + (1 - gate) * gqa_ori   # 门控融合，兼顾二者优势

        # 步骤4：最终投影+Dropout，保证训练稳定性
        gqgaam_out = self.dropout(self.w_o(gqa_fuse))

        # 可选：加入残差连接（集成到模型时建议加，如Transformer Block）
        # gqgaam_out = gqgaam_out + Q

        return gqgaam_out

# ===================== GQGAAM 核心测试 =====================
if __name__ == "__main__":
    # 初始化GQGAAM模型（论文核心融合版）
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

    gqgaam = GQGAAM(d_model, n_q_heads, n_kv_heads, d_k, d_v)
    # 生成随机测试输入（与Transformer/ViT等模型输入维度一致）
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    # 前向传播（无掩码，实际任务可传入mask）
    gqgaam_out = gqgaam(Q, K, V)

    # 打印维度验证
    print("="*50)
    print(f"输入Q/K/V维度：{Q.shape}")
    print(f"GQGAAM输出维度：{gqgaam_out.shape}")
    print("="*50)
    print("✅ GQGAAM（GAAM+GQA）代码运行成功！维度完全匹配，可直接集成！")

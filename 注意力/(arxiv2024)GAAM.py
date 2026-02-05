class GAAM(nn.Module):
    """
    纯GAAM（Gaussian Adaptive Attention Mechanism）
    无点积注意力，仅实现高斯自适应核心：可学习高斯核权重+特征自适应调制
    输出：高斯自适应调制后的特征（与输入维度一致，供GQA融合）
    """
    def __init__(self, d_model, n_heads, d_k, eps=1e-8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads  # 与GQA的查询头数一致
        self.d_k = d_k
        self.eps = eps
        self.dropout = nn.Dropout(dropout)

        # 线性投影：将输入特征映射为Q/K（GAAM无需V，高斯核直接调制Q/K特征）
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        # 输出投影+残差连接适配
        self.w_o = nn.Linear(n_heads * d_k, d_model, bias=False)

        # 高斯自适应参数MLP：生成每个头的高斯方差σ（可学习，>0）
        self.sigma_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_heads)
        )
        self.softplus = nn.Softplus()  # 保证σ>0

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        for m in self.sigma_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def _split_heads(self, x):
        """拆分多头：[b, seq, d_model] → [b, n_heads, seq, d_k]"""
        b, seq, _ = x.shape
        return x.view(b, seq, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        """合并多头：[b, n_heads, seq, d_k] → [b, seq, d_model]"""
        b, _, seq, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, seq, -1)

    def forward(self, Q, K, mask=None):
        """
        GAAM前向：高斯核建模自适应权重 → 调制特征 → 投影输出
        Input: Q/K [b, seq, d_model], mask [b, 1, seq, seq]
        Output: gaam_feat [b, seq, d_model] 高斯调制后的特征
        """
        b, seq, _ = Q.shape
        # 1. Q/K投影+多头拆分
        q_proj = self._split_heads(self.w_q(Q))  # [b, h, seq, d_k]
        k_proj = self._split_heads(self.w_k(K))  # [b, h, seq, d_k]

        # 2. 生成可学习高斯方差σ（每个头独立σ，自适应）
        q_avg = Q.mean(dim=1)  # [b, d_model] 全局平均池化
        k_avg = K.mean(dim=1)
        sigma = self.sigma_mlp(torch.cat([q_avg, k_avg], dim=-1))  # [b, h]
        sigma = self.softplus(sigma) + self.eps  # 保证σ>0
        sigma = sigma.unsqueeze(-1).unsqueeze(-1)  # [b, h, 1, 1] 广播适配

        # 3. 高斯核自适应权重计算（GAAM核心公式）
        qk_diff = q_proj.unsqueeze(3) - k_proj.unsqueeze(2)  # [b, h, seq, seq, d_k]
        l2_sq = torch.sum(qk_diff ** 2, dim=-1)  # ||Q-K||² [b, h, seq, seq]
        gaam_weight = torch.exp(-l2_sq / (2 * sigma ** 2 + self.eps))  # 高斯核权重

        # 4. 掩码+归一化
        if mask is not None:
            gaam_weight = gaam_weight.masked_fill(mask, -1e9)
        gaam_weight = F.softmax(gaam_weight, dim=-1)
        gaam_weight = self.dropout(gaam_weight)

        # 5. 高斯自适应特征调制（加权K特征，作为GAAM输出）
        gaam_feat = torch.matmul(gaam_weight, k_proj)  # [b, h, seq, d_k]
        gaam_feat = self._merge_heads(gaam_feat)        # [b, seq, h*d_k]
        gaam_feat = self.dropout(self.w_o(gaam_feat))   # [b, seq, d_model]

        return gaam_feat

# GAAM测试

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
    gaam = GAAM(d_model, n_q_heads, d_k)
    q_gaam = torch.randn(batch_size, seq_len, d_model)
    k_gaam = torch.randn(batch_size, seq_len, d_model)
    gaam_out = gaam(q_gaam, k_gaam)
    print(f"GAAM输入维度：{q_gaam.shape}, 输出维度：{gaam_out.shape}")  # 维度一致
     

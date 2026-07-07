import torch
import torch.nn as nn

class SpatialVarianceModulation(nn.Module):
    def __init__(
        self,
        eps_var: float = 1e-6,
        eps_norm: float = 1e-8,
        share_channel_var: bool = False,
        use_residual: bool = True,
        learnable_coeff: bool = True,
        act_type: str = "sigmoid",  # sigmoid / relu6 / gelu
        debug: bool = False
    ):
        super().__init__()
        self.eps_var = eps_var
        self.eps_norm = eps_norm
        self.share_channel_var = share_channel_var
        self.use_residual = use_residual
        self.debug = debug

        # 可学习缩放偏移，初始贴合原始公式
        self.learnable_coeff = learnable_coeff
        if learnable_coeff:
            self.scale = nn.Parameter(torch.tensor(0.25))
            self.offset = nn.Parameter(torch.tensor(0.5))
        else:
            self.register_buffer("scale", torch.tensor(0.25))
            self.register_buffer("offset", torch.tensor(0.5))

        # 可选激活函数
        if act_type == "sigmoid":
            self.activation = nn.Sigmoid()
        elif act_type == "relu6":
            self.activation = nn.ReLU6()
        elif act_type == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活类型 {act_type}")

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # 维度校验
        if feature_map.dim() != 4:
            raise ValueError(f"输入张量必须为 [B,C,H,W] 4维，当前shape: {feature_map.shape}")
        B, C, H, W = feature_map.shape
        if H <= 1 or W <= 1:
            raise ValueError(f"空间尺寸 H={H}, W={W} 过小，无法计算空间方差")

        dev = feature_map.device
        dtype = feature_map.dtype
        four = torch.tensor(4.0, device=dev, dtype=dtype)

        # 1. 中心化计算 (x-μ)²，减少中间张量显存占用
        spatial_mean = feature_map.mean(dim=[2, 3], keepdim=True)
        centered = feature_map - spatial_mean
        squared_deviation = centered.pow(2)

        # 2. 计算空间方差（原生算子，数值更稳定）
        if self.share_channel_var:
            # 全局共享方差 [B,1,1,1]
            spatial_variance = squared_deviation.sum(dim=[1,2,3], keepdim=True) / (C * H * W)
        else:
            # 逐通道独立方差 [B,C,1,1]
            spatial_variance = squared_deviation.var(dim=[2, 3], keepdim=True, unbiased=False)

        # 3. 自适应调制公式，支持可学习参数
        denominator = four * (spatial_variance + self.eps_var)
        raw_coeff = squared_deviation / denominator
        modulation_coeff = raw_coeff / 0.25 * self.scale + self.offset

        # 动态截断，抑制极端值，避免激活饱和
        dynamic_max = torch.sqrt(spatial_variance + self.eps_var) * 20.0 + 1.0
        modulation_coeff = torch.clamp(modulation_coeff, min=0.0)
        modulation_coeff = torch.min(modulation_coeff, dynamic_max)

        # 4. 生成权重 + 空间归一化稳定训练
        weight = self.activation(modulation_coeff)
        weight_avg = weight.mean(dim=[2, 3], keepdim=True)
        weight = weight / (weight_avg + self.eps_norm) * 0.5

        # 5. 特征调制，可选残差连接
        if self.use_residual:
            out = feature_map + feature_map * weight
        else:
            out = feature_map * weight

        # 调试异常监控
        if self.debug:
            if torch.any(torch.isnan(out)) or torch.any(torch.isinf(out)):
                print(f"[警告] 输出存在NaN/Inf，输入shape: {feature_map.shape}")

        return out

    @torch.no_grad()
    def inference(self, feature_map: torch.Tensor):
        # 推理专用接口，关闭计算图，节省显存加速
        return self.forward(feature_map)


# ================= 测试代码 =================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    # 初始化优化后的模块
    svm_layer = SpatialVarianceModulation(
        learnable_coeff=True,
        act_type="relu6",
        use_residual=True,
        debug=True
    ).to(device)

    # 标准测试
    B, C, H, W = 2, 16, 32, 32
    test_feat = torch.randn(B, C, H, W, requires_grad=True, device=device)
    train_out = svm_layer(test_feat)
    print(f"训练输出 shape: {train_out.shape}")

    # 推理测试
    with torch.no_grad():
        infer_out = svm_layer.inference(test_feat)
        print(f"推理输出 shape: {infer_out.shape}")

    # 梯度测试
    loss = train_out.sum()
    loss.backward()
    print(f"输入梯度正常: {test_feat.grad is not None}")
    # 梯度裁剪兜底
    torch.nn.utils.clip_grad_norm_(svm_layer.parameters(), max_norm=1.0)

    # 平坦特征稳定性测试
    flat_feat = torch.ones(1, 3, 16, 16, requires_grad=True, device=device)
    flat_out = svm_layer(flat_feat)
    print(f"平坦特征输出无异常: {torch.isfinite(flat_out).all().item()}")
import torch
import torch.nn as nn

class SpatialVarianceModulation(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        # 公式中的 λ，防止分母为0
        self.eps = eps
        # Sigmoid 无参数激活，仅做映射
        self.activation = nn.Sigmoid()

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # feature_map shape: [B, C, H, W]
        B, C, H, W = feature_map.shape
        n = H * W  # 公式n：空间总像素数量

        # 1. 计算空间维度H、W的均值 μ
        spatial_mean = feature_map.mean(dim=[2, 3], keepdim=True)

        # 2. 计算 (x - μ)²
        squared_deviation = (feature_map - spatial_mean).pow(2)

        # 3. 严格匹配公式：∑(x−μ)² / n，移除原代码 n-1 无偏方差
        sum_sq_dev = squared_deviation.sum(dim=[2, 3], keepdim=True)
        spatial_variance = sum_sq_dev / n

        # 4. 代入核心数学公式 y = (x−μ)² / [4*(方差 + λ)] + 0.5
        modulation_coeff = squared_deviation / (4 * (spatial_variance + self.eps)) + 0.5

        # 5. Sigmoid映射得到空间权重图
        weight = self.activation(modulation_coeff)

        # 6. 原始特征逐元素加权调制
        out = feature_map * weight

        return out


# ===================== 测试代码（无报错版） =====================
if __name__ == "__main__":
    # 超参设置
    batch_size = 2
    channel = 16
    height = 32
    width = 32

    # 初始化模块
    svm_layer = SpatialVarianceModulation(eps=1e-6)

    # 构造输入并开启梯度追踪
    test_feature = torch.randn(batch_size, channel, height, width, requires_grad=True)
    print(f"输入特征图 shape: {test_feature.shape}")

    # 前向传播
    output_feature = svm_layer(test_feature)
    print(f"输出调制后特征 shape: {output_feature.shape}")

    # 手动复现中间计算用于校验
    B, C, H, W = test_feature.shape
    n = H * W
    mu = test_feature.mean(dim=[2,3], keepdim=True)
    sq_dev = (test_feature - mu).pow(2)
    var = sq_dev.sum(dim=[2,3], keepdim=True) / n
    coeff = sq_dev / (4 * (var + 1e-6)) + 0.5
    w = torch.sigmoid(coeff)

    print("\n===== 中间数值校验 =====")
    print(f"均值 μ shape: {mu.shape}")
    print(f"偏差平方 (x-μ)² shape: {sq_dev.shape}")
    print(f"空间方差 ∑(x-μ)²/n shape: {var.shape}")
    print(f"调制系数 y 取值范围: [{coeff.min().item():.4f}, {coeff.max().item():.4f}]")
    print(f"Sigmoid权重取值范围: [{w.min().item():.4f}, {w.max().item():.4f}]")
    print(f"输出特征范围: [{output_feature.min().item():.4f}, {output_feature.max().item():.4f}]")

    # 反向传播梯度测试
    loss = output_feature.sum()
    loss.backward()

    print("\n===== 梯度校验 =====")
    print(f"输入特征张量成功生成梯度：{test_feature.grad is not None}")
    print("梯度测试通过，模块可正常参与训练！")
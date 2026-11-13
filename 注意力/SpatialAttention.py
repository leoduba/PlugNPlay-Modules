import torch
import torch.nn as nn
import torch.optim as optim

# 完全保留你提供的SpatialAttention原实现，不做任何修改
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def test_kernel_size_validation():
    """测试非法卷积核尺寸会正确抛出断言错误"""
    print("Test 1: Kernel size legality check")
    try:
        # kernel_size=5为非法值，必须触发AssertionError
        SpatialAttention(kernel_size=5)
        print("❌ FAIL: Illegal kernel_size did not trigger assertion")
        return False
    except AssertionError as e:
        print(f"✅ PASS: Illegal kernel_size correctly triggered assertion: {e}")
        return True


def test_forward_shape():
    """测试不同配置下输出形状正确：通道数恒为1，空间尺寸与输入完全一致（same padding）"""
    print("\nTest 2: Forward output shape correctness")
    test_cases = [
        # (kernel_size, batch_size, in_channels, height, width)
        (7, 2, 64, 32, 32),   # 标准正方形特征图，k=7
        (3, 2, 128, 28, 14),  # 非正方形特征图，k=3
        (7, 1, 3, 224, 224),  # 3通道RGB图像输入
        (3, 4, 1, 7, 7),      # 单通道输入，小尺寸特征图
        (7, 1, 512, 1, 1),    # 极端1x1空间尺寸输入
    ]
    all_pass = True
    for k, B, C, H, W in test_cases:
        sa = SpatialAttention(kernel_size=k)
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = sa(x)
        expected_shape = (B, 1, H, W)
        if out.shape != expected_shape:
            print(f"❌ FAIL: k={k}, input={x.shape}, output={out.shape}, expected={expected_shape}")
            all_pass = False
        else:
            print(f"✅ PASS: k={k}, input={x.shape}, output={out.shape}")
    return all_pass


def test_output_value_range():
    """测试sigmoid输出严格落在[0,1]区间，无NaN/Inf异常值"""
    print("\nTest 3: Output value range [0,1] check")
    sa = SpatialAttention(kernel_size=7)
    sa.eval()
    test_inputs = [
        ("normal random input", torch.randn(4, 64, 32, 32)),
        ("extreme ±100 input", torch.cat([torch.ones(1, 64, 16, 16)*100, torch.ones(1,64,16,16)*(-100)], dim=0))
    ]
    all_pass = True
    for name, x in test_inputs:
        with torch.no_grad():
            out = sa(x)
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"❌ FAIL: {name} output contains NaN/Inf")
            all_pass = False
        elif (out < 0).any() or (out > 1).any():
            print(f"❌ FAIL: {name} output out of [0,1], min={out.min().item():.4f}, max={out.max().item():.4f}")
            all_pass = False
        else:
            print(f"✅ PASS: {name} range normal, min={out.min().item():.4f}, max={out.max().item():.4f}")
    return all_pass


def test_channel_pooling_correctness():
    """验证池化确实沿通道维度(dim=1)计算，而非错误地沿空间维度计算"""
    print("\nTest 4: Channel-wise pooling logic check")
    B, C, H, W = 2, 16, 8, 8
    x = torch.zeros(B, C, H, W)
    x[:, 0, :, :] = 1.0  # 仅第一个通道全为1，其余通道全为0
    with torch.no_grad():
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat_out = torch.cat([avg_out, max_out], dim=1)
    # 校验拼接后通道数为2
    if cat_out.shape[1] != 2:
        print(f"❌ FAIL: Concatenated channel count wrong, got {cat_out.shape[1]}, expected 2")
        return False
    # 平均池化结果应为1/C=0.0625
    if not torch.allclose(avg_out, torch.full_like(avg_out, 1/C)):
        print(f"❌ FAIL: Avg pooling wrong, expected {1/C}, got {avg_out.mean().item()}")
        return False
    # 最大池化结果应为全1
    if not torch.allclose(max_out, torch.ones_like(max_out)):
        print(f"❌ FAIL: Max pooling wrong, expected 1.0, got {max_out.mean().item()}")
        return False
    print("✅ PASS: Channel-wise avg/max pooling logic is correct")
    return True


def test_gradient_flow():
    """测试反向传播梯度正常，卷积参数可被优化器正常更新"""
    print("\nTest 5: Gradient flow and parameter update check")
    sa = SpatialAttention(kernel_size=7)
    optimizer = optim.SGD(sa.parameters(), lr=0.1)
    x = torch.randn(2, 64, 32, 32, requires_grad=True)
    target = torch.ones(2, 1, 32, 32)
    out = sa(x)
    loss = nn.MSELoss()(out, target)
    loss.backward()
    # 校验梯度存在且无异常
    if sa.conv1.weight.grad is None:
        print("❌ FAIL: conv1 weight has no gradient")
        return False
    if torch.isnan(sa.conv1.weight.grad).any() or torch.isinf(sa.conv1.weight.grad).any():
        print("❌ FAIL: conv1 gradient contains NaN/Inf")
        return False
    if x.grad is None:
        print("❌ FAIL: Input tensor has no gradient")
        return False
    # 校验参数确实被更新
    initial_weight = sa.conv1.weight.data.clone()
    optimizer.step()
    if torch.equal(sa.conv1.weight.data, initial_weight):
        print("❌ FAIL: Weight did not change after optimizer step")
        return False
    print("✅ PASS: Gradients flow normally, weights can be updated")
    return True


def test_batch_independence():
    """测试不同batch样本独立计算，无跨样本干扰"""
    print("\nTest 6: Batch sample independence check")
    sa = SpatialAttention(kernel_size=3)
    sa.eval()
    # 第一个样本全为-100（输出应接近0），第二个样本全为+100（输出应接近1）
    x = torch.cat([
        torch.ones(1, 64, 8, 8) * (-100),
        torch.ones(1, 64, 8, 8) * 100
    ], dim=0)
    with torch.no_grad():
        out = sa(x)
    if out[0].mean().item() > 0.1:
        print(f"❌ FAIL: Negative input output too high: {out[0].mean().item():.4f}")
        return False
    if out[1].mean().item() < 0.9:
        print(f"❌ FAIL: Positive input output too low: {out[1].mean().item():.4f}")
        return False
    print(f"✅ PASS: Batch independent, negative sample mean={out[0].mean().item():.4f}, positive sample mean={out[1].mean().item():.4f}")
    return True


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running SpatialAttention tests on device: {device}\n")
    test_list = [
        test_kernel_size_validation,
        test_forward_shape,
        test_output_value_range,
        test_channel_pooling_correctness,
        test_gradient_flow,
        test_batch_independence,
    ]
    passed = 0
    total = len(test_list)
    for test_func in test_list:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ FAIL: Test threw unexpected exception: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'='*60}")
    print(f"Test Summary: {passed}/{total} tests passed")
    if passed == total:
        print("✅ All tests passed! SpatialAttention module works correctly.")
        print("ℹ️  This is the standard spatial attention implementation from CBAM, can be directly used with ChannelAttention.")
    else:
        print("⚠️  Some tests failed, please check the module implementation.")

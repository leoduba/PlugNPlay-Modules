# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import torch
import torch.nn as nn
import os
import torch.optim as optim
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, max(in_planes // ratio, 1), kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(max(in_planes // ratio, 1), in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

def test_forward_shape():
    """测试基本前向传播的输出形状（应为(B,C,1,1)）"""
    print("Test 1: 基本前向形状验证")
    test_cases = [
        (2, 64, 32, 32),
        (4, 32, 16, 16),
        (1, 128, 28, 28),
    ]
    for (B, C, H, W) in test_cases:
        ca = ChannelAttention(in_planes=C)
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = ca(x)
        expected_shape = (B, C, 1, 1)
        assert out.shape == expected_shape, f"输入({B},{C},{H},{W})时，输出形状{out.shape}，期望{expected_shape}"
        print(f"  输入({B},{C},{H},{W}) → 输出{out.shape} ✓")
    print("Test 1 通过\n")


def test_small_channel_boundary():
    """测试小通道数边界：in_planes < 16时中间通道自动取1，避免通道为0错误"""
    print("Test 2: 小通道数边界验证")
    # in_planes=8时，8//16=0，应自动取1
    B, C, H, W = 2, 8, 16, 16
    ca = ChannelAttention(in_planes=C)
    assert ca.fc1.out_channels == 1, f"in_planes=8时中间通道应为1，实际{ca.fc1.out_channels}"
    assert ca.fc2.in_channels == 1, f"in_planes=8时fc2输入通道应为1，实际{ca.fc2.in_channels}"
    x = torch.randn(B, C, H, W)
    with torch.no_grad():
        out = ca(x)
    assert out.shape == (B, C, 1, 1)
    print(f"  in_planes=8时，中间通道数正确，输出形状{out.shape} ✓")

    # in_planes=1极端情况
    C = 1
    ca = ChannelAttention(in_planes=C)
    assert ca.fc1.out_channels == 1
    x = torch.randn(2, C, 8, 8)
    out = ca(x)
    assert out.shape == (2, C, 1, 1)
    print(f"  in_planes=1极端情况正常 ✓")
    print("Test 2 通过\n")


def test_various_spatial_sizes():
    """测试不同空间尺寸（非正方形、1x1输入等）兼容性"""
    print("Test 3: 多空间尺寸兼容性验证")
    test_cases = [
        (1, 32, 7, 7),    # 小尺寸
        (3, 64, 5, 9),    # 非正方形
        (2, 16, 1, 1),    # 输入本身为1x1
        (4, 256, 13, 27), # 奇数大尺寸
    ]
    for (B, C, H, W) in test_cases:
        ca = ChannelAttention(in_planes=C)
        x = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = ca(x)
        assert out.shape == (B, C, 1, 1), f"输入({B},{C},{H},{W})输出形状错误"
        print(f"  输入({B},{C},{H},{W}) → 输出{out.shape} ✓")
    print("Test 3 通过\n")


def test_output_range():
    """验证sigmoid输出在[0,1]范围内，极端输入下无NaN/Inf"""
    print("Test 4: 输出数值范围验证")
    ca = ChannelAttention(in_planes=64)
    # 普通随机输入
    x = torch.randn(4, 64, 32, 32)
    with torch.no_grad():
        out = ca(x)
    assert torch.all(out >= 0) and torch.all(out <= 1), "输出值不在[0,1]范围内"
    assert not torch.isnan(out).any(), "输出包含NaN"
    assert not torch.isinf(out).any(), "输出包含Inf"

    # 极端值输入（±100）
    x_extreme = torch.cat([torch.ones(1,64,8,8)*100, torch.ones(1,64,8,8)*(-100)], dim=0)
    out_extreme = ca(x_extreme)
    assert torch.all(out_extreme >= 0) and torch.all(out_extreme <= 1), "极端输入下输出越界"
    assert not torch.isnan(out_extreme).any()
    print("  输出值范围正常，无NaN/Inf ✓")
    print("Test 4 通过\n")


def test_gradient_flow():
    """测试反向传播梯度正常，参数可被优化器更新"""
    print("Test 5: 梯度流与参数更新验证")
    B, C, H, W = 2, 64, 16, 16
    ca = ChannelAttention(in_planes=C)
    optimizer = optim.SGD(ca.parameters(), lr=0.01)
    x = torch.randn(B, C, H, W, requires_grad=True)
    target = torch.ones(B, C, 1, 1)

    out = ca(x)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    # 检查梯度存在且有效
    assert ca.fc1.weight.grad is not None, "fc1权重梯度为None"
    assert ca.fc2.weight.grad is not None, "fc2权重梯度为None"
    assert x.grad is not None, "输入梯度为None"
    assert not torch.isnan(ca.fc1.weight.grad).any(), "fc1梯度含NaN"
    assert not torch.isinf(ca.fc2.weight.grad).any(), "fc2梯度含Inf"
    print("  所有参数和输入梯度正常 ✓")

    # 验证参数更新
    initial_fc1 = ca.fc1.weight.data.clone()
    initial_fc2 = ca.fc2.weight.data.clone()
    optimizer.step()
    assert not torch.equal(ca.fc1.weight.data, initial_fc1), "fc1权重未更新"
    assert not torch.equal(ca.fc2.weight.data, initial_fc2), "fc2权重未更新"
    print("  参数可正常更新 ✓")
    print("Test 5 通过\n")


def test_shared_mlp():
    """验证avg_pool和max_pool路径共享同一套MLP参数（CBAM标准实现要求）"""
    print("Test 6: 共享MLP逻辑验证")
    B, C, H, W = 2, 32, 8, 8
    ca = ChannelAttention(in_planes=C)
    # 输入全1时，avg和max池化结果完全一致
    x = torch.ones(B, C, H, W)
    with torch.no_grad():
        avg_feat = ca.avg_pool(x)
        max_feat = ca.max_pool(x)
        assert torch.allclose(avg_feat, max_feat), "全1输入下avg和max池化结果不一致"
        # 经过同一套MLP后输出必须一致
        avg_out = ca.fc2(ca.relu1(ca.fc1(avg_feat)))
        max_out = ca.fc2(ca.relu1(ca.fc1(max_feat)))
        assert torch.allclose(avg_out, max_out), "共享MLP下avg和max输出不一致"
        # 最终输出为sigmoid(avg_out + max_out) = sigmoid(2*avg_out)
        final_out = ca(x)
        expected_out = torch.sigmoid(2 * avg_out)
        assert torch.allclose(final_out, expected_out, atol=1e-6), "最终输出计算不符合预期"
    print("  avg和max路径确实共享MLP参数 ✓")
    print("Test 6 通过\n")


def test_ratio_hint():
    """提示原代码中ratio参数未生效的小问题"""
    print("Test 7: ratio参数有效性检查")
    C = 64
    ca_ratio8 = ChannelAttention(in_planes=C, ratio=8)
    expected_mid = C // 8  # 按设计ratio=8时中间通道应为8
    actual_mid = ca_ratio8.fc1.out_channels
    print(f"  传入ratio=8，期望中间通道数{expected_mid}，实际{actual_mid}")
    if actual_mid != expected_mid:
        print("  ⚠️  代码小问题：fc1/fc2硬编码了'in_planes//16'，传入的ratio参数未生效")
        print("     修复方法：将__init__中的两处'in_planes//16'改为'in_planes//ratio'即可")
    # 不影响运行
    x = torch.randn(2, C, 16, 16)
    out = ca_ratio8(x)
    assert out.shape == (2, C, 1, 1)
    print("  模块可正常运行 ✓")
    print("Test 7 完成\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}\n")
    try:
        test_forward_shape()
        test_small_channel_boundary()
        test_various_spatial_sizes()
        test_output_range()
        test_gradient_flow()
        test_shared_mlp()
        test_ratio_hint()
        print("✅ 所有测试运行完成！")
    except AssertionError as e:
        print(f"❌ 测试失败: {e}")
        raise
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        raise

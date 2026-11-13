 

import sys
import traceback

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 被测试模块定义（与你给出的实现保持一致）
# ============================================================
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        # 通道混洗实现
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(batch_size, -1, height, width)


class LEFF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        expanded_channels = in_channels * 4

        # 1x1卷积扩展通道
        self.conv1x1_expand = nn.Conv2d(in_channels, expanded_channels, kernel_size=1)

        # 深度卷积序列
        self.dw_conv3 = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,
                                  padding=1, groups=expanded_channels)
        self.dw_conv5 = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=5,
                                  padding=2, groups=expanded_channels)
        self.dw_conv3_2 = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,
                                    padding=1, groups=expanded_channels)

        # 通道混洗
        self.shuffle1 = ChannelShuffle(groups=expanded_channels // 2)
        self.shuffle2 = ChannelShuffle(groups=expanded_channels // 2)

        # 最后的1x1卷积恢复通道
        self.conv1x1_reduce = nn.Conv2d(expanded_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Step 1: 1x1卷积扩展通道
        x_1 = self.conv1x1_expand(x)          # [b, 4c, h, w]
        # Step 2: 3x3深度卷积
        x_2 = self.dw_conv3(x_1)              # [b, 4c, h, w]
        # Step 3: 通道混洗
        x2_s = self.shuffle1(x_2)             # [b, 4c, h, w]
        # Step 4: 5x5深度卷积
        x_3 = self.dw_conv5(x2_s)             # [b, 4c, h, w]
        # Step 5: 残差连接
        x_4 = x_3 + x_2                       # [b, 4c, h, w]
        # Step 6: 通道混洗
        x4_s = self.shuffle2(x_4)             # [b, 4c, h, w]
        # Step 7: 3x3深度卷积
        x_5 = self.dw_conv3_2(x4_s)           # [b, 4c, h, w]
        # Step 8: 残差连接
        x_6 = x_5 + x_2                       # [b, 4c, h, w]
        # Step 9: 恢复通道
        output = self.conv1x1_reduce(x_6)     # [b, c, h, w]
        return output


# ============================================================
# 测试工具
# ============================================================
PASSED = []
FAILED = []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print("[PASS] {}".format(name))
    except Exception as e:  # noqa: BLE001
        FAILED.append((name, e))
        print("[FAIL] {} -> {}".format(name, e))
        traceback.print_exc()


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 一、ChannelShuffle 测试
# ============================================================
def test_shuffle_shape_preserved():
    m = ChannelShuffle(groups=4)
    x = torch.randn(2, 8, 5, 5)
    y = m(x)
    assert y.shape == x.shape, "形状改变: {} -> {}".format(tuple(x.shape), tuple(y.shape))
    assert torch.isfinite(y).all(), "出现 NaN/Inf"


def test_shuffle_is_real_permutation():
    """验证混洗确实按 (g, c_g) 转置重排，而不是恒等映射"""
    groups, c_per_g, B, H, W = 4, 3, 1, 1, 1
    channels = groups * c_per_g
    x = torch.arange(B * channels).reshape(B, channels, H, W).float()
    y = ChannelShuffle(groups=groups)(x)

    # 期望：out[i] = in[(i % groups) * c_per_g + (i // groups)]
    expected = torch.tensor(
        [x[0, (i % groups) * c_per_g + (i // groups), 0, 0].item() for i in range(channels)]
    )
    got = y[0, :, 0, 0]
    assert torch.equal(got, expected), "混洗索引与预期不符:\n got={}\n exp={}".format(got, expected)


def test_shuffle_groups_one_identity():
    m = ChannelShuffle(groups=1)
    x = torch.randn(3, 6, 4, 4)
    assert torch.equal(m(x), x), "groups=1 应为恒等映射"


def test_shuffle_illegal_groups_raises():
    m = ChannelShuffle(groups=5)  # 8 不能被 5 整除
    x = torch.randn(1, 8, 2, 2)
    ok = False
    try:
        m(x)
    except RuntimeError:
        ok = True
    assert ok, "通道数不能被 groups 整除时应抛出 RuntimeError"


# ============================================================
# 二、LEFF 测试
# ============================================================
def test_leff_shape_preserved():
    cases = [
        (2, 64, 32, 32),
        (1, 3, 224, 224),
        (4, 16, 7, 7),
        (2, 32, 28, 14),   # 非对称空间尺寸
        (1, 8, 1, 1),      # 极端 1x1
    ]
    for B, C, H, W in cases:
        m = LEFF(C).to(device()).eval()
        x = torch.randn(B, C, H, W).to(device())
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape, \
            "in_channels={} 形状不匹配: {} -> {}".format(C, tuple(x.shape), tuple(y.shape))
        assert torch.isfinite(y).all(), "in_channels={} 输出出现 NaN/Inf".format(C)


def test_leff_depthwise_config():
    C = 16
    m = LEFF(C)
    exp = C * 4
    for name in ["dw_conv3", "dw_conv5", "dw_conv3_2"]:
        conv = getattr(m, name)
        assert conv.groups == exp, "{} groups 应为 {}, 实际 {}".format(name, exp, conv.groups)
        assert conv.in_channels == exp and conv.out_channels == exp, "{} 通道配置错误".format(name)
        assert conv.in_channels % conv.groups == 0, "{} groups 不整除通道数".format(name)
    # 空间尺寸保持：3x3 pad=1、5x5 pad=2
    assert m.dw_conv3.padding == (1, 1), "dw_conv3 padding 应为 1"
    assert m.dw_conv5.padding == (2, 2), "dw_conv5 padding 应为 2"
    assert m.dw_conv3_2.padding == (1, 1), "dw_conv3_2 padding 应为 1"
    # 通道扩展/还原
    assert m.conv1x1_expand.out_channels == exp, "扩展通道数应为 4c"
    assert m.conv1x1_reduce.in_channels == exp and m.conv1x1_reduce.out_channels == C, "还原通道数应为 c"
    # 混洗分组数
    assert m.shuffle1.groups == exp // 2 and m.shuffle2.groups == exp // 2, "混洗分组数应为 4c//2"


def test_leff_residual_structure():
    """用 hook 还原中间张量，验证两处残差均跳接到 x_2（第一个深度卷积输出）"""
    torch.manual_seed(0)
    C = 8
    m = LEFF(C).eval()
    x = torch.randn(2, C, 6, 6)

    rec = {}

    def save_out(tag):
        def hook(_mod, _inp, out):
            rec[tag] = out.detach().clone()
        return hook

    def save_in(tag):
        def hook(_mod, inp):
            rec[tag] = inp[0].detach().clone()
        return hook

    hs = [
        m.dw_conv3.register_forward_hook(save_out("x2")),
        m.dw_conv5.register_forward_hook(save_out("x3")),
        m.shuffle2.register_forward_pre_hook(save_in("shuffle2_in")),
        m.conv1x1_reduce.register_forward_pre_hook(save_in("reduce_in")),
    ]
    with torch.no_grad():
        out = m(x)
    for h in hs:
        h.remove()

    assert "x2" in rec and "x3" in rec, "hook 未取到中间张量"
    x_2, x_3 = rec["x2"], rec["x3"]
    # 残差 1: shuffle2 的输入应等于 x_3 + x_2
    assert torch.allclose(rec["shuffle2_in"], x_3 + x_2, atol=1e-5), \
        "第一处残差结构不符合 x_4 = x_3 + x_2"
    # 残差 2: 最后一个 1x1 卷积的输入应等于 shuffle2(x_4) + x_2
    x_4 = x_3 + x_2
    x4_s = m.shuffle2(x_4)
    x_5 = m.dw_conv3_2(x4_s)
    assert torch.allclose(rec["reduce_in"], x_5 + x_2, atol=1e-5), \
        "第二处残差跳接的不是 x_2（若设计为 x_5 + x_4 则此处会失败）"
    assert torch.allclose(out, m.conv1x1_reduce(x_5 + x_2), atol=1e-5), "前向整体输出与手工复现不一致"


def test_leff_numerical_robustness():
    C = 4
    m = LEFF(C).eval()
    for name, x in [
        ("all_zero", torch.zeros(1, C, 8, 8)),
        ("all_one", torch.ones(1, C, 8, 8)),
        ("large", torch.full((1, C, 8, 8), 100.0)),
        ("small", torch.full((1, C, 8, 8), -100.0)),
    ]:
        with torch.no_grad():
            y = m(x)
        assert torch.isfinite(y).all(), "{} 输入下输出出现 NaN/Inf".format(name)


def test_leff_grad_flow():
    C = 8
    m = LEFF(C).to(device())
    x = torch.randn(2, C, 8, 8).to(device(), dtype=torch.float32)
    x.requires_grad_(True)
    y = m(x)
    loss = y.pow(2).mean()
    loss.backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, "参数 {} 无梯度".format(name)
        assert torch.isfinite(p.grad).all(), "参数 {} 梯度含 NaN/Inf".format(name)
        assert p.grad.abs().sum() > 0, "参数 {} 梯度全为 0".format(name)
    assert x.grad is not None and torch.isfinite(x.grad).all(), "输入梯度异常"

    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = optim.SGD(m.parameters(), lr=0.1)
    opt.step()
    updated = sum(1 for n, p in m.named_parameters() if not torch.equal(p.detach(), before[n]))
    assert updated == len(before), "仅有 {}/{} 个参数被更新".format(updated, len(before))


def test_leff_batch1_and_modes():
    C = 8
    m = LEFF(C)
    x = torch.randn(1, C, 8, 8)
    m.train()
    y_train = m(x)
    m.eval()
    with torch.no_grad():
        y_eval = m(x)
    assert y_train.shape == x.shape
    # 无 BN/Dropout，两种模式输出应完全一致
    assert torch.allclose(y_train, y_eval, atol=1e-6), "train/eval 输出不一致（模块含随机性或模式相关层）"


# ============================================================
# 主入口
# ============================================================
def main():
    dev = device()
    print("Running LEFF tests on device: {}\n".format(dev))

    print("== ChannelShuffle ==")
    check("shuffle: 输出形状与输入一致", test_shuffle_shape_preserved)
    check("shuffle: 确为通道重排（索引正确）", test_shuffle_is_real_permutation)
    check("shuffle: groups=1 时为恒等映射", test_shuffle_groups_one_identity)
    check("shuffle: 通道不能整除 groups 时报错", test_shuffle_illegal_groups_raises)

    print("\n== LEFF ==")
    check("leff: 输出形状与输入一致", test_leff_shape_preserved)
    check("leff: 深度卷积/1x1 卷积配置正确", test_leff_depthwise_config)
    check("leff: 两处残差均跳接 x_2", test_leff_residual_structure)
    check("leff: 极端输入数值鲁棒", test_leff_numerical_robustness)
    check("leff: 梯度流与参数更新", test_leff_grad_flow)
    check("leff: batch=1 与 train/eval 一致", test_leff_batch1_and_modes)

    print("\n" + "=" * 50)
    print("通过 {}/{}".format(len(PASSED), len(PASSED) + len(FAILED)))
    if FAILED:
        print("失败用例：")
        for name, e in FAILED:
            print("  - {}: {}".format(name, e))
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()

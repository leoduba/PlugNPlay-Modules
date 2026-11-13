import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import torch
import torch.nn as nn
 

import os
class DeformableConv2d(nn.Module):
    # 带mask调制,即每个核的点都被0-1调制
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, dilation=1, groups=1, bias=True, *,
                 offset_groups=1, with_mask=False
                 ):
        super().__init__()
        assert in_dim % groups == 0, "输入通道数必须能被groups整除"
        assert out_dim % groups == 0, "输出通道数必须能被groups整除"
        self.stride = stride
        self.padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        self.param_generator = nn.Conv2d(in_dim, (3 if with_mask else 2) * offset_groups * kernel_size ** 2, 3, 1, 1)
        nn.init.constant_(self.param_generator.weight, 0.0)  # 关键初始化
        nn.init.constant_(self.param_generator.bias, 0.0)  # 初始无偏移

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None

        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x 


#测试
import math
import torch
import torch.nn as nn
import torch.optim as optim

 

# 仅 stride=1 时，类内部 padding 公式保证 "same"，offset 空间尺寸与输出尺寸一致
WORKING_CONFIGS = [(1, 1, 3), (1, 2, 3), (1, 3, 3), (1, 1, 5), (1, 2, 5)]


def out_size(size, padding, dilation, kernel, stride):
    """标准卷积输出尺寸：floor((H + 2p - d*(k-1) - 1) / s) + 1"""
    return math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1


def _equiv_conv(deform):
    """用 deform 模块的权重/偏置构造等价的标准卷积（groups=1，padding 取模块内部值）。"""
    out_dim, in_per_group, k, _ = deform.weight.shape
    conv = nn.Conv2d(
        in_per_group, out_dim, k,
        stride=deform.stride, padding=deform.padding,
        dilation=deform.dilation, bias=deform.bias is not None,
    )
    with torch.no_grad():
        conv.weight.copy_(deform.weight)
        if deform.bias is not None:
            conv.bias.copy_(deform.bias)
    return conv


def test_output_shape_various_configs():
    """(k, dilation) 组合下，stride=1 时输出尺寸必须与公式一致且数值有限。"""
    print("[1] output shape over (k, dilation), stride=1 ...")
    torch.manual_seed(0)
    B, Cin, H, W, Cout = 2, 8, 20, 20, 4
    for stride, dil, k in WORKING_CONFIGS:
        m = DeformableConv2d(Cin, Cout, k, stride=stride, dilation=dil)
        # 内部 padding 必须等于约定的公式
        assert m.padding == ((stride - 1) + dil * (k - 1)) // 2
        x = torch.randn(B, Cin, H, W)
        y = m(x)
        eh = out_size(H, m.padding, dil, k, stride)
        ew = out_size(W, m.padding, dil, k, stride)
        assert y.shape == (B, Cout, eh, ew), (k, stride, dil, y.shape, (eh, ew))
        assert torch.isfinite(y).all()
    print("    OK")


def test_offset_zero_equals_standard_conv():
    """offset 为 0 时，DeformableConv2d 必须严格退化为等价的 nn.Conv2d（含内部 padding）。"""
    print("[2] offset=0 == standard conv ...")
    torch.manual_seed(0)
    B, Cin, H, W, Cout = 2, 6, 14, 14, 5
    for stride, dil, k in WORKING_CONFIGS:
        m = DeformableConv2d(Cin, Cout, k, stride=stride, dilation=dil, with_mask=False)
        m.eval()
        conv = _equiv_conv(m)
        x = torch.randn(B, Cin, H, W)
        with torch.no_grad():
            y_def = m(x)
            y_std = conv(x)
        assert y_def.shape == y_std.shape, (stride, dil, k)
        assert torch.allclose(y_def, y_std, atol=1e-4), (stride, dil, k,
                                                          (y_def - y_std).abs().max().item())
    print("    OK")


def test_mask_modulation_at_init():
    """with_mask=True 且初始 mask=sigmoid(0)=0.5、offset=0 时，输出应为标准卷积的 0.5 倍。"""
    print("[3] mask=0.5 modulation ...")
    torch.manual_seed(0)
    B, Cin, H, W, Cout, k = 2, 6, 14, 14, 5, 3
    m = DeformableConv2d(Cin, Cout, k, with_mask=True)
    m.eval()
    conv = _equiv_conv(m)
    with torch.no_grad():
        nn.init.constant_(conv.bias, 0.0)  # bias 不参与 mask 调制，置 0 便于精确比较
        nn.init.constant_(m.bias, 0.0)
    x = torch.randn(B, Cin, H, W)
    with torch.no_grad():
        y_def = m(x)
        y_std = conv(x)
    assert torch.allclose(y_def, 0.5 * y_std, atol=1e-4), (y_def - 0.5 * y_std).abs().max().item()
    print("    OK")


def test_mask_splitting_shapes():
    """带 mask 时 offset/mask 的通道拆分必须满足底层算子要求（各自为 og*k^2）。"""
    print("[4] offset/mask channel split ...")
    torch.manual_seed(0)
    for og in (1, 2):
        m = DeformableConv2d(4, 4, 3, offset_groups=og, with_mask=True)
        x = torch.randn(1, 4, 10, 10)
        y = m(x)  # 不报错即说明通道数匹配
        assert y.shape == (1, 4, 10, 10)
    print("    OK")


def test_gradient_flow():
    """反向传播：主权重、param_generator、输入都能拿到有效梯度，且能正常更新。"""
    print("[5] gradient flow ...")
    torch.manual_seed(0)
    m = DeformableConv2d(6, 5, 3, with_mask=True)  # stride=1
    x = torch.randn(2, 6, 16, 16, requires_grad=True)
    opt = optim.SGD(m.parameters(), lr=0.1)

    before = m.weight.detach().clone()
    y = m(x)
    loss = y.pow(2).mean()
    opt.zero_grad()
    loss.backward()

    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} grad is None"
        assert torch.isfinite(p.grad).all(), f"{name} grad not finite"
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # offset/mask 生成器确实被训练到
    assert m.param_generator.weight.grad.abs().sum() > 0

    opt.step()
    assert not torch.equal(m.weight.detach(), before)
    print("    OK")


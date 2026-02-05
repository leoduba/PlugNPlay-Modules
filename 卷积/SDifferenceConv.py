 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
import math
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
# 论文：
# 论文地址：
class SDifferenceConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SDifferenceConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)       
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        grad_weight = -self.weight.clone()  # 克隆权重，避免原地修改
        hw = self.weight.size(-1)
        grad_weight[:, :, int((hw-1)/2), int((hw-1)/2)] = torch.sum(self.weight, dim=[2, 3])

        if self.padding_mode != "zeros":
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            grad_weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, grad_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

 
sd_conv = SDifferenceConv(
    in_channels=3,    # 输入通道（RGB图）
    out_channels=16,  # 输出通道
    kernel_size=3,    # 3x3卷积核
    padding=1,        # 填充1，保持特征图尺寸不变
    stride=1          # 步幅1
)
# 生成测试输入：[batch_size, in_channels, H, W]（模拟4张RGB图，256x256）
x = torch.randn(4, 3, 256, 256)
# 前向传播
out = sd_conv(x)
# 打印输入/输出维度
print(f"输入维度：{x.shape}")
print(f"输出维度：{out.shape}")
# 验证维度是否符合预期（padding=1, stride=1 → H/W不变）
assert out.shape == (4, 16, 256, 256), "维度匹配失败！"
print("✅ 维度匹配测试通过\n")
 

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple
# 论文：
# 论文地址：
class DHiF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DHiF, self).__init__()
        self.kernel_s_ope = kernel_size
        self.ope_channels = self.kernel_s_ope*self.kernel_s_ope  # kernel_size*kernel_size
        self.operator_conv = nn.Sequential(nn.Linear(self.ope_channels, self.ope_channels*self.ope_channels), nn.Tanh())

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)       # kernel_size*kernel_size [3,3]
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = Parameter(torch.Tensor(in_channels, out_channels, *self.kernel_size))  # [out_channels:1, in_channels:1, 3, 3]
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
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
        b, c, h, w = input.size()
        h1 = h//self.stride[0]
        w1 = w//self.stride[0]
        x_unfold = F.unfold(input, kernel_size=self.kernel_s_ope, stride=self.stride, padding=self.padding, dilation=self.dilation)
        x_unfold = x_unfold.view(b, c, -1, h1, w1).permute(0,1,3,4,2).unsqueeze(4)  # b, c, kk, h, w -> b, c, h, w, 1, kk
        operator = self.operator_conv(torch.norm(x_unfold, p=2, dim=1)).view(b,1,h1,w1,self.ope_channels,self.ope_channels)  # b, 1, h, w, kk,kk
        x_operator = torch.einsum('bchwij, blhwjk -> bchwik', x_unfold, operator).squeeze(4).permute(0,1,4,2,3).contiguous()  # b, c, h, w, 1, kk -> b, c, kk, h, w
        self.weight_reshape = self.weight.view(self.in_channels, self.out_channels, -1).permute(1,0,2)  # c, out_c, k, k -> out_c, c, kk
        if self.bias is not None:
            output = torch.einsum('bckhw, ock -> bohw', x_operator+x_unfold.squeeze(4).permute(0,1,4,2,3),
                                  self.weight_reshape).contiguous() + self.bias[None, :, None, None]
        else:
            output = torch.einsum('bckhw, ock -> bohw', x_operator+x_unfold.squeeze(4).permute(0,1,4,2,3),
                                  self.weight_reshape).contiguous()

        return output
dhif = DHiF(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    padding=1
)

# 前向传播
x = torch.randn(2, 64, 32, 32)
output = dhif(x)
print(output.shape)  # torch.Size([2, 128, 32, 32])

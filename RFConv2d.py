import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from itertools import repeat
#论文地址：https://github.com/Shanghua-Gao/RFNext/ 
#论文： 
# 补充get_padding实现（兼容无timm环境）
def get_padding(kernel_size, stride=1, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


def value_crop(dilation, min_dilation, max_dilation):
    if min_dilation is not None and dilation < min_dilation:
        dilation = min_dilation
    if max_dilation is not None and dilation > max_dilation:
        dilation = max_dilation
    return dilation


def rf_expand(dilation, expand_rate, num_branches, min_dilation=1, max_dilation=None):
    # 增加类型校验，确保dilation是数值型
    dilation = (int(dilation[0]), int(dilation[1])) if isinstance(dilation, (list, tuple, torch.Tensor)) else dilation
    assert num_branches >= 2, "number of branches must >=2"
    
    delta_dilation0 = expand_rate * dilation[0]
    delta_dilation1 = expand_rate * dilation[1]
    rate_list = []
    
    for i in range(num_branches):
        d0 = value_crop(
            int(round(dilation[0] - delta_dilation0 + i * 2 * delta_dilation0/(num_branches-1))),
            min_dilation, max_dilation
        )
        d1 = value_crop(
            int(round(dilation[1] - delta_dilation1 + i * 2 * delta_dilation1/(num_branches-1))),
            min_dilation, max_dilation
        )
        rate_list.append((d0, d1))

    # 去重并保持原顺序
    unique_rate_list = list(dict.fromkeys(rate_list))  # 比set更稳定的去重方式
    return unique_rate_list


class RFConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 num_branches=3,
                 expand_rate=0.5,
                 min_dilation=1,
                 max_dilation=None,
                 init_weight=0.01,
                 search_interval=1250,
                 max_search_step=0,
                 rf_mode='rfsearch',
                 pretrained=None):
        
        # 预处理pretrained参数（仅保留权重合并必要的键校验）
        pretrained = pretrained or {}
        merge_required_keys = ['rates', 'num_rates', 'sample_weights', 'weight']
        if pretrained and rf_mode == 'rfmerge':
            for k in merge_required_keys:
                assert k in pretrained, f"rfmerge mode missing key in pretrained: {k}"
            
            rates = pretrained['rates']
            num_rates = pretrained['num_rates']
            sample_weights = pretrained['sample_weights']
            sample_weights = self.normalize(sample_weights[:num_rates.item()])
            max_dilation_rate = rates[num_rates.item() - 1]
            
            # 统一kernel_size/stride为列表
            kernel_size = _pair(kernel_size)
            stride = _pair(stride)
            
            # 计算新卷积核尺寸
            new_kernel_size = (
                kernel_size[0] + (max_dilation_rate[0].item() - 1) * (kernel_size[0] // 2) * 2,
                kernel_size[1] + (max_dilation_rate[1].item() - 1) * (kernel_size[1] // 2) * 2
            )
            new_dilation = (1, 1)
            new_padding = (
                get_padding(new_kernel_size[0], stride[0], new_dilation[0]),
                get_padding(new_kernel_size[1], stride[1], new_dilation[1])
            )

            # 合并多分支权重
            old_weight = pretrained['weight']
            new_weight = torch.zeros(
                size=(old_weight.shape[0], old_weight.shape[1], new_kernel_size[0], new_kernel_size[1]),
                dtype=old_weight.dtype,
                device=old_weight.device
            )
            
            for r, rate in enumerate(rates[:num_rates.item()]):
                rate = (rate[0].item(), rate[1].item())
                for i in range(-(kernel_size[0]//2), kernel_size[0]//2 + 1):
                    for j in range(-(kernel_size[1]//2), kernel_size[1]//2 + 1):
                        new_weight[:, :,
                                   new_kernel_size[0]//2 - i*rate[0],
                                   new_kernel_size[1]//2 - j*rate[1]] += \
                            old_weight[:, :, kernel_size[0]//2 - i, kernel_size[1]//2 - j] * sample_weights[r]

            # 更新参数
            kernel_size = new_kernel_size
            padding = new_padding
            dilation = new_dilation
            pretrained['rates'][0] = torch.FloatTensor([1, 1])
            pretrained['num_rates'] = torch.IntTensor([1])
            pretrained['weight'] = new_weight
            pretrained['sample_weights'] = pretrained['sample_weights'] * 0.0 + init_weight

        # 初始化父类
        super(RFConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode
        )

        # 初始化RF相关参数
        self.rf_mode = rf_mode
        self.pretrained = pretrained
        self.num_branches = max(2, num_branches)
        self.max_dilation = max_dilation
        self.min_dilation = min_dilation
        self.expand_rate = expand_rate
        self.init_weight = init_weight
        self.search_interval = search_interval
        self.max_search_step = max_search_step

        # 可学习的分支权重
        self.sample_weights = nn.Parameter(torch.Tensor(self.num_branches))
        
        # 注册缓冲区（使用默认值初始化，不依赖pretrained）
        self.register_buffer('counter', torch.zeros(1, dtype=torch.int32))
        self.register_buffer('current_search_step', torch.zeros(1, dtype=torch.int32))
        self.register_buffer('rates', torch.ones(self.num_branches, 2, dtype=torch.int32))
        self.register_buffer('num_rates', torch.ones(1, dtype=torch.int32))

        # 初始化膨胀率和分支权重
        self.rates[0] = torch.tensor([self.dilation[0], self.dilation[1]], dtype=torch.int32)
        self.sample_weights.data.fill_(self.init_weight)

        # 加载预训练参数（仅加载存在的键，跳过缺失的缓冲区）
        if pretrained:
            # 过滤掉pretrained中不存在的键，避免加载错误
            pretrained_filtered = {k: v for k, v in pretrained.items() if k in self.state_dict()}
            self.load_state_dict(pretrained_filtered, strict=False)

        # 不同RF模式的初始化
        if self.rf_mode == 'rfsearch':
            self.estimate()
            self.expand()
        elif self.rf_mode == 'rfsingle':
            self.estimate()
            self.max_search_step = 0
            self.sample_weights.requires_grad = False
        elif self.rf_mode == 'rfmultiple':
            self.estimate()
            self.expand()
            self.sample_weights.data.fill_(self.init_weight)
            self.max_search_step = 0
        elif self.rf_mode == 'rfmerge':
            self.max_search_step = 0
            self.sample_weights.requires_grad = False
        else:
            raise NotImplementedError(f"rf_mode {self.rf_mode} not supported")

        # 单分支模式校验
        if self.rf_mode in ['rfsingle', 'rfmerge']:
            assert self.num_rates.item() == 1, "Single branch mode requires num_rates=1"

    # 修复拼写错误
    def normalize(self, w):
        abs_w = torch.abs(w)
        norm_w = abs_w / torch.sum(abs_w)
        return norm_w

    def _conv_forward_dilation(self, input, dilation_rate):
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                self.weight, self.bias, self.stride,
                _pair(0), dilation_rate, self.groups
            )
        else:
            padding = (
                dilation_rate[0] * (self.kernel_size[0] - 1) // 2,
                dilation_rate[1] * (self.kernel_size[1] - 1) // 2
            )
            return F.conv2d(
                input, self.weight, self.bias, self.stride,
                padding, dilation_rate, self.groups
            )

    def forward(self, x):
        if self.num_rates.item() == 1:
            return super().forward(x)
        else:
            # 归一化分支权重
            norm_w = self.normalize(self.sample_weights[:self.num_rates.item()])
            # 多分支卷积求和
            x_out = 0.0
            for i in range(self.num_rates.item()):
                dilation_rate = (self.rates[i][0].item(), self.rates[i][1].item())
                x_out += self._conv_forward_dilation(x, dilation_rate) * norm_w[i]
            
            # 训练时更新感受野
            if self.training and self.max_search_step > 0:
                self.searcher()
            return x_out

    def searcher(self):
        self.counter += 1
        if (self.counter % self.search_interval == 0 and 
            self.current_search_step < self.max_search_step):
            self.counter.zero_()
            self.current_search_step += 1
            self.estimate()
            self.expand()

    def estimate(self):
        # 估算最优膨胀率
        norm_w = self.normalize(self.sample_weights[:self.num_rates.item()])
        sum0, sum1, w_sum = 0, 0, 0
        
        for i in range(self.num_rates.item()):
            sum0 += norm_w[i].item() * self.rates[i][0].item()
            sum1 += norm_w[i].item() * self.rates[i][1].item()
            w_sum += norm_w[i].item()
        
        # 计算加权平均膨胀率
        estimated_d0 = value_crop(int(round(sum0 / w_sum)), self.min_dilation, self.max_dilation)
        estimated_d1 = value_crop(int(round(sum1 / w_sum)), self.min_dilation, self.max_dilation)
        self.dilation = (estimated_d0, estimated_d1)
        
        # 更新padding
        self.padding = (
            get_padding(self.kernel_size[0], self.stride[0], self.dilation[0]),
            get_padding(self.kernel_size[1], self.stride[1], self.dilation[1])
        )
        
        # 更新rates缓冲区
        self.rates[0] = torch.tensor([self.dilation[0], self.dilation[1]], dtype=torch.int32)
        self.num_rates[0] = 1
        
        # 打印日志
        print(f"Estimated dilation: {self.dilation}")

    def expand(self):
        # 扩展为多分支膨胀率
        rates = rf_expand(
            self.dilation, self.expand_rate, self.num_branches,
            self.min_dilation, self.max_dilation
        )
        # 更新rates缓冲区
        for i, rate in enumerate(rates):
            if i < self.rates.shape[0]:
                self.rates[i] = torch.tensor(rate, dtype=torch.int32)
        self.num_rates[0] = len(rates)
        # 重置分支权重
        self.sample_weights.data.fill_(self.init_weight)
        
        # 打印日志
        print(f"Expanded dilations: {self.rates[:len(rates)].cpu().tolist()}")

# ------------------------------
# 测试用例（最终版）
# ------------------------------
def test_rfconv_basic():
    """测试基础功能（rfsearch模式）"""
    print("="*50)
    print("测试基础RFConv2d（rfsearch模式）")
    # 初始化卷积层
    conv = RFConv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        dilation=(1, 1),
        num_branches=3,
        expand_rate=0.5,
        max_search_step=2,  # 允许2次搜索
        rf_mode='rfsearch'
    )
    
    # 模拟输入（B=2, C=3, H=32, W=32）
    x = torch.randn(2, 3, 32, 32)
    
    # 训练模式前向传播（触发searcher）
    conv.train()
    out1 = conv(x)
    # 模拟多次迭代触发感受野更新
    conv.counter = torch.tensor([1249], dtype=torch.int32)  # 接近search_interval
    out2 = conv(x)
    
    # 验证输出形状
    assert out1.shape == (2, 16, 32, 32), f"输出形状错误，预期(2,16,32,32)，实际{out1.shape}"
    assert out2.shape == (2, 16, 32, 32), f"更新感受野后输出形状错误"
    print("基础RFConv2d测试通过！")
    print("="*50)

def test_rfconv_single_branch():
    """测试单分支模式（rfsingle）"""
    print("测试单分支RFConv2d（rfsingle模式）")
    conv = RFConv2d(
        in_channels=16,
        out_channels=32,
        kernel_size=3,
        dilation=(2, 2),
        rf_mode='rfsingle'
    )
    
    x = torch.randn(1, 16, 64, 64)
    out = conv(x)
    
    # 验证单分支约束
    assert conv.num_rates.item() == 1, "单分支模式num_rates应为1"
    assert out.shape == (1, 32, 64, 64), f"单分支输出形状错误，预期(1,32,64,64)，实际{out.shape}"
    print("单分支RFConv2d测试通过！")
    print("="*50)

def test_rfconv_multiple_branch():
    """测试多分支模式（rfmultiple）"""
    print("测试多分支RFConv2d（rfmultiple模式）")
    conv = RFConv2d(
        in_channels=32,
        out_channels=64,
        kernel_size=3,
        dilation=(1, 1),
        num_branches=4,  # 4个分支
        rf_mode='rfmultiple'
    )
    
    x = torch.randn(2, 32, 16, 16)
    out = conv(x)
    
    # 验证多分支数量
    assert conv.num_rates.item() >= 2, "多分支模式num_rates应>=2"
    assert out.shape == (2, 64, 16, 16), f"多分支输出形状错误，预期(2,64,16,16)，实际{out.shape}"
    print("多分支RFConv2d测试通过！")
    print("="*50)

def test_rfconv_rfexpand():
    """测试膨胀率扩展函数"""
    print("测试rf_expand函数")
    # 测试基础扩展
    dilation = (2, 2)
    rates = rf_expand(dilation, 0.5, 3)
    assert len(rates) == 3, f"扩展分支数错误，预期3，实际{len(rates)}"
    assert all(isinstance(r, tuple) for r in rates), "膨胀率应为元组类型"
    
    # 测试边界约束
    rates = rf_expand((1, 1), 0.5, 3, max_dilation=2)
    assert max([r[0] for r in rates]) <= 2, "膨胀率超出max_dilation约束"
    print("rf_expand函数测试通过！")
    print("="*50)

def test_rfconv_pretrained_merge():
    """测试预训练权重合并（rfmerge模式）"""
    print("测试RFConv2d权重合并（rfmerge模式）")
    # 模拟预训练参数（仅包含权重合并必要的键）
    pretrained = {
        'rates': torch.tensor([[1,1], [2,2], [3,3]], dtype=torch.int32),
        'num_rates': torch.tensor([3], dtype=torch.int32),
        'sample_weights': torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32),
        'weight': torch.randn(16, 3, 3, 3),  # (out, in, kH, kW)
    }
    
    # 初始化merge模式卷积（显式设置bias=False）
    conv = RFConv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        bias=False,  # 匹配pretrained无bias的场景
        rf_mode='rfmerge',
        pretrained=pretrained
    )
    
    x = torch.randn(1, 3, 32, 32)
    out = conv(x)
    
    # 验证merge后参数
    assert conv.dilation == (1, 1), "merge模式dilation应为(1,1)"
    assert conv.num_rates.item() == 1, "merge模式num_rates应为1"
    assert out.shape == (1, 16, 32, 32), f"merge模式输出形状错误，预期(1,16,32,32)，实际{out.shape}"
    print("RFConv2d权重合并测试通过！")
    print("="*50)

# 运行所有测试
if __name__ == "__main__":
    test_rfconv_basic()
    test_rfconv_single_branch()
    test_rfconv_multiple_branch()
    test_rfconv_rfexpand()
    test_rfconv_pretrained_merge()
    print("所有测试用例执行成功！")

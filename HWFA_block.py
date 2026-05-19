import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p
class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class HWFA(nn.Module):
    def __init__(self, in_ch, out_ch ,J_n):
        super(HWFA, self).__init__()
        self.J_n = J_n
        self.wt = DWTForward(J=J_n, mode='zero', wave='haar')
        self.conv2 = nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1) 
        self.bn    = nn.BatchNorm2d(out_ch)
        self.relu  =  nn.ReLU(inplace=True)
        self.conv1 = Conv(2*out_ch,out_ch) 
        self.out_ch = out_ch
        
        
    def forward(self, x ,y):
        yL, yH = self.wt(x)

        y_HL = yH[self.J_n-1][:,:,0,::]
        y_LH = yH[self.J_n-1][:,:,1,::]
        y_HH = yH[self.J_n-1][:,:,2,::]
        
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)        
        x = self.relu(self.bn(self.conv2(x)))
        #mid = torch.cat([x,y], dim=1)
        #out = self.conv1(mid)
        mid1 = x[:,:self.out_ch//2,:,:]
        mid2 = y[:,:self.out_ch//2,:,:]
        out = torch.cat([mid2,mid1],dim=1)

        return out

if __name__ == '__main__':
    x = torch.randn(1, 3, 640, 640) # 输入特征
    y = torch.randn(1, 512, 80, 80) # 被融合的特征
    model = HWFA(3, 512, 3)         # 3:输入特征通道数，512:输出特征通道数，3:小波分解层数（1次/2）
    y = model(x,y)
    print(y.shape)


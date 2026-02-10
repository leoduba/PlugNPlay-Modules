class DMKDC(nn.Module):
     
  def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
      super().__init__()
      # 1. 定义3种深度卷积核（核心：分组数=输入通道数）
      self.dwconv = nn.ModuleList([
          # 卷积核1：方形核（3x3）- 提取局部方形区域特征
          nn.Conv2d(in_channels, in_channels, square_kernel_size, 
                    padding=square_kernel_size // 2, groups=in_channels),
          # 卷积核2：水平条带核（1x11）- 提取水平方向长距离特征（如横线、行结构）
          nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), 
                    padding=(0, band_kernel_size // 2), groups=in_channels),
          # 卷积核3：垂直条带核（11x1）- 提取垂直方向长距离特征（如竖线、列结构）
          nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), 
                    padding=(band_kernel_size // 2, 0), groups=in_channels)
      ])
  
      # 2. 批归一化+激活函数（特征后处理）
      self.bn = nn.BatchNorm2d(in_channels)
      self.act = nn.SiLU()
  
      # 3. 动态权重生成器（核心：自适应池化+1x1卷积）
      self.dkw = nn.Sequential(
          nn.AdaptiveAvgPool2d(1),  # 全局平均池化：(B,C,H,W) → (B,C,1,1)，提取全局特征
          nn.Conv2d(in_channels, in_channels * 3, 1)  # 1x1卷积：生成3组权重（对应3个卷积核）
      )
  def forward(self, x):
      # 步骤1：生成动态权重
      x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
      x_dkw = F.softmax(x_dkw, dim=0)  # 在第0维（3个卷积核）做softmax，权重和为1
      
      # 步骤2：3个卷积核分别卷积 + 动态权重加权
      x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(len(self.dwconv))]).sum(0)
      
      # 步骤3：批归一化+激活，输出最终特征
      return self.act(self.bn(x))

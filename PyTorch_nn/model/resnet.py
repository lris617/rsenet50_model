import torch.nn as nn
import torch

class BasicBlock(nn.Module):  ## （）内为继承nn的模型
    expansion = 1    ### 这个参数在resnet34层中并没有什么用处  这个参数是为了控制在一大部分中的通道数变化的

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
        @param in_channel:  此块输入的通道数
        @param out_channel: 输出的通道数
        @param stride: 在第一个卷积层的步长
        @param downsample: 是否进行下采样
        @param kwargs: 其他参数（可变长参数）
        """
        ### 父类初始化
        super(BasicBlock, self).__init__()
        ### 自定义操作赋值给变量
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False) #定义卷积层
        self.bn1 = nn.BatchNorm2d(out_channel) #定义归一化
        self.relu = nn.ReLU()# 定义激活函数
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False) # 定义卷积层
        self.bn2 = nn.BatchNorm2d(out_channel)# 定义归一化

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False) # 定义卷积层
        self.bn3 = nn.BatchNorm2d(out_channel)# 定义归一化

        self.downsample = downsample   ##定义下采样部分

    def forward(self, x):
    	### 在这里就是构造残差块的基本结构
        identity = x ## 先将最开始的输入 进行赋值到identity  这一部分是为了进行恒等映射
        if self.downsample is not None:  
        # 如果downsample 不是空值（下采样）的话  
        #就在后方进行下采样层相应的操作，因为在上述分析模块部分已经说到，
        #在虚线部分，会因为通道数不一致，要进行下采样操作，使得通道数一致。
            identity = self.downsample(x)
        ### 两部分 卷积批标准化 卷积批标准化
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)  ## 过一层激活层
        
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity   ## 将此基本结构块的输入与本结构块的最后一层的输出进行叠加，
        #形成最终的输出

        out = self.relu(out) # 过激活函数

        return out  ## 返回此结构块的输出 也就是下一个残差块的基本结构的输入了


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    """

    expansion = 4  ## 这个就是通道数变化的系数 

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
		## resnext50_32x4d 和resnext101_32x8d 会使用
        width = int(out_channel * (width_per_group / 64.)) * groups
        
		### 定义三个卷积过程
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        ##下采样
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
            
		## 基本块为三次卷积层的过程 并非跟resnet34一致
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out ## 返回输出



class ResNet(nn.Module):  ##继承自nn.Module函数

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        """
        @param block:  传入实例化BasicBlock，就是上一部分代码的基本块
        @param blocks_num:  块的个数此为一个列表 列表长度为 resNet几大块  
        对应列表中的每一个数——> 即为每一部分的基本块的块数,
        例如resnet34 中的blocks_num = [3, 4, 6, 3]   可以看上述图中的resnet34的结构 本人画出的四大块
        
        @param num_classes: 几分类
        @param include_top: 判定条件是否采用适应性平均池化
        @param groups:
        @param width_per_group:
        """
        super(ResNet, self).__init__()	
        ## 进行赋值
        self.include_top = include_top
        self.in_channel = 64   ### 输入的通道数

        self.groups = groups
        self.width_per_group = width_per_group
        ### 最开始的一大层     先进行卷积核为7*7 卷积操作
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
		 ## 然后进行归一化
        self.bn1 = nn.BatchNorm2d(self.in_channel)
         ## 然后过激活函数  增加非线性表达
        self.relu = nn.ReLU(inplace=True)
         ## 然后经过最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #### 进行ResNet的四大部分[3, 4, 6, 3]
        # 构造每一部分函数：_make_layer ：为本类的成员函数  在下方
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        
        ### 一个判定条件  为True则是会有自适应平均池化
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)   ### 自适应平均池化  自适应去需要进行均值的数值

            self.fc = nn.Linear(512 * block.expansion, num_classes)   ## 全连接层

        ### 遍历每一个模块进行模块的权重的初始化
        for m in self.modules():## 遍历所有的层
            if isinstance(m, nn.Conv2d):   ## m 是否为卷积的实例化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				### 以何凯明大佬命名的初始化的函数

    ### 此函数为构造每一大部分的函数
    def _make_layer(self, block, channel, block_num, stride=1):
        """
        @param block: 基本块   resnet34层传入的是BasicBlock,更多层数的会传入瓶颈块 Bottleneck
        @param channel: 每一大部分的通道数
        @param block_num: 此部分的基本块的数量
        @param stride:  步长
        @return:
        """
        downsample = None
        # 步长不为1   因为需要downsample进行对于不同残差块之间的统一   上一个残差块的输出与本残差块的输出宽高保持一致
        ### 如果步长不为1 也就是每个大部分的第一个基本块的第一个卷积层，
        ### 或者 输入通道与通道不匹配，也就是resnet 更多层会出现 一大部分中通道数发生变化
        ### 会出现通道数不匹配 所以要进行下采样，统一通道数的
        if stride != 1 or self.in_channel != channel * block.expansion:
            ###block.expansion = 1
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),  ## 进行卷积保持宽高一致
                nn.BatchNorm2d(channel * block.expansion)  ## 批归一化
            )
        #定义一个层列表
        layers = []
        ## 添加基本块  先添加一个基本块  之后的用循环，因为第一个基本块，可能会出现下采样的情况
        layers.append(
            block(self.in_channel, channel, downsample=downsample, stride=stride, groups=self.groups, width_per_group=self.width_per_group)
        )
        # 注意注意    只有在不同大部分的情况下才进行此操作  因为上一大部分跟下一大部分的channel通道不一样
        self.in_channel = channel * block.expansion    ## 下一层的输入等于本层的输出
        ### 进行循环 通过小的基本块 构造一个大部分
        for _ in range(1, block_num):
            layers.append(
                block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group)
            )
		## 返回本大部分
        return nn.Sequential(*layers)

    def forward(self, x):
        #开始一部分
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ## 中间四大部分
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:## 判定
            x = self.avgpool(x)   ### 自适应平均池化
            x = torch.flatten(x, 1)  ## 拉直层
            x = self.fc(x) # 全连接层

        return x  ##返回输出

## 对于不同层的resnet网络 ，所传入的参数设定
def resnet34(num_classes=1, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1, include_top=True, pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    if pretrained:
        try:
            import torchvision
            # 加载 torchvision 的 ImageNet 预训练权重
            tv_model = torchvision.models.resnet50(pretrained=True)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in tv_model.state_dict().items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print("Loaded ImageNet pretrained weights for custom ResNet50.")
        except Exception as e:
            print(f"Failed to load ImageNet pretrained weights: {e}")
    return model

def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

def resnext50_32x4d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

def resnext101_32x8d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['FPN']

class Bottleneck(nn.Module):
    expansion = 4
    #这里，我们先实现的是Bottleneck的左半边(1*1->3*3->1*1)，expansion = 最后一层的输出通道数/第一层的输入通道数
    def __init__(self,in_planes,planes,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    #这里加上了residual
    def forward(self,x):
        residual = x            #右边的short_cut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.con3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out += self.relu(out)

        return out

class FPN(nn.Module):
    #这里传入的参数block用于传入Bottleneck
    def __init__(self,block,layers):
        super(FPN, self).__init__()
        self.inplanes = 64

        #这部分是FPN最开始的一个Conv1,为了节省内存，没有将这部分纳入金字塔
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        #Bottom-up layers
        self.layer1 = self._make_layer(block,layers[0])
        self.layer2 = self._make_layer(block,layers[1],stride=2)
        self.layer3 = self._make_layer(block,layers[2],stride=2)
        self.layer4 = self._make_layer(block,layers[3],stride=2)

        #Top layer
        self.toplayer = nn.Conv2d(2048,256,kernel_size=1,stride=1,padding=0)

        #FPN朝右边延伸的，Smooth layers
        self.smooth1 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth2 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.smooth3 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)

        #FPN的左边的特征金字塔连接到右边
        self.latlayer1 = nn.Conv2d(1024,256,kernel_size=1,stride=1,padding=0)
        self.latlayer2 = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)
        self.latlayer3 = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)

        #这部分是初始化conv的值，还有BN的值
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #blocks是Bottleneck的数量
    def _make_layer(self,block,planes,blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes!= block.expansion*planes:
            #这里的downsample用来实现Bottleneck的右边
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,block.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(block.expansion*planes)
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers)

    #通过这个操作来结合左边过来的1*1+top-bottom
    def _upsample_add(self,x,y):
        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear')+y

    def forward(self,x):
        #Bottom-up
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        #top-bottom
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5,self.latlayer1(c4))
        p3 = self._upsample_add(p4,self.latlayer2(c3))
        p2 = self._upsample_add(p3,self.latlayer3(c2))

        #smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2,p3,p4,p5

#101以上的才为Bottleneck
def FPN101():
    return FPN(Bottleneck,[2,2,2,2])
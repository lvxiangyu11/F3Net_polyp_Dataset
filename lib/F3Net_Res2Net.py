from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.models.tresnet.layers.anti_aliasing import AntiAliasDownsampleLayer
from src.models.tresnet_v2.layers.squeeze_and_excite import SEModule
def Res2Net_v1b_26w_4s(opt, **kwargs):
    return Res2Net(Bottle2neck, [3, 4, 6, 3], opt, baseWidth=26, scale=4, **kwargs)


class Bottle2neck(nn.Module):
    """
        inplace:输入通道数
        planes:输出通道数
        stride:conv stride
        downsample:None if stride==1 else downsample
        baseWidth:conv3x3的宽度
        scale:scale的数量
        type:'normal':noraml set, 'stage':first block of a new stage
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal', dilation=1,
                 padding=1, use_se=False, anti_alias_layer=None):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []

        k = 1
        for i in range(self.nums):
            if (anti_alias_layer != None) & (stride == 2):
                convs.append(nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1,
                              bias=False, dilation=dilation),
                    anti_alias_layer(channels=width, filt_size=3, stride=2)
                ))
            else:
                convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=padding,
                                       bias=False, dilation=dilation))
                # convs.append(aggBlock(width, width, k_size=k, stride=stride))
                # k += 2

            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = (SEModule(width * scale, reduce_layer_planes)) if use_se else None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':  # 为什么stage不加前边的？？？？？
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)  # 为什么第三个加池化部分？？？？
        if self.se is not None : out = self.se(out)
        out = self.conv3(out)  # 16，256，32，32
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    # 这里的block一直用的都是Bottle2Neck
    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale, use_se=use_se
                            , anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale,
                                anti_alias_layer=anti_alias_layer))

        return nn.Sequential(*layers)

    def __init__(self, block, layers, opt, baseWidth=16, scale=4, classesNum=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        if opt.Anti_Alias_Downsample_use:
            self.anti_alias_layer = partial(AntiAliasDownsampleLayer, remove_aa_jit=False)
        else:
            self.anti_alias_layer = None


        self.stem = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0],
                                       anti_alias_layer=self.anti_alias_layer)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       anti_alias_layer=self.anti_alias_layer)


        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       anti_alias_layer=self.anti_alias_layer)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_se=False,
                                           anti_alias_layer=self.anti_alias_layer)


        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, classesNum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # m是否为卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # kaiming初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

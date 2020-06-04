# -*- coding:utf-8 -*-
'''RetinaFPN in PyTorch.'''

import torch.nn as nn
import torch.nn.functional as F

import math

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks, backbone):
        super(FPN, self).__init__()
        self.in_planes = 64
        self.backbone = backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.mobilenetv2 = MobilenetV2()

        # fpn for detection subnet (RetinaNet) P6,P7
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)  # p6
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # p7

        # pure fpn layers for detection subnet (RetinaNet)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # c5 -> p5
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # c4 -> p4
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # c3 -> p3
        # smooth
        self.toplayer0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p5
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p4
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p3

        # pure fpn layers for keypoint subnet
        # Lateral layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # c5 -> p5
        self.flatlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # c4 -> p4
        self.flatlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # c3 -> p3
        self.flatlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # c2 -> p2
        # smooth
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p4
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p3
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # smooth p2

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: top feature map to be upsampled.
          y: lateral feature map.

        Returns:
          added feature map.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='nearest', align_corners=None) + y  # bilinear, False
    def forward(self, x):
        # Bottom-up
        # if self.backbone == 'resnet50':
        #     c1 = F.relu(self.bn1(self.conv1(x)))
        #     c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        #     c2 = self.layer1(c1)
        #     c3 = self.layer2(c2)
        #     c4 = self.layer3(c3)
        #     c5 = self.layer4(c4)
        # else:
        #     c2, c3, c4, c5 = self.mobilenetv2(x)

        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # pure fpn for detection subnet, RetinaNet
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p5 = self.toplayer0(p5)
        p4 = self.toplayer1(p4)
        p3 = self.toplayer2(p3)

        # pure fpn for keypoints estimation
        fp5 = self.toplayer(c5)
        fp4 = self._upsample_add(fp5,self.flatlayer1(c4))
        fp3 = self._upsample_add(fp4,self.flatlayer2(c3))
        fp2 = self._upsample_add(fp3,self.flatlayer3(c2))
        # Smooth
        fp4 = self.smooth1(fp4)
        fp3 = self.smooth2(fp3)
        fp2 = self.smooth3(fp2)

        return [[fp2,fp3,fp4,fp5],[p3, p4, p5, p6, p7]]

#For mobilenetv2

def _make_divisible(v, divisor, min_value=None):
    """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_value:
        :return:
        """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def conv_3x3_bn(inp, out, stride):
    return nn.Sequential(
        nn.Conv2d(inp, out, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out),
        nn.ReLU6(inplace=True)
    )
def conv_1x1_bn(inp, out):
    return nn.Sequential(
        nn.Conv2d(inp, out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out),
        nn.ReLU6(inplace=True)
)
class InvertedResidual(nn.Module):
    def __init__(self, inp, out, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2], 'Stride must be in [1, 2]'
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == out

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, out, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out)
            )
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobilenetV2(nn.Module):
    def __init__(self, num_class=1000, width_mult=1.):
        super(MobilenetV2, self).__init__()
        self.cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, self.input_channel, 2)]
        self.layers = nn.Sequential(*layers)
        block = InvertedResidual
        # input_channel = self.input_channel
        # for t, c, n, s in self.cfgs:
        #     output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
        #     for i in range(n):
        #         layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
        #         input_channel = output_channel
        # self.features = nn.Sequential(*layers)
        self.layer1 = self._make_layer(self.cfgs[0], width_mult, block)
        self.layer2 = self._make_layer(self.cfgs[1], width_mult, block)
        self.my_conv1 = self.my_conv(self.input_channel, 256)
        self.layer3 = self._make_layer(self.cfgs[2], width_mult, block)
        self.my_conv2 = self.my_conv(self.input_channel, 512)
        self.layer4 = self._make_layer(self.cfgs[3], width_mult, block)
        self.layer5 = self._make_layer(self.cfgs[4], width_mult, block)
        self.my_conv3 = self.my_conv(self.input_channel, 1024)
        self.layer6 = self._make_layer(self.cfgs[5], width_mult, block)
        self.layer7 = self._make_layer(self.cfgs[6], width_mult, block)
        self.my_conv4 = self.my_conv(self.input_channel, 2048)
        output_channel = _make_divisible(2048 * width_mult, 4 if width_mult == 0.1 else 8)
        self.conv = conv_1x1_bn(self.input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_class)

        self._initialize_weights()
    def my_conv(self, inp, out):
        return nn.Conv2d(inp, out, 1, 1, 0, bias=False)
    def _make_layer(self, cfg, width_mult, block):
        t, c, n, s = cfg
        output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
        layer = []
        for i in range(n):
            layer.append(block(self.input_channel, output_channel, s if i == 0 else 1, t))
            self.input_channel = output_channel
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.layers(x)
        # x = self.features(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.my_conv1(x)
        x = self.layer3(x)
        x2 = self.my_conv2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x3 = self.my_conv3(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x4 = self.my_conv4(x)
        return x1, x2, x3, x4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight_data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight_data.normal_(0, 0.01)
                    m.bias.data.zero_()

def mobilnetv2(**kwargs):
    return MobilenetV2(**kwargs)
#end mobilenetv2
def FPN50():
    # [3,4,6,3] -> resnet50
    return FPN(Bottleneck, [3,4,6,3], backbone='resnet50')

def FPN101():
    # [3,4,23,3] -> resnet101
    return FPN(Bottleneck, [3,4,23,3], backbone='resnet101')

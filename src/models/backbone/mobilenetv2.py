import torch.nn as nn
import math
import torch.nn.functional as F

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
    def __init__(self, width_mult=1.):
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

        self.input_channel = self.make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [self.conv_3x3_bn(3, self.input_channel, 2)]
        self.layers = nn.Sequential(*layers)
        block = InvertedResidual
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

        self._initialize_weights()
    def forward(self, x):
        x = self.layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c2 = self.my_conv1(x)
        x = self.layer3(x)
        c3 = self.my_conv2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        c4 = self.my_conv3(x)
        x = self.layer6(x)
        x = self.layer7(x)
        c5 = self.my_conv4(x)

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
        fp4 = self._upsample_add(fp5, self.flatlayer1(c4))
        fp3 = self._upsample_add(fp4, self.flatlayer2(c3))
        fp2 = self._upsample_add(fp3, self.flatlayer3(c2))
        # Smooth
        fp4 = self.smooth1(fp4)
        fp3 = self.smooth2(fp3)
        fp2 = self.smooth3(fp2)

        return [[fp2, fp3, fp4, fp5], [p3, p4, p5, p6, p7]]

    def make_divisible(self, v, divisor, min_value=None):
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

    def conv_3x3_bn(self, inp, out, stride):
        return nn.Sequential(
            nn.Conv2d(inp, out, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU6(inplace=True)
        )

    def conv_1x1_bn(self, inp, out):
        return nn.Sequential(
            nn.Conv2d(inp, out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU6(inplace=True)
        )

    def my_conv(self, inp, out):
        return nn.Conv2d(inp, out, 1, 1, 0, bias=False)
    def _make_layer(self, cfg, width_mult, block):
        t, c, n, s = cfg
        output_channel = self.make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
        layer = []
        for i in range(n):
            layer.append(block(self.input_channel, output_channel, s if i == 0 else 1, t))
            self.input_channel = output_channel
        return nn.Sequential(*layer)

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
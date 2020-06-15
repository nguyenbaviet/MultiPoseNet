# -*- coding:utf-8 -*-
'''RetinaFPN in PyTorch.'''

import torch.nn as nn
import torch.nn.functional as F
from src.models.backbone.mobilenetv2 import MobilenetV2
from src.models.backbone.resnet import Resnet

class FPN(nn.Module):
    def __init__(self, backbone):
        super(FPN, self).__init__()
        #basic backbone
        if backbone == 'resnet50':
            self.backbone = Resnet([3,4,6,3])
        elif backbone == 'resnet101':
            self.backbone = Resnet([3,4,23,3])
        elif backbone == 'mobilenetv2':
            self.backbone = MobilenetV2()
        else:
            raise ValueError('Do not support bacbone %s. Expect backbone in [resnet50, resnet101, mobilenetv2]' %backbone)

        #detection backbone
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

        #keypoints backbone
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
        c2, c3, c4, c5 = self.backbone(x)

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
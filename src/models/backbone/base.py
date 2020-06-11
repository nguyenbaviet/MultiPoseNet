# -*- coding:utf-8 -*-
'''RetinaFPN in PyTorch.'''

import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

class Base(nn.Module):
    def __init__(self, backbone):
        super(Base, self).__init__()
        #basic backbone
        self.backbone = backbone

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
    @abstractmethod
    def forward(self, x):
        pass
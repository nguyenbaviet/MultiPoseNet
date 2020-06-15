# -*- coding:utf-8 -*-
# keypoint subnet + detection subnet(RetinaNet) + PRN
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from torch.nn import init
from src.models.backbone.fpn import FPN
from src.utils.utils import BBoxTransform, ClipBoxes
from src.utils.anchors import Anchors
import src.utils.losses as losses

from src.utils.modules import ClassificationModel, nms, RegressionModel, Concat, Base_PRN

class PoseNet(nn.Module):
    def __init__(self, backbone, node_count=1024, coeff=2):
        super(PoseNet, self).__init__()
        self.fpn = FPN(backbone)
        ##################################################################################
        # keypoints subnet
        # intermediate supervision
        self.convfin_k2 = nn.Conv2d(256, 19, kernel_size=1, stride=1, padding=0)
        self.convfin_k3 = nn.Conv2d(256, 19, kernel_size=1, stride=1, padding=0)
        self.convfin_k4 = nn.Conv2d(256, 19, kernel_size=1, stride=1, padding=0)
        self.convfin_k5 = nn.Conv2d(256, 19, kernel_size=1, stride=1, padding=0)

        # 2 conv(kernel=3x3)，change channels from 256 to 128
        self.convt1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convt4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.convs1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.convs4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=8, mode='nearest', align_corners=None)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='nearest', align_corners=None)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        # self.upsample4 = nn.Upsample(size=(120,120),mode='bilinear',align_corners=True)

        self.concat = Concat()
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.convfin = nn.Conv2d(256, 18, kernel_size=1, stride=1, padding=0)

        ##################################################################################
        # detection subnet
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=1)
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = losses.FocalLoss()

        ##################################################################################
        # prn subnet
        self.prn = Base_PRN(node_count, coeff)

        ##################################################################################
        # initialize weights
        self._initialize_weights_norm()
        prior = 0.01
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()  # from retinanet
    def _initialize_weights_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  # resnet101 conv2d doesn't add bias
                    init.constant_(m.bias, 0.0)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    @abstractmethod
    def forward(self, x):
        pass
    @abstractmethod
    def build_loss(self, saved_for_loss, *args):
        pass
    def _predict_bboxANDkeypoints(self, x):
        features = self.fpn(x)
        p2, p3, p4, p5 = features[0]    #fpn features for keypoints subnet
        features = features[1]  #fpn features for detection subnet

        ##################################################################################
        # keypoints subnet
        p5 = self.convt1(p5)
        p5 = self.convs1(p5)
        p4 = self.convt2(p4)
        p4 = self.convs2(p4)
        p3 = self.convt3(p3)
        p3 = self.convs3(p3)
        p2 = self.convt4(p2)
        p2 = self.convs4(p2)

        p5 = self.upsample1(p5)
        p4 = self.upsample2(p4)
        p3 = self.upsample3(p3)

        concat = self.concat(p5, p4, p3, p2)
        predict_keypoint = self.convfin(F.relu(self.conv2(concat)))
        del p5, p4, p3, p2, concat

        ##################################################################################
        # detection subnet
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(x)

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, x)

        scores = torch.max(classification, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores > 0.05)[0, :, 0]  # 0.05

        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return predict_keypoint, [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :],
                              0.5)  # threshold = 0.5, inpsize=480

        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

        return predict_keypoint, [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

    def _predict_prn(self, x):
        res = self.prn.flatten(x)
        out = self.prn.drop(F.relu(self.prn.dens1(res)))
        out = self.prn.drop(F.relu(self.prn.bneck(out)))
        out = F.relu(self.prn.dens2(out))
        out = self.prn.add(out, res)
        out = self.prn.softmax(out)
        out = out.view(out.size()[0], self.prn.height, self.prn.width, 17)

        return out

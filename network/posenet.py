# -*- coding:utf-8 -*-
# keypoint subnet + detection subnet(RetinaNet) + PRN
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict
from network.fpn import FPN50, FPN101
from torch.nn import init

from network.utils import BBoxTransform, ClipBoxes
from network.anchors import Anchors
import network.losses as losses
# from lib.nms.pth_nms import pth_nms
from lib.NMS.nms.gpu_nms import gpu_nms as pth_nms

def nms(dets, thresh):
    "Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor"""
    dets = dets.cpu().numpy()
    return pth_nms(dets, thresh)


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, up1, up2, up3, up4):
        return torch.cat((up1, up2, up3, up4), 1)


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Add(nn.Module):
    def forward(self, input1, input2):
        return torch.add(input1, input2)


class PRN(nn.Module):
    def __init__(self,node_count, coeff):
        super(PRN, self).__init__()
        self.flatten   = Flatten()
        self.height    = coeff*28
        self.width     = coeff*18
        self.dens1     = nn.Linear(self.height*self.width*17, node_count)
        self.bneck     = nn.Linear(node_count, node_count)
        self.dens2     = nn.Linear(node_count, self.height*self.width*17)
        self.drop      = nn.Dropout()
        self.add       = Add()
        self.softmax   = nn.Softmax(dim=1)

    def forward(self, x):
        res = self.flatten(x)
        out = self.drop(F.relu(self.dens1(res)))
        out = self.drop(F.relu(self.bneck(out)))
        out = F.relu(self.dens2(out))
        out = self.add(out, res)
        out = self.softmax(out)
        out = out.view(out.size()[0], self.height, self.width, 17)

        return out

class poseNet(nn.Module):
    def __init__(self,prn_node_count=1024, prn_coeff=2, backbone='resnet50'):
        super(poseNet, self).__init__()
        if backbone == 'resnet101':
            self.fpn = FPN101()
        if backbone == 'resnet50':
            self.fpn = FPN50()
        # self.fpn = FPN101(type)

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
        #viet
        # self.upsample1 = F.interpolate(scale_factor=8, mode='nearest', align_corners=None)
        # self.upsample2 = F.interpolate(scale_factor=4, mode='nearest', align_corners=None)
        # self.upsample3 = F.interpolate(scale_factor=2, mode='nearest', align_corners=None)
        #end viet
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
        self.prn = PRN(prn_node_count, prn_coeff)

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

    def forward(self, x):

        img_batch, subnet_name = x

        if subnet_name == 'keypoint_subnet':
            return self.keypoint_forward(img_batch)
        elif subnet_name == 'detection_subnet':
            return self.detection_forward(img_batch)
        elif subnet_name == 'prn_subnet':
            return self.prn_forward(img_batch)
        else:  # entire_net
            features = self.fpn(img_batch)
            p2, p3, p4, p5 = features[0]  # fpn features for keypoint subnet
            features = features[1]  # fpn features for detection subnet

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
            # print('pre:')
            # print(p5.shape)
            # print(p4.shape)
            # print(p3.shape)
            # print(p2.shape)

            p5 = self.upsample1(p5)
            p4 = self.upsample2(p4)
            p3 = self.upsample3(p3)
            # print('after')
            # print(p5.shape)
            # print(p4.shape)
            # print(p3.shape)
            # raise EOFError

            concat = self.concat(p5, p4, p3, p2)
            predict_keypoint = self.convfin(F.relu(self.conv2(concat)))
            del p5, p4, p3, p2, concat

            ##################################################################################
            # detection subnet
            regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
            anchors = self.anchors(img_batch)

            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]#0.05

            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return predict_keypoint, [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)  # threshold = 0.5, inpsize=480

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return predict_keypoint, [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


    def keypoint_forward(self, img_batch):
        saved_for_loss = []

        p2, p3, p4, p5 = self.fpn(img_batch)[0] # fpn features for keypoint subnet

        ##################################################################################
        # keypoints subnet
        # intermediate supervision
        saved_for_loss.append(self.convfin_k2(p2))
        saved_for_loss.append(self.upsample3(self.convfin_k3(p3)))
        saved_for_loss.append(self.upsample2(self.convfin_k4(p4)))
        saved_for_loss.append(self.upsample1(self.convfin_k5(p5)))

        #
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

        predict_keypoint = self.convfin(F.relu(self.conv2(self.concat(p5, p4, p3, p2))))
        saved_for_loss.append(predict_keypoint)

        return predict_keypoint, saved_for_loss

    def detection_forward(self, img_batch):
        saved_for_loss = []

        features = self.fpn(img_batch)[1]  # fpn features for detection subnet

        ##################################################################################
        # detection subnet
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)

        saved_for_loss.append(classification)
        saved_for_loss.append(regression)
        saved_for_loss.append(anchors)

        return [], saved_for_loss

    def prn_forward(self, img_batch):
        saved_for_loss = []

        res = self.prn.flatten(img_batch)
        out = self.prn.drop(F.relu(self.prn.dens1(res)))
        out = self.prn.drop(F.relu(self.prn.bneck(out)))
        out = F.relu(self.prn.dens2(out))
        out = self.prn.add(out,res)
        out = self.prn.softmax(out)
        out = out.view(out.size()[0], self.prn.height, self.prn.width, 17)

        saved_for_loss.append(out)

        return out, saved_for_loss

    @staticmethod
    def build_loss(saved_for_loss, *args):

        subnet_name = args[0]

        if subnet_name == 'keypoint_subnet':
            return build_keypoint_loss(saved_for_loss, args[1], args[2])
        elif subnet_name == 'detection_subnet':
            return build_detection_loss(saved_for_loss, args[1])
        elif subnet_name == 'prn_subnet':
            return build_prn_loss(saved_for_loss, args[1])
        else:
            return 0

class poseNet_detectANDkeypoint(poseNet):
    def forward(self, img_batch):
        features = self.fpn(img_batch)
        p2, p3, p4, p5 = features[0]  # fpn features for keypoint subnet
        features = features[1]  # fpn features for detection subnet

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

        # p5 = self.upsample1(p5)
        # p4 = self.upsample2(p4)
        # p3 = self.upsample3(p3)
        p5 = F.interpolate(p5, scale_factor=8, mode='nearest', align_corners=None)
        p4 = F.interpolate(p4, scale_factor=4, mode='nearest', align_corners=None)
        p3 = F.interpolate(p3, scale_factor=2, mode='nearest', align_corners=None)

        concat = self.concat(p5, p4, p3, p2)
        predict_keypoint = self.convfin(F.relu(self.conv2(concat)))

        ##################################################################################
        # detection subnet
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        return predict_keypoint, regression, classification
        # anchors = self.anchors(img_batch)
        #
        # transformed_anchors = self.regressBoxes(anchors, regression)
        # transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
        #
        # return predict_keypoint, classification, transformed_anchors
        # scores = torch.max(classification, dim=2, keepdim=True)[0]
        # scores_over_thresh = (scores > 0.05)[0, :, 0]  # 0.05
        #
        # if scores_over_thresh.sum() == 0:
        #     # no boxes to NMS, just return
        #     return predict_keypoint, [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
        #
        # classification = classification[:, scores_over_thresh, :]
        # transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        # scores = scores[:, scores_over_thresh, :]
        #
        # anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :],
        #                       0.5)  # threshold = 0.5, inpsize=480
        #
        # nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        #
        # return predict_keypoint, [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

    def export_onnx(self, export_path):
        input_tensor = torch.rand(1, 3, 480, 480)
        with torch.no_grad():
            torch.onnx.export(self, input_tensor, export_path, export_params=True, opset_version=11)

class poseNet_PRN(poseNet):
    def forward(self, img_batch):
        pass
def build_keypoint_loss(saved_for_loss, heat_temp, heat_weight):

    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(size_average=True).cuda()
    total_loss = 0
    div1 = 1.
    #div2 = 100.

    for j in range(5):

        pred1 = saved_for_loss[j][:, :18, :, :] * heat_weight
        gt1 = heat_weight * heat_temp

        #pred2 = saved_for_loss[j][:, 18:, :, :]
        #gt2 = mask_all

        # Compute losses
        loss1 = criterion(pred1, gt1)/div1  # heatmap_loss
        #loss2 = criterion(pred2, gt2)/div2  # mask_loss
        total_loss += loss1
        #total_loss += loss2

        # Get value from Tensor and save for log
        saved_for_log[names[j*2]] = loss1.item()
        #saved_for_log[names[j*2+1]] = loss2.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-1].data[:, :18, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-1].data[:, :18, :, :]).item()
    #saved_for_log['max_mask'] = torch.max(
    #    saved_for_loss[-1].data[:, 18:, :, :]).item()
    #saved_for_log['min_mask'] = torch.min(
    #    saved_for_loss[-1].data[:, 18:, :, :]).item()

    return total_loss, saved_for_log

def build_detection_loss(saved_for_loss, anno):
    '''
    :param saved_for_loss: [classifications, regressions, anchors]
    :param anno: annotations
    :return: classification_loss, regression_loss
    '''
    saved_for_log = OrderedDict()

    # Compute losses
    focalLoss = losses.FocalLoss()
    classification_loss, regression_loss = focalLoss(*saved_for_loss, anno)
    classification_loss = classification_loss.mean()
    regression_loss = regression_loss.mean()
    total_loss = classification_loss + regression_loss

    # Get value from Tensor and save for log
    saved_for_log['total_loss'] = total_loss.item()
    saved_for_log['classification_loss'] = classification_loss.item()
    saved_for_log['regression_loss'] = regression_loss.item()

    return total_loss, saved_for_log

def build_prn_loss(saved_for_loss, label):
    '''
    :param saved_for_loss: [out]
    :param label: label
    :return: prn loss
    '''
    saved_for_log = OrderedDict()

    criterion = nn.BCELoss(size_average=True).cuda()
    total_loss = 0

    # Compute losses
    loss1 = criterion(saved_for_loss[0], label)
    total_loss += loss1

    # Get value from Tensor and save for log
    saved_for_log['PRN loss'] = loss1.item()

    return total_loss, saved_for_log

def build_names():
    names = []
    for j in range(2, 6):
        names.append('heatmap_loss_k%d' % j)
        names.append('seg_loss_k%d' % j)
    names.append('heatmap_loss')
    names.append('seg_loss')
    return names


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.lib.NMS.nms.gpu_nms import gpu_nms as pth_nms


def nms(dets, thresh):
    "Dispatch to either CPU or GPU NMS implementations.\
    Accept dets as tensor"""
    dets = dets.cpu().numpy()
    return pth_nms(dets, thresh)

def build_names():
    names = []
    for j in range(2, 6):
        names.append('heatmap_loss_k%d' % j)
        names.append('seg_loss_k%d' % j)
    names.append('heatmap_loss')
    names.append('seg_loss')
    return names

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

class Base_PRN(nn.Module):
    def __init__(self,node_count, coeff):
        super(Base_PRN, self).__init__()
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
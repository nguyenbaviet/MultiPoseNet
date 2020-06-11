import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from src.utils.modules import build_names
from src.models.base import PoseNet

class KeypointsEstimation(PoseNet):
    def forward(self, x):
        saved_for_loss = []

        p2, p3, p4, p5 = self.fpn(x)[0]  # fpn features for keypoint subnet

        ##################################################################################
        # keypoints subnet
        # intermediate supervision
        saved_for_loss.append(self.convfin_k2(p2))
        saved_for_loss.append(self.upsample3(self.convfin_k3(p3)))
        saved_for_loss.append(self.upsample2(self.convfin_k4(p4)))
        saved_for_loss.append(self.upsample1(self.convfin_k5(p5)))

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

    def build_loss(self, saved_for_loss, *args):
        heat_temp = args[0]
        heat_weight = args[1]

        names = build_names()
        saved_for_log = OrderedDict()
        criterion = nn.MSELoss(size_average=True).cuda()
        total_loss = 0
        div1 = 1.

        for j in range(5):
            pred1 = saved_for_loss[j][:, :18, :, :] * heat_weight
            gt1 = heat_weight * heat_temp

            # Compute losses
            loss1 = criterion(pred1, gt1) / div1  # heatmap_loss
            total_loss += loss1

            # Get value from Tensor and save for log
            saved_for_log[names[j * 2]] = loss1.item()

        saved_for_log['max_ht'] = torch.max(
            saved_for_loss[-1].data[:, :18, :, :]).item()
        saved_for_log['min_ht'] = torch.min(
            saved_for_loss[-1].data[:, :18, :, :]).item()

        return total_loss, saved_for_log
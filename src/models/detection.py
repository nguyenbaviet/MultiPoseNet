import torch
from collections import OrderedDict

import src.utils.losses as losses
from src.models.base import PoseNet


class Detection(PoseNet):
    def forward(self, x):
        # x, _ = x
        saved_for_loss = []

        features = self.fpn(x)[1]  # fpn features for detection subnet

        ##################################################################################
        # detection subnet
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(x)

        saved_for_loss.append(classification)
        saved_for_loss.append(regression)
        saved_for_loss.append(anchors)

        return [], saved_for_loss
    def build_loss(self, saved_for_loss, *args):
        '''
            :param saved_for_loss: [classifications, regressions, anchors]
            :param anno: annotations
            :return: classification_loss, regression_loss
            '''

        anno = args[0]
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
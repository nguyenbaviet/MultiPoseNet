import torch
import torch.nn.functional as F

from src.models.base import PoseNet

class detectionANDkeypoints(PoseNet):
    def forward(self, x):
        features = self.fpn(x)
        p2, p3, p4, p5 = features[0]  # fpn features for keypoints subnet
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

        p5 = self.upsample1(p5)
        p4 = self.upsample2(p4)
        p3 = self.upsample3(p3)

        concat = self.concat(p5, p4, p3, p2)
        predict_keypoint = self.convfin(F.relu(self.conv2(concat)))

        ##################################################################################
        # detection subnet
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        return predict_keypoint, regression, classification
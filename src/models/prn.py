import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from src.models.base import PoseNet


class PRN(PoseNet):
    def forward(self, x):
        saved_for_loss = []

        res = self.prn.flatten(x)
        out = self.prn.drop(F.relu(self.prn.dens1(res)))
        out = self.prn.drop(F.relu(self.prn.bneck(out)))
        out = F.relu(self.prn.dens2(out))
        out = self.prn.add(out, res)
        out = self.prn.softmax(out)
        out = out.view(out.size()[0], self.prn.height, self.prn.width, 17)

        saved_for_loss.append(out)

        return out, saved_for_loss
    def build_loss(self, saved_for_loss, *args):
        '''
        :param saved_for_loss: [out]
        :param label: label
        :return: prn loss
        '''

        label = args[0]
        saved_for_log = OrderedDict()

        criterion = nn.BCELoss(size_average=True).cuda()
        total_loss = 0

        # Compute losses
        loss1 = criterion(saved_for_loss[0], label)
        total_loss += loss1

        # Get value from Tensor and save for log
        saved_for_log['PRN loss'] = loss1.item()

        return total_loss, saved_for_log
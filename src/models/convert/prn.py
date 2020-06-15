import torch.nn.functional as F

from src.models.base import PoseNet

class _PRN(PoseNet):
    def forward(self, x):
        res = self.prn.flatten(x)
        out = self.prn.drop(F.relu(self.prn.dens1(res)))
        out = self.prn.drop(F.relu(self.prn.bneck(out)))
        out = F.relu(self.prn.dens2(out))
        out = self.prn.add(out, res)
        out = self.prn.softmax(out)
        out = out.view(out.size()[0], self.prn.height, self.prn.width, 17)
        return out
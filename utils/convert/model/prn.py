import torch.nn.functional as F

from src.models.base import PoseNet

class _PRN(PoseNet):
    def forward(self, x):
        res = self.flatten(x)
        out = self.drop(F.relu(self.dens1(res)))
        out = self.drop(F.relu(self.bneck(out)))
        out = F.relu(self.dens2(out))
        out = self.add(out, res)
        out = self.softmax(out)
        out = out.view(out.size()[0], self.height, self.width, 17)

        return out
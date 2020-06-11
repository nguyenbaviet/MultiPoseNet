import os, sys
sys.path.append(os.getcwd())

from src.models.base import PoseNet
from src.runner.tester import Tester
from src.utils.utils import init_Test_params
import yaml


def main(cfg):
    gpus = ','.join([str(x) for x in cfg['COMMON']['GPUS']])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    model = PoseNet(backbone=cfg['MODEL']['backbone'])
    for name, module in model.named_children():
        for para in module.parameters():
            para.requires_grad = False
    params = init_Test_params(cfg)
    tester = Tester(model, params)
    tester.test()
if __name__ == '__main__':
    with open('configs/val.yaml') as f:
        cfg = yaml.full_load(f)
    main(cfg)
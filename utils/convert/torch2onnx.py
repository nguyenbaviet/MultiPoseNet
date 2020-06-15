import sys, os
sys.path.append(os.getcwd())

import argparse
from src.models.convert.detectionANDkeypoints import detectionANDkeypoints
from src.models.convert.prn import _PRN
from src.utils.net_utils import load_net
from utils.convert._utils import export_onnx
import yaml

def main(cfg):
    print("===> creating convert {} with backbone '{}'".format(cfg['model'], cfg['backbone']))
    if cfg['model'] == 'detectionANDkeypoints':
        model = detectionANDkeypoints(cfg['backbone'])
    else:
        model = _PRN(cfg['backbone'])
    if cfg['checkpoint']:
        if os.path.isfile(cfg['checkpoint']):
            print("=> loading checkpoint '{}'".format(cfg['checkpoint']))

            _, _ = load_net(cfg['checkpoint'], model, load_state_dict=True)
            print("=> loaded checkpoint '{}'".format(cfg['checkpoint']))
        else:
            print("=> no checkpoint found at '{}'".format(cfg['checkpoint']))
    else:
        print("=> no checkpoint found at '{}'".format(cfg['checkpoint']))
    model.eval()
    export_onnx(model, cfg['output_path'])

if __name__ == '__main__':
    with open('configs/convert_onnx.yaml') as f:
        cfg = yaml.full_load(f)
    main(cfg)
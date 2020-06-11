import sys, os
sys.path.append(os.getcwd())

import argparse
from utils.convert.model.detectionANDkeypoints import detectionANDkeypoints
from utils.convert.model.prn import _PRN
from src.utils.net_utils import load_net

def main(args):
    print("===> creating model {} with backbone '{}'".format(args.model, args.backbone))
    if args.model == 'detectionANDkeypoints':
        model = detectionANDkeypoints(args.backbone)
    else:
        model = _PRN(args.backbone)
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=> loading checkpoint '{}'".format(args.checkpoint))

            _, _ = load_net(args.checkpoint, model, load_state_dict=True)
            print("=> loaded checkpoint '{}'".format(args.checkpoint))
        else:
            print("=> no checkpoint found at '{}'".format(args.checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
    model.eval()
    model.export_onnx(args.out_onnx)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model', default='detectionANDkeypoints')
    args.add_argument('--backbone', default='resnet101')
    args.add_argument('--out-onnx', default='resnet101.onnx')
    args.add_argument('--checkpoint', default=os.getcwd() + '/demo/models/ckpt_baseline_resnet101.h5')

    main(args.parse_args())
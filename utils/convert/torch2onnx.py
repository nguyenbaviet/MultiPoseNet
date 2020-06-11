import sys, os
sys.path.append(os.getcwd())

import argparse
from network.posenet import poseNet_detectANDkeypoint
from network.net_utils import load_net

def main(args):
    print("===> creating model with backbone '{}'".format(args.backbone))
    model = poseNet_detectANDkeypoint(backbone=args.backbone)
    # model.cuda()
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

def test(model_path, result_path):
    import onnx
    from onnx import version_converter, helper

    # load model
    original_model = onnx.load(model_path)

    # converts oppset v9 to v8
    converted_model = original_model

    # change attribute of all Upsample nodes
    for node in converted_model.graph.node:
        if node.op_type == 'Upsample':
            # get id-attribute_name map
            id = {attribute.name: id for id, attribute in enumerate(node.attribute)}
            # get & remove "scales" attribute
            att_scales = node.attribute.pop(id['scales'])
            _, _, scale_height, scale_width = att_scales.floats  # CARE IT DEPENDS ON ORDER. HERE [B, C, W, H] IS EXPECTED
            # append new attributes 'scale_width' & 'scale_height'
            node.attribute.extend([
                helper.make_attribute('width_scale', scale_width),
                helper.make_attribute('height_scale', scale_height)
            ])

    # save
    onnx.save(converted_model, result_path)
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--layers', default=101, type=int)
    args.add_argument('--backbone', default='resnet101')
    args.add_argument('--out-onnx', default='resnet101.onnx')
    args.add_argument('--checkpoint', default=os.getcwd() + '/demo/models/ckpt_baseline_resnet101.h5')

    main(args.parse_args())
    # test('resnet101.onnx', 'mytest.onnx')
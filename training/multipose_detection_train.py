import os, sys
root_path = os.path.realpath(__file__).split('/training/multipose_detection_train.py')[0]
os.chdir(root_path)
sys.path.append(root_path)

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.batch_processor import batch_processor
from network.posenet import poseNet
from datasets.coco import get_loader
from training.trainer import Trainer
import torch.nn as nn
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', default='resnet50')
args = parser.parse_args()

# Hyper-params
coco_root = '/mnt/hdd10tb/Datasets/COCO2017/'
backbone = 'resnet50'  # 'resnet50'
opt = 'adam'
weight_decay = 0.000
inp_size = 608  # input size 608*608
feat_stride = 4

# model parameters in MultiPoseNet
fpn_resnet_para = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
fpn_retinanet_para = ['conv6', 'conv7', 'latlayer1', 'latlayer2',
                      'latlayer3', 'toplayer0', 'toplayer1', 'toplayer2']
fpn_keypoint_para = ['toplayer', 'flatlayer1', 'flatlayer2',
                     'flatlayer3', 'smooth1', 'smooth2', 'smooth3']
retinanet_para = ['regressionModel', 'classificationModel']
keypoint_para = ['convt1', 'convt2', 'convt3', 'convt4', 'convs1', 'convs2', 'convs3', 'convs4', 'upsample1',
                 'upsample2', 'upsample3', 'conv2', 'convfin', 'convfin_k2', 'convfin_k3', 'convfin_k4', 'convfin_k5']
prn_para = ['prn']

#####################################################################
# train detection subnet
data_dir = coco_root+'images/'
mask_dir = coco_root
json_path = '/home/vietnguyen/MultiPoseNet/COCO.json'

# Set Training parameters
params = Trainer.TrainParams()
params.exp_name = 'res50_detection_subnet/' if args.backbone == 'resnet50' else 'mobilenetv2_detection_subnet/'
params.subnet_name = 'detection_subnet'
params.save_dir = './extra/models/{}'.format(params.exp_name)
params.ckpt = '/home/vietnguyen/MultiPoseNet/extra/models/res50_detection_subnet/ckpt_39_0.59604.h5.best'
# params.ckpt = None
params.ignore_opt_state = True
# params.re_init = True

params.max_epoch = 50
params.init_lr = 1.e-5
params.lr_decay = 0.1

params.gpus = [0, 1, 2]
params.batch_size = 20 * len(params.gpus)
params.val_nbatch_end_epoch = 2000

params.print_freq = 50

# model
model = poseNet(backbone=args.backbone)

# import torch.utils.model_zoo as model_zoo
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }

# load pretrained
# if params.ckpt is None:
#     model.fpn.load_state_dict(model_zoo.load_url(
#         model_urls[backbone]), strict=False)
# Train detection subnet (RetinaNet), Fix the weights in backbone (ResNet) ans Key-point Subnet
# for name, module in model.fpn.named_children():
#     if name in fpn_resnet_para:
#         for para in module.parameters():
#             para.requires_grad = False
for name, module in model.fpn.named_children():
    if name in fpn_keypoint_para:
        for para in module.parameters():
            para.requires_grad = False
for name, module in model.named_children():
    if name in keypoint_para:
        for para in module.parameters():
            para.requires_grad = False
for name, module in model.named_children():
    if name in prn_para:
        for para in module.parameters():
            para.requires_grad = False

print("Loading dataset...")
# load training data
train_data = get_loader(json_path, data_dir, mask_dir, inp_size, feat_stride,
                        preprocess='resnet', batch_size=params.batch_size, training=True,
                        shuffle=True, num_workers=8, subnet=params.subnet_name)
print('train dataset len: {}'.format(len(train_data.dataset)))

# load validation data
valid_data = get_loader(json_path, data_dir, mask_dir, inp_size, feat_stride,
                        preprocess='resnet', batch_size=params.batch_size-10*len(params.gpus), training=False,
                        shuffle=False, num_workers=8, subnet=params.subnet_name)
print('val dataset len: {}'.format(len(valid_data.dataset)))

trainable_vars = [param for param in model.parameters() if param.requires_grad]
if opt == 'adam':
    print("Training with adam")
    params.optimizer = torch.optim.Adam(
        trainable_vars, lr=params.init_lr, weight_decay=weight_decay)

params.lr_scheduler = ReduceLROnPlateau(
    params.optimizer, 'min', factor=params.lr_decay, patience=3, verbose=True)
# if torch.cuda.device_count()> 1:
#     model = nn.DataParallel(model)
trainer = Trainer(model, params, batch_processor, train_data, valid_data)
trainer.train()

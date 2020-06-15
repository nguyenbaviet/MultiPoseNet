import os, sys
sys.path.append(os.getcwd())

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils.batch_processor import batch_processor
from src.models.detection import Detection
from src.models.keypointsEstimation import KeypointsEstimation
from src.models.prn import PRN
from src.runner.trainer import Trainer
from src.utils.utils import frozen_weights, init_train_params
from src.datasets.prn_data_pipeline import PRN_CocoDataset
from src.datasets.COCO_data_pipeline import Cocobbox, Cocokeypoints, bbox_collater
from torchvision.transforms import ToTensor
from src.datasets.utils.dataloader import sDataLoader

import yaml
from pycocotools.coco import COCO
import json

def main(cfg):
    gpus = ','.join([str(x) for x in cfg['COMMON']['GPUS']])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    if cfg['MODEL']['subnet'] == 'detection':
        model = Detection(backbone=cfg['MODEL']['backbone'])
    elif cfg['MODEL']['subnet'] == 'keypoints':
        model = KeypointsEstimation(backbone=cfg['MODEL']['backbone'])
    elif cfg['MODEL']['subnet'] == 'prn':
        model = PRN(backbone=cfg['MODEL']['backbone'])

    len_gpu = len(cfg['COMMON']['GPUS'])
    model = frozen_weights(model=model, cfg=cfg)
    params = init_train_params(cfg)

    print('===> Loading dataset')
    root = cfg['DATASET']['data_dir']
    mask_dir = cfg['DATASET']['mask_dir']
    inp_size = cfg['MODEL']['inp_size']
    feat_stride = cfg['MODEL']['feat_stride']
    batch_size = cfg['TRAIN']['batch_size'] * len_gpu
    if params.subnet_name == 'keypoints':
        with open(cfg['DATASET']['json_path']) as f:
            data = json.load(f)
        train_indexes = []
        val_indexes = []
        for count in range(len(data)):
            if data[count]['isValidation'] != 0.:
                val_indexes.append(count)
            else:
                train_indexes.append(count)
        train_data = Cocokeypoints(root=root, mask_dir=mask_dir, index_list=train_indexes, data=data, inp_size=inp_size,
                                   feat_stride=feat_stride, preprocess='resnet', transform=ToTensor())
        train_data = sDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

        val_data = Cocokeypoints(root=root, mask_dir=mask_dir, index_list=val_indexes, data=data, inp_size=inp_size,
                                 feat_stride=feat_stride, preprocess='resnet', transform=ToTensor())
        val_data = sDataLoader(val_data, batch_size=int(batch_size / 2), shuffle=False, num_workers=8)
    elif params.subnet_name == 'detection':
        with open(cfg['DATASET']['json_path']) as f:
            data = json.load(f)
        coco_root = os.path.abspath(os.path.join(mask_dir, os.pardir))
        train_anno = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')
        val_anno = os.path.join(coco_root, 'annotations', 'person_keypoints_val2017.json')
        coco_train = COCO(train_anno)
        images_ids = coco_train.getImgIds()
        data_indexes = []
        for count in range(len(data)):
            if int(data[count]['image_id']) in images_ids:
                data_indexes.append(count)
        train_data = Cocobbox(root=root, mask_dir=mask_dir, index_list=data_indexes, data=data, inp_size=inp_size,
                              feat_stride=feat_stride, coco=coco_train, preprocess='resnet', training=True)
        train_data = sDataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=bbox_collater)

        coco_val = COCO(val_anno)
        images_ids = coco_val.getImgIds()
        data_indexes = []
        for count in range(len(data)):
            if int(data[count]['image_id']) in images_ids:
                data_indexes.append(count)
        val_data = Cocobbox(root=root, mask_dir=mask_dir, index_list=data_indexes, data=data, inp_size=inp_size,
                              feat_stride=feat_stride, coco=coco_val, preprocess='resnet', training=False)
        val_data = sDataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=bbox_collater)
    else:
        num_of_keyponts = cfg['TRAIN']['num_of_keypoints']
        coco_root = os.path.abspath(os.path.join(root, os.path.pardir))
        coco_train = COCO(os.path.join(coco_root, 'annotations/person_keypoints_train2017.json'))
        train_data = DataLoader(dataset=PRN_CocoDataset(
            coco_train, num_of_keypoints=num_of_keyponts, coeff=2, threshold=0.21, inp_size=inp_size,
            feat_stride=feat_stride), batch_size=batch_size, num_workers=4, shuffle=True)
        coco_val = COCO(os.path.join(coco_root, 'annotations/person_keypoints_val2017.json'))
        val_data = DataLoader(dataset=PRN_CocoDataset(
            coco_val, num_of_keypoints=num_of_keyponts, coeff=2, threshold=0.21, inp_size=inp_size,
            feat_stride=feat_stride), batch_size=batch_size, num_workers=4, shuffle=True)

    print('train dataset len: {}'.format(len(train_data.dataset)))
    print('val dataset len: {}'.format(len(val_data.dataset)))
    trainable_vars = [param for param in model.parameters() if param.requires_grad]

    if cfg['TRAIN']['optimizer'] == 'adam':
        print('training with adam')
        params.optimizer = torch.optim.Adam(trainable_vars, lr=params.init_lr, weight_decay=cfg['TRAIN']['weight_decay'])
    params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=params.lr_decay, patience=3, verbose=True)
    trainer = Trainer(model, params, batch_processor, train_data, val_data)
    trainer.train()
if __name__ == '__main__':
    with open('configs/train.yaml') as f:
        cfg = yaml.full_load(f)
    main(cfg)
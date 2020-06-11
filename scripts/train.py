import os, sys
sys.path.append(os.getcwd())

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.utils.batch_processor import batch_processor
from src.models.detection import Detection
from src.models.keypointsEstimation import KeypointsEstimation
from src.models.prn import PRN
from src.datasets.coco import get_loader
from src.runner.trainer import Trainer
from src.utils.utils import frozen_weights, init_train_params
from src.datasets.coco_data.prn_data_pipeline import PRN_CocoDataset

import yaml
from pycocotools.coco import COCO

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
    if params.subnet_name != 'prn':
        train_data = get_loader(json_path=cfg['DATASET']['json_path'], data_dir=cfg['DATASET']['data_dir'],
                                mask_dir=cfg['DATASET']['mask_dir'],
                                inp_size=cfg['MODEL']['inp_size'], feat_stride=cfg['MODEL']['feat_stride'],
                                preprocess='resnet', batch_size=cfg['TRAIN']['batch_size'] * len_gpu, training=True,
                                shuffle=True, num_workers=8, subnet=cfg['MODEL']['subnet'])
        print('train dataset len: {}'.format(len(train_data.dataset)))

        valid_data = get_loader(json_path=cfg['DATASET']['json_path'], data_dir=cfg['DATASET']['data_dir'],
                                mask_dir=cfg['DATASET']['mask_dir'],
                                inp_size=cfg['MODEL']['inp_size'], feat_stride=cfg['MODEL']['feat_stride'],
                                preprocess='resnet', batch_size=cfg['TRAIN']['batch_size'] * len_gpu, training=False,
                                shuffle=False, num_workers=8, subnet=cfg['MODEL']['subnet'])
    else:
        num_of_keyponts = cfg['TRAIN']['num_of_keypoints']
        coco_root = os.path.abspath(os.path.join(cfg['DATASET']['data_dir'], os.path.pardir))
        coco_train = COCO(os.path.join(coco_root, 'annotations/person_keypoints_train2017.json'))
        train_data = DataLoader(dataset=PRN_CocoDataset(
            coco_train, num_of_keypoints=num_of_keyponts, coeff=2, threshold=0.21, inp_size=cfg['MODEL']['inp_size'],
            feat_stride=cfg['MODEL']['feat_stride']), batch_size=cfg['TRAIN']['batch_size'] * len_gpu, num_workers=4, shuffle=True)
        print('train dataset len: {}'.format(len(train_data.dataset)))
        coco_val = COCO(os.path.join(coco_root, 'annotations/person_keypoints_val2017.json'))
        valid_data = DataLoader(dataset=PRN_CocoDataset(
            coco_val, num_of_keypoints=num_of_keyponts, coeff=2, threshold=0.21, inp_size=cfg['MODEL']['inp_size'],
            feat_stride=cfg['MODEL']['feat_stride']), batch_size=cfg['TRAIN']['batch_size'] * len_gpu, num_workers=4, shuffle=True)
        print('val dataset len: {}'.format(len(valid_data.dataset)))
    trainable_vars = [param for param in model.parameters() if param.requires_grad]

    if cfg['TRAIN']['optimizer'] == 'adam':
        print('training with adam')
        params.optimizer = torch.optim.Adam(trainable_vars, lr=params.init_lr, weight_decay=cfg['TRAIN']['weight_decay'])
    params.lr_scheduler = ReduceLROnPlateau(params.optimizer, 'min', factor=params.lr_decay, patience=3, verbose=True)
    trainer = Trainer(model, params, batch_processor, train_data, valid_data)
    trainer.train()
if __name__ == '__main__':
    with open('configs/train.yaml') as f:
        cfg = yaml.full_load(f)
    main(cfg)
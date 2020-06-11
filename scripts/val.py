import os, sys
sys.path.append(os.getcwd())

from src.models.detection import Detection
from src.models.keypointsEstimation import KeypointsEstimation
from src.models.prn import PRN
from pycocotools.coco import COCO
from src.datasets.coco_data.prn_data_pipeline import PRN_CocoDataset
from torch.utils.data import DataLoader
from src.datasets.coco import get_loader
from src.runner.tester import Tester
from src.utils.batch_processor import batch_processor
from src.utils.utils import init_Test_params
import yaml


def main(cfg):
    gpus = ','.join([str(x) for x in cfg['COMMON']['GPUS']])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    if cfg['MODEL']['subnet'] == 'detection':
        model = Detection(backbone=cfg['MODEL']['backbone'])
    elif cfg['MODEL']['subnet'] == 'keypoints':
        model = KeypointsEstimation(backbone=cfg['MODEL']['backbone'])
    elif cfg['MODEL']['subnet'] == 'prn':
        model = PRN(backbone=cfg['MODEL']['backbone'])


    for name, module in model.named_children():
        for para in module.parameters():
            para.requires_grad = False
    len_gpu = len(cfg['COMMON']['GPUS'])
    params = init_Test_params(cfg)
    if cfg['MODEL']['subnet'] != 'prn':
        valid_data = get_loader(json_path=cfg['DATASET']['json_path'], data_dir=cfg['DATASET']['data_dir'], mask_dir=cfg['DATASET']['mask_dir'],
                                inp_size=cfg['MODEL']['inp_size'], feat_stride=cfg['MODEL']['feat_stride'],
                                preprocess='resnet', batch_size=cfg['VAL']['batch_size'] * len_gpu, training=False,
                                shuffle=False, num_workers=8, subnet=cfg['MODEL']['subnet'])
    else:
        coco_val = COCO(os.path.join(cfg['DATASET']['coco_root'], 'annotations/person_keypoints_val2017.json'))
        valid_data = DataLoader(dataset=PRN_CocoDataset(
            coco_val, num_of_keypoints=cfg['VAL']['num_of_keypoints'], coeff=2, threshold=0.21, inp_size=cfg['MODEL']['inp_size'],
            feat_stride=cfg['MODEL']['feat_stride']), batch_size=cfg['VAL']['batch_size'] * len_gpu, num_worker=4,
            shuffle=True)
    print('val dataset len: {}'.format(len(valid_data.dataset)))
    tester = Tester(model, params, batch_processor, valid_data)
    if cfg['VAL']['measure'] == 'loss':
        tester.val()
    elif cfg['VAL']['measure'] == 'accuracy':
        tester.coco_eval()
    else:
        raise ValueError('not supported measure %s. Expected measure in [loss, accuracy]' % cfg['VAL']['measure'])
if __name__ == '__main__':
    with open('configs/val.yaml') as f:
        cfg = yaml.full_load(f)
    main(cfg)
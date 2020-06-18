import os, sys
sys.path.append(os.getcwd())

from src.models.detection import Detection
from src.models.keypointsEstimation import KeypointsEstimation
from src.models.prn import PRN
from pycocotools.coco import COCO
from src.datasets.prn_data_pipeline import PRN_CocoDataset
from torch.utils.data import DataLoader
from src.runner.evaluator import Evaluator
from src.utils.batch_processor import batch_processor
from src.utils.utils import init_Test_params
import yaml
import json
from src.datasets.COCO_data_pipeline import Cocobbox, Cocokeypoints, bbox_collater
from torchvision.transforms import ToTensor
from src.datasets.utils.dataloader import sDataLoader


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
    root = cfg['DATASET']['data_dir']
    mask_dir = cfg['DATASET']['mask_dir']
    inp_size = cfg['MODEL']['inp_size']
    feat_stride = cfg['MODEL']['feat_stride']
    batch_size = cfg['VAL']['batch_size'] * len_gpu
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
        val_data = Cocokeypoints(root=root, mask_dir=mask_dir, index_list=val_indexes if cfg['VAL']['dataset_type'] == 'val' else train_indexes,
                                 data=data, inp_size=inp_size,
                                 feat_stride=feat_stride, preprocess='resnet', transform=ToTensor())
        val_data = sDataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    elif params.subnet_name == 'detection':
        with open(cfg['DATASET']['json_path']) as f:
            data = json.load(f)
        coco_root = os.path.abspath(os.path.join(mask_dir, os.pardir))
        train_anno = os.path.join(coco_root, 'annotations', 'person_keypoints_train2017.json')
        val_anno = os.path.join(coco_root, 'annotations', 'person_keypoints_val2017.json')
        anno = val_anno if cfg['VAL']['dataset_type'] == 'val' else train_anno
        coco = COCO(anno)
        images_ids = coco.getImgIds()
        data_indexes = []
        for count in range(len(data)):
            if int(data[count]['image_id']) in images_ids:
                data_indexes.append(count)
        val_data = Cocobbox(root=root, mask_dir=mask_dir, index_list=data_indexes, data=data, inp_size=inp_size,
                              feat_stride=feat_stride, coco=coco, preprocess='resnet', training=False)
        val_data = sDataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=bbox_collater)
    else:
        num_of_keyponts = cfg['TRAIN']['num_of_keypoints']
        coco_root = os.path.abspath(os.path.join(root, os.path.pardir))
        anno = 'annotations/person_keypoints_val2017.json' if cfg['VAL']['dataset_type'] == 'val' else 'annotations/person_keypoints_train2017.json'
        coco = COCO(os.path.join(coco_root, anno))
        val_data = DataLoader(dataset=PRN_CocoDataset(
            coco, num_of_keypoints=num_of_keyponts, coeff=2, threshold=0.21, inp_size=inp_size,
            feat_stride=feat_stride), batch_size=batch_size, num_workers=4, shuffle=False)
    print('val dataset len: {}'.format(len(val_data.dataset)))

    evaluator = Evaluator(model, params, batch_processor, val_data)

    if cfg['VAL']['measure'] == 'loss':
        evaluator.val()
    elif cfg['VAL']['measure'] == 'accuracy':
        evaluator.coco_eval()
    else:
        raise ValueError('not supported measure %s. Expected measure in [loss, accuracy]' % cfg['VAL']['measure'])
if __name__ == '__main__':
    with open('configs/val.yaml') as f:
        cfg = yaml.full_load(f)
    main(cfg)
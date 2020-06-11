import sys, os
sys.path.append(os.getcwd())

import json
from pycocotools.coco import COCO


annTypes = ['instances', 'captions', 'person_keypoints']
annType = annTypes[2]

for mode in [0, 1]:
    if mode == 0:
        dataType = 'val2017'
        annFile = '/mnt/hdd10tb/Datasets/COCO2017/annotations/%s_%s.json' %(annType, dataType)
        coco_kpt = []
    else:
        dataType = 'train2017'
        annFile = '/mnt/hdd10tb/Datasets/COCO2017/annotations/%s_%s.json' %(annType, dataType)
        coco_kpt = []

    coco = COCO(annFile)

    list_images = coco.getImgIds()
    for id in list_images:
        data = dict()
        data['image_id'] = id
        annos_list = coco.getAnnIds(id)
        if len(annos_list) == 0:
            continue
        annotations = []
        for annos in annos_list:
            t_data = coco.loadAnns(annos)[0]
            t_annos = dict()
            t_annos['bbox'] = t_data['bbox']
            t_annos['segmentation'] = t_data['segmentation']
            t_annos['area'] = t_data['area']
            t_annos['id'] = t_data['id']
            t_annos['iscrowd'] = t_data['iscrowd']
            t_annos['keypoints'] = t_data['keypoints']
            t_annos['num_keypoints'] = t_data['num_keypoints']
            t_annos['img_width'] = coco.loadImgs(id)[0]['width']
            t_annos['img_height'] = coco.loadImgs(id)[0]['height']

            annotations.append(t_annos)
        data['annorect'] = annotations
        coco_kpt.append(data)
    name = 'coco_val.json' if mode == 0 else 'coco_kpt.json'
    with open(name, 'w') as f:
        json.dump(coco_kpt, f)
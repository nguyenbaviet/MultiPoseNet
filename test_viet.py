from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os, sys
sys.path.append(os.getcwd())

coco_val = '/mnt/hdd10tb/Datasets/COCO2017/annotations/person_keypoints_val2017.json'
coco_pred = '/home/vietnguyen/MultiPoseNet/demo/multipose_coco2017_results.json'

coco = COCO(coco_val)
coco_pred = coco.loadRes(coco_pred)
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


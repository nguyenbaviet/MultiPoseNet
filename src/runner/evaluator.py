import os
from collections import OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from utils.lib.utils.log import logger
from utils.lib.utils import meter as meter_utils
import numpy as np
from src.runner.base import Base
import cv2
from src.utils.joint_utils import get_joint_list
import json

class Evaluator(Base):
    def coco_eval(self):

        coco_val = os.path.join(self.params.coco_root, 'annotations/person_keypoints_val2017.json')
        coco = COCO(coco_val)
        img_ids = coco.getImgIds(catIds=[1])
        multipose_results = []
        coco_order = [0, 14, 13, 16, 15, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]

        for img_id in tqdm(img_ids):

            img_name = coco.loadImgs(img_id)[0]['file_name']

            oriImg = cv2.imread(os.path.join(self.params.coco_root, 'images/val2017/', img_name)).astype(np.float32)
            multiplier = self._get_multiplier(oriImg)

            # Get results of original image
            orig_heat, orig_bbox_all = self._get_outputs(multiplier, oriImg)

            # Get results of flipped image
            swapped_img = oriImg[:, ::-1, :]
            flipped_heat, flipped_bbox_all = self._get_outputs(multiplier, swapped_img)

            # compute averaged heatmap
            heatmaps = self._handle_heat(orig_heat, flipped_heat)

            # segment_map = heatmaps[:, :, 17]
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            joint_list = get_joint_list(oriImg, param, heatmaps[:, :, :18], 1)
            joint_list = joint_list.tolist()

            joints = []
            for joint in joint_list:
                if int(joint[-1]) != 1:
                    joint[-1] = max(0, int(joint[-1]) - 1)
                    joints.append(joint)
            joint_list = joints

            prn_result = self.prn_process(joint_list, orig_bbox_all[1], img_name, img_id)
            for result in prn_result:
                keypoints = result['keypoints']
                coco_keypoint = []
                for i in range(17):
                    coco_keypoint.append(keypoints[coco_order[i] * 3])
                    coco_keypoint.append(keypoints[coco_order[i] * 3 + 1])
                    coco_keypoint.append(keypoints[coco_order[i] * 3 + 2])
                result['keypoints'] = coco_keypoint
                multipose_results.append(result)

        ann_filename = self.params.coco_result_filename
        with open(ann_filename, "w") as f:
            json.dump(multipose_results, f, indent=4)
        # load results in COCO evaluation tool
        coco_pred = coco.loadRes(ann_filename)
        # run COCO evaluation
        if self.params.subnet_name == 'detection':
            coco_eval = COCOeval(coco, coco_pred, 'bbox')
        elif self.params.subnet_name == 'keypoints':
            coco_eval = COCOeval(coco, coco_pred, 'keypoints')
        else:
            raise ValueError(
                'not support to compute accuracy for %s. Expected subnet name in [detection, keypoints]' % self.params.subnet_name)
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # if not self.params.testresult_write_json:
        #     os.remove(ann_filename)

    def val(self):
        self.model.eval()
        logs = OrderedDict()
        sum_loss = meter_utils.AverageValueMeter()
        logger.info('Val on validation set...')

        self.batch_timer.clear()
        self.data_timer.clear()
        self.batch_timer.tic()
        self.data_timer.tic()
        for step, batch in enumerate(self.val_data):
            self.data_timer.toc()

            inputs, gts, _ = self.batch_processor(self, batch)
            _, saved_for_loss = self.model(*inputs)
            self.batch_timer.toc()

            loss, saved_for_log = self.model.module.build_loss(saved_for_loss, *gts)
            sum_loss.add(loss.item())
            self._process_log(saved_for_log, logs)

            if step % self.params.print_freq == 0:
                self._print_log(step, logs, 'Validation', max_n_batch=len(self.val_data))

            self.data_timer.tic()
            self.batch_timer.tic()

        mean, std = sum_loss.value()
        logger.info('\n\nValidation loss: mean: {}, std: {}'.format(mean, std))
import sys, os
sys.path.append(os.getcwd())
import json
import numpy as np


count = 1
makeFigure = 0
validationCount = 0
isValidation = 0

dataset = []
for mode in [0, 1]:
    name = 'coco_kpt.json' if mode == 0 else 'coco_val.json'
    with open(name) as f:
        data = json.load(f)
    for i in range(len(data)):
        print(i)
        numPeople = len(data[i]['annorect'])

        prev_center = []

        if mode == 1:
            if i < 2645:
                validationCount += 1
                isValidation = 1
            else:
                isValidation = 0
        else:
            isValidation = 0

        img_id = data[i]['image_id']
        for p in range(numPeople):
            temp = dict()
            annorect = data[i]['annorect'][p]

            h = annorect['img_height']
            w = annorect['img_width']
            if annorect['num_keypoints'] < 5 or annorect['area'] < 32* 32:
                continue
            person_center = [annorect['bbox'][0] + annorect['bbox'][2] / 2, annorect['bbox'][1] + annorect['bbox'][3] / 2]
            flag = 0
            for k in range(len(prev_center)):
                dist = [prev_center[k][0][0] - person_center[0], prev_center[k][0][1] - person_center[1]]
                if np.linalg.norm(dist) < prev_center[k][1] * 0.3:
                    flag = 1
                    continue
            if flag == 1:
                continue
            temp['dataset'] = 'COCO' if mode == 0 else 'COCO_val'
            temp['isValidation'] = isValidation
            annos = annorect['keypoints']

            temp['img_paths'] = 'train2017/%012d.jpg' %img_id if mode == 0 else 'val2017/%012d.jpg' %img_id
            temp['img_width'] = w
            temp['img_height'] = h
            temp['objpos'] = person_center
            temp['image_id'] = img_id
            temp['bbox'] = annorect['bbox']
            temp['segment_area'] = annorect['area']
            temp['num_keypoints'] = annorect['num_keypoints']

            joint_self = []
            for part in range(17):
                if annos[3*part + 2] == 2:
                    t_vis = 1
                elif annos[3*part + 2] == 1:
                    t_vis = 0
                else:
                    t_vis = 2
                joint_self.append([annos[3 * part], annos[3 * part + 1], t_vis])
            temp['joint_self'] = joint_self
            temp['scale_provided'] = annorect['bbox'][3] / 368

            joint_others = []
            scale_provided_other = []
            objpos_other = []
            bbox_other = []
            segment_area_other = []
            num_keypoints_other = []
            for op in range(numPeople):
                if op == p or data[i]['annorect'][op]['num_keypoints'] == 0:
                    continue

                t_anno = data[i]['annorect'][op]['keypoints']
                scale_provided_other.append(data[i]['annorect'][op]['bbox'][3] / 368)
                objpos_other.append([data[i]['annorect'][op]['bbox'][0] + data[i]['annorect'][op]['bbox'][2] / 2,
                                     data[i]['annorect'][op]['bbox'][1] + data[i]['annorect'][op]['bbox'][3] / 2])
                bbox_other.append(data[i]['annorect'][op]['bbox'])
                segment_area_other.append(data[i]['annorect'][op]['area'])
                num_keypoints_other.append(data[i]['annorect'][op]['num_keypoints'])
                t_kp = []
                for part in range(17):
                    if t_anno[3*part + 2] == 2:
                        t_vis = 1
                    elif t_anno[3*part + 2] == 1:
                        t_vis = 0
                    else:
                        t_vis = 2
                    t_kp.append([t_anno[3*part], t_anno[3*part + 1], t_vis])
                joint_others.append(t_kp)

            temp['scale_provided_other'] = scale_provided_other
            temp['objpos_other'] = objpos_other
            temp['bbox_other'] = bbox_other
            temp['segment_area_other'] = segment_area_other
            temp['num_keypoints_other'] = num_keypoints_other
            temp['joint_others'] = joint_others

            temp['annolist_index'] = i
            temp['people_index'] = p
            temp['numOtherPeople'] = len(joint_others)
            dataset.append(temp)

            prev_center.append([temp['objpos'], max(annorect['bbox'][2], annorect['bbox'][3])])

with open('COCO.json', 'w') as f:
    json.dump(dataset, f)
                
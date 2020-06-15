import sys, os
sys.path.append(os.getcwd())

import threading
import json
import cv2
import numpy as np
from matplotlib import path
from pycocotools import mask
import time
import math

with open('coco_kpt.json') as f:
    data = json.load(f)
len_data = int(len(data) / 12100) + 1
def processing(data, start_index):
    for mode in [0, 1]:
        name = 'coco_val.json' if mode == 0 else 'coco_kpt.json'

        # with open(name) as f:
        #     data = json.load(f)
        data = data[start_index:start_index + 12100]

        L = len(data)
        for i in range(L):
            print(i)
            img_id = data[i]['image_id']
            if mode == 0:
                img_paths = '/mnt/hdd10tb/Datasets/COCO2017/images/val2017/%012d.jpg' % img_id
                img_name1 = '/mnt/hdd10tb/Datasets/COCO2017/viet_data/mask2017/val2017_mask_all_%012d.png' % img_id
                img_name2 = '/mnt/hdd10tb/Datasets/COCO2017/viet_data/mask2017/val2017_mask_miss_%012d.png' % img_id
            else:
                img_paths = '/mnt/hdd10tb/Datasets/COCO2017/images/train2017/%012d.jpg' % img_id
                img_name1 = '/mnt/hdd10tb/Datasets/COCO2017/viet_data/mask2017/train2017_mask_all_%012d.png' % img_id
                img_name2 = '/mnt/hdd10tb/Datasets/COCO2017/viet_data/mask2017/train2017_mask_miss_%012d.png' % img_id
            if os.path.isfile(img_name2):
                continue
            img = cv2.imread(img_paths)
            h, w, _ = img.shape
            mask_all = np.zeros((h, w), dtype=int)
            mask_miss = np.zeros((h, w), dtype=int)
            flag = 0
            start = time.time()
            for p in range(len(data[i]['annorect'])):
                try:
                    seg = data[i]['annorect'][p]['segmentation'][0]
                except:
                    mask_crowd = data[i]['annorect'][p]['segmentation']
                    compressed_rle = mask.frPyObjects(mask_crowd, mask_crowd.get('size')[0], mask_crowd.get('size')[1])
                    mask_crowd = mask.decode(compressed_rle)
                    mask_crowd -= np.array(np.logical_and(mask_all, mask_crowd))
                    flag += 1
                    # data[i]['mask_crowd'] = mask_crowd
                    continue
                pth = path.Path([(seg[0::2][index], seg[1::2][index]) for index in range(len(seg[0::2]))])
                _mask = np.zeros((h, w), dtype=int)
                list_check = []
                p_bbox = data[i]['annorect'][p]['bbox']
                for i_h in range(math.floor(p_bbox[1]), math.ceil(p_bbox[1] + p_bbox[3])):
                    for i_w in range(math.floor(p_bbox[0]), math.ceil(p_bbox[0] + p_bbox[2] )):
                        list_check.append((i_w + 1, i_h + 1))
                p_check = pth.contains_points(list_check)
                p_ind = 0
                for i_h in range(math.floor(p_bbox[1]) , math.ceil(p_bbox[1] + p_bbox[3])):
                    for i_w in range(math.floor(p_bbox[0]), math.ceil(p_bbox[0] + p_bbox[2])):
                        _mask[i_h][i_w] = 1 if p_check[p_ind] else 0
                        p_ind += 1
                # print('Time1: ', time.time() - start)
                # print(np.sum(_mask))
                # _mask = np.zeros((h, w), dtype=int)
                # start = time.time()
                # for i_h in range(h):
                #     for i_w in range(w):
                #         if pth.contains_points([(i_w + 1, i_h + 1)])[0]:
                #             _mask[i_h][i_w] = 1
                # print('Time2: ', time.time() -start)
                # print(np.sum(_mask))
                # jkasdf
                mask_all = np.array(np.logical_or(_mask, mask_all), dtype=int)

                if data[i]['annorect'][p]['num_keypoints'] <= 0:
                    mask_miss = np.array(np.logical_or(_mask, mask_miss), dtype=int)
                if flag == 1:
                    mask_miss = np.array(np.logical_not(np.logical_or(mask_miss, mask_crowd)), dtype=int)
                    mask_all = np.array(np.logical_or(mask_all, mask_crowd))
                else:
                    mask_miss = np.array(np.logical_not(mask_miss), dtype=int)
            print('Time: ', time.time() - start)
            i_mask_miss = np.array([mask_miss, mask_miss, mask_miss]).transpose(1, 2, 0)
            # i_mask_all = np.array([mask_all, mask_all, mask_all]).transpose(1, 2, 0)
            # cv2.imwrite(img_name1, i_mask_all)
            cv2.imwrite(img_name2, i_mask_miss)

thread = [threading.Thread(target=processing, args=(data, 12100*i)) for i in range(len_data)]

for t in thread:
    t.start()
for t in thread:
    t.join()





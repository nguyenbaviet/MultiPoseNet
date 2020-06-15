from __future__ import print_function

import cv2
import math
import datetime

import torch
import torch.nn as nn
from utils.lib.utils.log import logger
from utils.lib.utils import meter as meter_utils
import src.utils.net_utils as net_utils
from utils.lib.utils.timer import Timer
from src.datasets.utils.preprocessing import resnet_preprocess
from src.datasets.utils.prn_gaussian import crop
from src.utils.utils import TestParams

from skimage.filters import gaussian


import numpy as np


def _factor_closest(num, factor, is_ceil=True):
    """Returns the closest integer to `num` that is divisible by `factor`

    Actually, that's a lie. By default, we return the closest factor _greater_
    than the input. If, however, you set `it_ceil` to `False`, we return the
    closest factor _less_ than the input.
    """
    num = float(num) / factor
    num = np.ceil(num) if is_ceil else np.floor(num)
    return int(num) * factor


def crop_with_factor(im, dest_size, factor=32, pad_val=0, basedon='min'):
    """Scale and pad an image to the desired size and divisibility

    Scale the specified dimension of the input image to `dest_size` then pad
    the result until it is cleanly divisible by `factor`.

    Args:
        im (Image): The input image.
        dest_size (int): The desired size of the unpadded, scaled image's
            dimension specified by `basedon`.
        factor (int): Pad the scaled image until it is factorable
        pad_val (number): Value to pad with.
        basedon (string): Specifies which dimension to base the scaling on.
            Valid inputs are 'min', 'max', 'w', and 'h'. Defaults to 'min'.

    Returns:
        A tuple of three elements:
            - The scaled and padded image.
            - The scaling factor.
            - The size of the non-padded part of the resulting image.
    """
    # Compute the scaling factor.
    im_size_min, im_size_max = np.min(im.shape[0:2]), np.max(im.shape[0:2])
    im_base = {'min': im_size_min,
               'max': im_size_max,
               'w': im.shape[1],
               'h': im.shape[0]}
    im_scale = float(dest_size) / im_base.get(basedon, im_size_min)

    # Scale the image.
    im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

    # Compute the padded image shape. Ensure it's divisible by factor.
    h, w = im.shape[:2]
    new_h, new_w = _factor_closest(h, factor), _factor_closest(w, factor)
    new_shape = [new_h, new_w] if im.ndim < 3 else [new_h, new_w, im.shape[-1]]

    # Pad the image.
    im_padded = np.full(new_shape, fill_value=pad_val, dtype=im.dtype)
    im_padded[0:h, 0:w] = im

    return im_padded, im_scale, im.shape

class Base(object):

    TestParams = TestParams

    def __init__(self, model, params, batch_processor=None, val_data=None):
        assert isinstance(params, TestParams)
        self.params = params
        self.batch_timer = Timer()
        self.data_timer = Timer()
        self.val_data = val_data if val_data else None
        self.batch_processor = batch_processor if batch_processor else None

        # load convert
        self.model = model
        ckpt = self.params.ckpt

        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.eval()
        self.model.module.freeze_bn()

    def _get_multiplier(self, img):
        """Computes the sizes of image at different scales
        :param img: numpy array, the current image
        :returns : list of float. The computed scales
        """
        scale_search = [0.5, 1., 1.5, 2, 2.5]
        return [x * self.params.inp_size / float(img.shape[0]) for x in scale_search]

    def _get_outputs(self, multiplier, img):
        """Computes the averaged heatmap and paf for the given image
        :param multiplier:
        :param origImg: numpy array, the image being processed
        :param convert: pytorch convert
        :returns: numpy arrays, the averaged paf and heatmap
        """

        heatmap_avg = np.zeros((img.shape[0], img.shape[1], 18))
        bbox_all = []

        for m in range(len(multiplier)):
            scale = multiplier[m]
            inp_size = scale * img.shape[0]

            # padding
            im_cropped, im_scale, real_shape = crop_with_factor(
                img, inp_size, factor=32, pad_val=128)
            im_data = resnet_preprocess(im_cropped)

            im_data = np.expand_dims(im_data, 0)
            with torch.no_grad():
                im_data = torch.from_numpy(im_data).type(torch.FloatTensor).cuda(device=self.params.gpus[0])

            # heatmaps, [scores, classification, transformed_anchors] = self.convert([im_data, self.params.subnet_name])
            heatmaps, [scores, classification, transformed_anchors] = self.model.module._predict_bboxANDkeypoints(im_data)
            heatmaps = heatmaps.cpu().detach().numpy().transpose(0, 2, 3, 1)
            scores = scores.cpu().detach().numpy()
            classification = classification.cpu().detach().numpy()
            transformed_anchors = transformed_anchors.cpu().detach().numpy()

            heatmap = heatmaps[0, :int(im_cropped.shape[0] / 4), :int(im_cropped.shape[1] / 4), :]
            heatmap = cv2.resize(heatmap, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[0:real_shape[0], 0:real_shape[1], :]
            heatmap = cv2.resize(
                heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg = heatmap_avg + heatmap / len(multiplier)

            # bboxs
            idxs = np.where(scores > 0.5)
            bboxs=[]
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]/im_scale
                if int(classification[idxs[0][j]]) == 0:  # class0=people
                    bboxs.append(bbox.tolist())
            bbox_all.append(bboxs)

        return heatmap_avg, bbox_all

    def _handle_heat(self, normal_heat, flipped_heat):
        """Compute the average of normal and flipped heatmap
        :param normal_heat: numpy array, the normal heatmap
        :param flipped_heat: numpy array, the flipped heatmap
        :returns: numpy arrays, the averaged heatmap
        """

        # The order to swap left and right of heatmap
        swap_heat = np.array((0, 1, 5, 6, 7, 2, 3, 4, 11, 12,
                              13, 8, 9, 10, 15, 14, 17, 16))#, 18

        averaged_heatmap = (normal_heat + flipped_heat[:, ::-1, :][:, :, swap_heat]) / 2.

        return averaged_heatmap

    def prn_process(self, kps, bbox_list, file_name, image_id=0):

        prn_result = []

        idx = 0
        ks = []
        for j in range(17):  # joint type
            t = []
            for k in kps:
                if k[-1] == j:  # joint type
                    x = k[0]
                    y = k[1]
                    v = 1  # k[2]
                    if v > 0:
                        t.append([x, y, 1, idx])
                        idx += 1
            ks.append(t)
        peaks = ks

        w = int(18 * self.params.coeff)
        h = int(28 * self.params.coeff)

        bboxes = []
        for bbox_item in bbox_list:
            bboxes.append([bbox_item[0], bbox_item[1], bbox_item[2]-bbox_item[0], bbox_item[3]-bbox_item[1]])

        if len(bboxes) == 0 or len(peaks) == 0:
            return prn_result

        weights_bbox = np.zeros((len(bboxes), h, w, 4, 17))

        for joint_id, peak in enumerate(peaks):  # joint_id: which joint
            for instance_id, instance in enumerate(peak):  # instance_id: which people
                p_x = instance[0]
                p_y = instance[1]
                for bbox_id, b in enumerate(bboxes):
                    is_inside = p_x > b[0] - b[2] * self.params.in_thres and \
                                p_y > b[1] - b[3] * self.params.in_thres and \
                                p_x < b[0] + b[2] * (1.0 + self.params.in_thres) and \
                                p_y < b[1] + b[3] * (1.0 + self.params.in_thres)
                    if is_inside:
                        x_scale = float(w) / math.ceil(b[2])
                        y_scale = float(h) / math.ceil(b[3])
                        x0 = int((p_x - b[0]) * x_scale)
                        y0 = int((p_y - b[1]) * y_scale)
                        if x0 >= w and y0 >= h:
                            x0 = w - 1
                            y0 = h - 1
                        elif x0 >= w:
                            x0 = w - 1
                        elif y0 >= h:
                            y0 = h - 1
                        elif x0 < 0 and y0 < 0:
                            x0 = 0
                            y0 = 0
                        elif x0 < 0:
                            x0 = 0
                        elif y0 < 0:
                            y0 = 0
                        p = 1e-9
                        weights_bbox[bbox_id, y0, x0, :, joint_id] = [1, instance[2], instance[3], p]
        old_weights_bbox = np.copy(weights_bbox)

        for j in range(weights_bbox.shape[0]):
            for t in range(17):
                weights_bbox[j, :, :, 0, t] = gaussian(weights_bbox[j, :, :, 0, t])

        output_bbox = []
        for j in range(weights_bbox.shape[0]):
            inp = weights_bbox[j, :, :, 0, :]
            input = torch.from_numpy(np.expand_dims(inp, axis=0)).cuda().float()
            # output, _ = self.convert([input, 'prn_subnet'])
            output = self.model.module._predict_prn(input)
            temp = np.reshape(output.data.cpu().numpy(), (56, 36, 17))
            output_bbox.append(temp)

        output_bbox = np.array(output_bbox)

        keypoints_score = []

        for t in range(17):
            indexes = np.argwhere(old_weights_bbox[:, :, :, 0, t] == 1)
            keypoint = []
            for i in indexes:
                cr = crop(output_bbox[i[0], :, :, t], (i[1], i[2]), N=15)
                score = np.sum(cr)

                kp_id = old_weights_bbox[i[0], i[1], i[2], 2, t]
                kp_score = old_weights_bbox[i[0], i[1], i[2], 1, t]
                p_score = old_weights_bbox[i[0], i[1], i[2], 3, t]  ## ??
                bbox_id = i[0]

                score = kp_score * score

                s = [kp_id, bbox_id, kp_score, score]

                keypoint.append(s)
            keypoints_score.append(keypoint)

        bbox_keypoints = np.zeros((weights_bbox.shape[0], 17, 3))
        bbox_ids = np.arange(len(bboxes)).tolist()

        # kp_id, bbox_id, kp_score, my_score
        for i in range(17):
            joint_keypoints = keypoints_score[i]
            if len(joint_keypoints) > 0:  # if have output result in one type keypoint

                kp_ids = list(set([x[0] for x in joint_keypoints]))

                table = np.zeros((len(bbox_ids), len(kp_ids), 4))

                for b_id, bbox in enumerate(bbox_ids):
                    for k_id, kp in enumerate(kp_ids):
                        own = [x for x in joint_keypoints if x[0] == kp and x[1] == bbox]

                        if len(own) > 0:
                            table[bbox, k_id] = own[0]
                        else:
                            table[bbox, k_id] = [0] * 4

                for b_id, bbox in enumerate(bbox_ids):  # all bbx, from 0 to ...

                    row = np.argsort(-table[bbox, :, 3])  # in bbx(bbox), sort from big to small, keypoint score

                    if table[bbox, row[0], 3] > 0:  # score
                        for r in row:  # all keypoints
                            if table[bbox, r, 3] > 0:
                                column = np.argsort(
                                    -table[:, r, 3])  # sort all keypoints r, from big to small, bbx score

                                if bbox == column[0]:  # best bbx. best keypoint
                                    bbox_keypoints[bbox, i, :] = [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][
                                        0]
                                    break
                                else:  # for bbx column[0], the worst keypoint is row2[0],
                                    row2 = np.argsort(table[column[0], :, 3])
                                    if row2[0] == r:
                                        bbox_keypoints[bbox, i, :] = \
                                            [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][0]
                                        break
            else:  # len(joint_keypoints) == 0:
                for j in range(weights_bbox.shape[0]):
                    b = bboxes[j]
                    x_scale = float(w) / math.ceil(b[2])
                    y_scale = float(h) / math.ceil(b[3])

                    for t in range(17):
                        indexes = np.argwhere(old_weights_bbox[j, :, :, 0, t] == 1)
                        if len(indexes) == 0:
                            max_index = np.argwhere(output_bbox[j, :, :, t] == np.max(output_bbox[j, :, :, t]))
                            bbox_keypoints[j, t, :] = [max_index[0][1] / x_scale + b[0],
                                                       max_index[0][0] / y_scale + b[1], 0]

        my_keypoints = []

        for i in range(bbox_keypoints.shape[0]):
            k = np.zeros(51)
            k[0::3] = bbox_keypoints[i, :, 0]
            k[1::3] = bbox_keypoints[i, :, 1]
            k[2::3] = bbox_keypoints[i, :, 2]

            pose_score = 0
            count = 0
            for f in range(17):
                if bbox_keypoints[i, f, 0] != 0 and bbox_keypoints[i, f, 1] != 0:
                    count += 1
                pose_score += bbox_keypoints[i, f, 2]
            pose_score /= 17.0

            my_keypoints.append(k)

            image_data = {
                'image_id': image_id,
                'file_name': file_name,
                'category_id': 1,
                'bbox': bboxes[i],
                'score': pose_score,
                'keypoints': k.tolist()
            }
            prn_result.append(image_data)

        return prn_result

    def _load_ckpt(self, ckpt):
        _, _ = net_utils.load_net(ckpt, self.model, load_state_dict=True)

    def _process_log(self, src_dict, dest_dict):
        for k, v in src_dict.items():
            if isinstance(v, (int, float)):
                dest_dict.setdefault(k, meter_utils.AverageValueMeter())
                dest_dict[k].add(float(v))
            else:
                dest_dict[k] = v

    def _print_log(self, step, log_values, title='', max_n_batch=None):
        log_str = '{}\n'.format(self.params.exp_name)
        log_str += '{}: epoch {}'.format(title, 0)

        log_str += '[{}/{}]'.format(step, max_n_batch)

        i = 0
        for k, v in log_values.items():
            if isinstance(v, meter_utils.AverageValueMeter):
                mean, std = v.value()
                log_str += '\n\t{}: {:.10f}'.format(k, mean)
                i += 1

        if max_n_batch:
            # print time
            data_time = self.data_timer.duration + 1e-6
            batch_time = self.batch_timer.duration + 1e-6
            rest_seconds = int((max_n_batch - step) * batch_time)
            log_str += '\n\t({:.2f}/{:.2f}s,' \
                       ' fps:{:.1f}, rest: {})'.format(data_time, batch_time,
                                                       self.params.batch_size / batch_time,
                                                       str(datetime.timedelta(seconds=rest_seconds)))
            self.batch_timer.clear()
            self.data_timer.clear()

        logger.info(log_str)

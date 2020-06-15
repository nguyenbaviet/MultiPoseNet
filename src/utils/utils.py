import torch
import torch.nn as nn
import numpy as np
import sys

from collections import OrderedDict
from utils.lib.utils.log import logger

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes

def frozen_weights(model, cfg):
    modules = cfg['FROZEN_MODULES']
    if not isinstance(modules, list):
        modules = [modules]
    for module in modules:
        if module == 'backbone':
            for name, m in model.fpn.named_children():
                if name in cfg['FROZEN_WEIGHTS_LIST']['backbone']:
                    for para in m.parameters():
                        para.requires_grad = False
        elif module == 'detection':
            for name, m in model.fpn.named_children():
                if name in cfg['FROZEN_WEIGHTS_LIST']['detection']['backbone']:
                    for para in m.parameters():
                        para.requires_grad = False
            for name, m in model.named_children():
                if name in cfg['FROZEN_WEIGHTS_LIST']['detection']['detection']:
                    for para in m.parameters():
                        para.requires_grad = False
        elif module == 'keypoints':
            for name, m in model.fpn.named_children():
                if name in cfg['FROZEN_WEIGHTS_LIST']['keypoints']['backbone']:
                    for para in m.parameters():
                        para.requires_grad = False
            for name, m in model.named_children():
                if name in cfg['FROZEN_WEIGHTS_LIST']['keypoints']['keypoints']:
                    for para in m.parameters():
                        para.requires_grad = False
        elif module == 'prn':
            for name, m in model.named_children():
                if name in cfg['FROZEN_WEIGHTS_LIST']['prn']:
                    for para in m.parameters():
                        para.requires_grad = False
        else:
            raise ValueError('No modules name %s. Expected module in [backbone, detection, keypoints, prn]' %module)
    return model
class TrainParams(object):
    # required params
    exp_name = 'experiment_name'
    subnet_name = 'keypoint_subnet'
    batch_size = 32
    max_epoch = 30
    optimizer = None

    # learning rate scheduler
    lr_scheduler = None         # should be an instance of ReduceLROnPlateau or _LRScheduler
    max_grad_norm = np.inf

    # params based on your local env
    gpus = [1]
    save_dir = None             # default `save_dir` is `outputs/{exp_name}`

    # loading existing checkpoint
    ckpt = None                 # path to the ckpt file, will load the last ckpt in the `save_dir` if `None`
    re_init = False             # ignore ckpt if `True`
    zero_epoch = False          # force `last_epoch` to zero
    ignore_opt_state = False    # ignore the saved optimizer states

    # saving checkpoints
    save_freq_epoch = 1             # save one ckpt per `save_freq_epoch` epochs
    save_freq_step = sys.maxsize    # save one ckpt per `save_freq_setp` steps, default value is inf
    save_nckpt_max = sys.maxsize    # max number of saved ckpts

    # validation during training
    val_freq = 500              # run validation per `val_freq` steps
    val_nbatch = 10             # number of batches to be validated
    val_nbatch_end_epoch = 200  # max number of batches to be validated after each epoch

    # visualization
    print_freq = 20             # print log per `print_freq` steps
    use_tensorboard = False     # use tensorboardX if True
    visualization_fn = None     # custom function to handle `log_dict`, default value is `default_visualization_fn`

    def update(self, params_dict):
        state_dict = self.state_dict()
        for k, v in params_dict.items():
            if k in state_dict or hasattr(self, k):
                setattr(self, k, v)
            else:
                logger.warning('Unknown option: {}: {}'.format(k, v))

    def state_dict(self):
        state_dict = OrderedDict()
        for k in TrainParams.__dict__.keys():
            if not k.startswith('_'):
                state_dict[k] = getattr(self, k)
        del state_dict['update']
        del state_dict['state_dict']

        return state_dict

    def __str__(self):
        state_dict = self.state_dict()
        text = 'TrainParams {\n'
        for k, v in state_dict.items():
            text += '\t{}: {}\n'.format(k, v)
        text += '}\n'
        return text
def init_train_params(cfg):
    params = TrainParams()

    params.gpus = cfg['COMMON']['GPUS']

    params.subnet_name = cfg['MODEL']['subnet']
    params.max_epoch = cfg['TRAIN']['num_epochs']
    params.batch_size = cfg['TRAIN']['batch_size'] * len(params.gpus)
    params.init_lr = cfg['TRAIN']['init_lr']
    params.lr_decay = cfg['TRAIN']['lr_decay']

    params.save_dir = cfg['COMMON']['saved_dir']
    params.ckpt = cfg['TRAIN']['ckpt']
    params.re_init = cfg['TRAIN']['re_init']
    params.ignore_opt_state = cfg['TRAIN']['ignore_opt_state']
    params.val_nbatch_end_epoch = cfg['TRAIN']['val_nbatch_end_epoch']
    params.print_freq = cfg['TRAIN']['print_freq']
    params.use_tensorboard = cfg['TRAIN']['use_tensorboard']

    return params


class TestParams(object):

    trunk = 'resnet101'  # select the convert
    coeff = 2
    in_thres = 0.21

    testdata_dir = './demo/test_videos/'
    testresult_dir = './demo/output/'
    testresult_write_image = True  # write image results or not
    testresult_write_json = False  # write json results or not
    gpus = [0]
    ckpt = './demo/models/ckpt_baseline_resnet101.h5'  # checkpoint file to load, no need to change this
    coco_root = 'coco_root/'
    coco_result_filename = './extra/multipose_coco2017_results.json'

    # # required params
    inp_size = 480  # input size 480*480
    exp_name = 'multipose101'
    subnet_name = 'keypoints'
    batch_size = 32
    print_freq = 20
def init_Test_params(cfg):
    params = TestParams()
    params.gpus = cfg['COMMON']['GPUS']
    params.ckpt = cfg['VAL']['ckpt']
    params.coco_root = cfg['DATASET']['coco_root']

    if cfg['VAL']['dataset_type'] == 'val':
        params.subnet_name = cfg['MODEL']['subnet']
        params.batch_size = cfg['VAL']['batch_size']
    else:
        params.output_path = cfg['VAL']['output_path']
        params.video_path = cfg['VAL']['video_path']
        params.coco_result_filename = cfg['VAL']['json_result_path']
    return params
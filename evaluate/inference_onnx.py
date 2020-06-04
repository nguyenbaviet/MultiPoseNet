import os, sys
sys.path.append(os.getcwd())
from __future__ import print_function

import os
import cv2
import math
import datetime
import numpy as np
import json
from collections import OrderedDict
from network.joint_utils import get_joint_list, plot_result
from tqdm import tqdm

import torch
import torch.nn as nn
from lib.utils.log import logger
import lib.utils.meter as meter_utils
import network.net_utils as net_utils
from lib.utils.timer import Timer
from datasets.coco_data.preprocessing import resnet_preprocess
from datasets.coco_data.prn_gaussian import gaussian, crop

class Inference(object):
    def __init__(self, kp_bbox_model, prn):
        pass
    self.kp_bbox_model = kp_bbox_model
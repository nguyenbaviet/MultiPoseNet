import os, sys
root_path = os.path.realpath(__file__).split('/evaluate/multipose_coco_eval.py')[0]
os.chdir(root_path)
sys.path.append(root_path)

from network.posenet import poseNet
from evaluate.tester import Tester

backbone = 'resnet101'

# Set Training parameters
params = Tester.TestParams()
params.subnet_name = 'both'
params.inp_size = 480  # input picture size = (inp_size, inp_size)
params.coeff = 2
params.in_thres = 0.21
params.coco_root = '/mnt/hdd10tb/Datasets/COCO2017/'
params.testresult_write_json = True  # Whether to write json result
params.coco_result_filename = './demo/multipose_coco2017_results.json'
params.ckpt = '/home/vietnguyen/MultiPoseNet/extra/models/res50_detection_subnet/ckpt_39_0.59604.h5.best'

# model
model = poseNet(backbone)

for name, module in model.named_children():
    for para in module.parameters():
        para.requires_grad = False

tester = Tester(model, params)
tester.coco_eval()  # pic_test

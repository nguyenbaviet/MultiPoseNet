import os, sys
root_path = os.path.realpath(__file__).split('/evaluate/multipose_test.py')[0]
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
params.testdata_dir = './demo/test_images/'
params.testresult_dir = './demo/output/'
params.testresult_write_image = True  # Whether to write result pictures
params.testresult_write_json = False  # Whether to write json result
# params.ckpt = './demo/models/ckpt_baseline_resnet101.h5'
params.ckpt = '/home/vietnguyen/MultiPoseNet/extra/models/res50_detection_subnet/ckpt_39_0.59604.h5.best'
# params.gpus = [2]

# model
model = poseNet(backbone=backbone)

for name, module in model.named_children():
    for para in module.parameters():
        para.requires_grad = False

tester = Tester(model, params)
tester.test()  # pic_test

from __future__ import print_function

import cv2
from src.utils.joint_utils import get_joint_list, plot_result

import torch
from src.datasets.utils.preprocessing import resnet_preprocess

import numpy as np
from src.runner.base import Base


class Tester(Base):
    def test(self):
        import time
        # img_list = os.listdir(self.params.testdata_dir)
        multipose_results = []
        cap = cv2.VideoCapture(self.params.video_path)
        out = cv2.VideoWriter(self.params.output_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280, 720))
        id = 0
        start = time.time()
        while True:
            _, img = cap.read()
            if img is None:
                break
            id += 1
            shape_dst = np.max(img.shape)
            scale = float(shape_dst) / self.params.inp_size
            pad_size = np.abs(img.shape[1] - img.shape[0])
            img_resized = np.pad(img, ([0, pad_size], [0, pad_size], [0, 0]), 'constant')[:shape_dst, :shape_dst]
            img_resized = cv2.resize(img_resized, (self.params.inp_size, self.params.inp_size))
            img_input = resnet_preprocess(img_resized)
            img_input = torch.from_numpy(np.expand_dims(img_input, 0))

            with torch.no_grad():
                img_input = img_input.cuda()
            heatmaps, [scores, classification, transformed_anchors] = self.model.module._predict_bboxANDkeypoints(img_input)
            heatmaps = heatmaps.cpu().detach().numpy()
            heatmaps = np.squeeze(heatmaps, 0)
            heatmaps = np.transpose(heatmaps, (1, 2, 0))
            heatmap_max = np.max(heatmaps[:, :, :18], 2)
            # segment_map = heatmaps[:, :, 17]
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            joint_list = get_joint_list(img_resized, param, heatmaps[:, :, :18], scale)
            joint_list = joint_list.tolist()
            del img_resized

            joints = []
            for joint in joint_list:
                if int(joint[-1]) != 1:
                    joint[-1] = max(0, int(joint[-1]) - 1)
                    joints.append(joint)
            joint_list = joints

            # bounding box from retinanet
            scores = scores.cpu().detach().numpy()
            classification = classification.cpu().detach().numpy()
            transformed_anchors = transformed_anchors.cpu().detach().numpy()
            idxs = np.where(scores > 0.5)
            bboxs=[]
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]*scale
                if int(classification[idxs[0][j]]) == 0:  # class0=people
                    bboxs.append(bbox.tolist())
            print(len(bboxs))
            img_name = ''
            prn_result = self.prn_process(joint_list, bboxs, img_name)
            for result in prn_result:
                multipose_results.append(result)

            if self.params.testresult_write_image:
                canvas = plot_result(img, prn_result)
                # cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_1heatmap.png'), heatmap_max * 256)
                # cv2.imwrite(os.path.join(self.params.testresult_dir, img_name.split('.', 1)[0] + '_2canvas.png'), canvas)
                canvas = cv2.resize(canvas, (1280, 720))
                out.write(canvas)
        print(id)
        print('FPS: ', id / (time.time() - start))
        # if self.params.testresult_write_json:
        #     with open(self.params.testresult_dir+'multipose_results.json', "w") as f:
        #         json.dump(multipose_results, f)
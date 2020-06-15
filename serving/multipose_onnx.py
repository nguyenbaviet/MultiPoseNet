from openvino.inference_engine import IENetwork, IEPlugin, IECore
import cv2
import numpy as np
import time

class Inference:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dAk, self.input_dAk = self.init_extractor(cfg['detectionANDkeypoints']['xml'], cfg['detectionANDkeypoints']['bin'])
        self.prn, self.input_prn = self.init_extractor(cfg['prn']['xml'], cfg['prn']['bin'])
    def init_extractor(self, model_xml, model_bin):
        plugin_dir = None
        plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
        net = IENetwork.from_ir(model=model_xml, weights=model_bin)
        input_blob = next(iter(net.inputs))

        # Load network to the plugin
        exec_net = plugin.load(network=net)
        del net
        return exec_net, input_blob
    def infer(self, x):
        pass
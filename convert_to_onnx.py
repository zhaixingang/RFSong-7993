from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot
from data import AnnotationTransform,VOCDetection, BaseTransform, VOC_Config
from models.RFB_Net_vgg import build_net
import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
import cv2

parser = argparse.ArgumentParser(description='Receptive Field Block Net')
parser.add_argument('--img_dir', default='images', type=str,
                    help='Dir to save results')
parser.add_argument('-m', '--trained_model', default='weights/epoches_112.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--output_onnx', default='./onnx/RFSong-7993.onnx', type=str,
                    help='output_onnx')
parser.add_argument('--net_len', default=300, type=int,
                    help='net_len')
args = parser.parse_args()


cfg = VOC_Config
img_dim = 300
num_classes = 2
rgb_means = (104, 117, 123)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


class ObjectDetector:
    def __init__(self, net, detection, transform, num_classes=2, thresh=0.1, cuda=True):
        self.net = net
        self.detection = detection
        self.transform = transform
        self.num_classes = num_classes
        self.thresh = thresh
        self.cuda = cuda

    def predict(self, img):
        _t = {'im_detect': Timer(), 'misc': Timer()}
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

        with torch.no_grad():
            x = self.transform(img).unsqueeze(0)
            if self.cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x)  # forward pass
        boxes, scores = self.detection.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        # scale each detection back up to the image
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        _t['misc'].tic()
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            #print(scores[:, j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # keep = nms(c_bboxes,c_scores)

            keep = nms(c_dets, 0.2, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets

        nms_time = _t['misc'].toc()
        total_time = detect_time+nms_time

        #print('total time: ', total_time)
        return all_boxes, total_time

if __name__ == '__main__':
    # load net
    net = build_net('test', img_dim, num_classes)    # initialize detector
    state_dict = torch.load(args.trained_model)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    output_onnx = args.output_onnx
    device = torch.device("cpu")
    net = net.to(device)
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ["input"]
    output_names = []
    inputs = torch.randn(1, 3, args.net_len, args.net_len).to(device)

    torch_out = torch.onnx._export(net, inputs, output_onnx, opset_version=9, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names)

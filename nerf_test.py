import argparse
import copy
import json
import os
from pathlib import Path
from threading import Thread
import torch.nn.functional as F
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

from utils.CoordNerfUtils import *

def createPoses():
    T1 = torch.tensor(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, -5],
         [0, 0, 0, 1]]).float()
    T2 = torch.tensor(
        [[0, 0, 1, -5],
         [0, 1, 0, 0],
         [-1, 0, 0, 0],
         [0, 0, 0, 1]]).float()
    T3 = torch.tensor(
        [[-1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, -1, 5],
         [0, 0, 0, 1]]).float()
    T4 = torch.tensor(
        [[0, 0, -1, 5],
         [0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 1]]).float()
    return torch.stack([T1, T2, T3, T4]).to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='nerf_test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    imgsz = opt.img_size
    weights = opt.weights
    device = opt.device
    nviews = opt.batch_size

    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    model.eval()

    imgs = torch.randn(nviews, 3, imgsz//2, imgsz).to(device)
    depths = torch.randn(nviews, 1, imgsz//2, imgsz).to(device)**2

    poses = createPoses()

    feature = model(imgs, feature=True)

    st = time.time()

    sizes = [feat.shape[-2:] for feat in feature]

    nSize = len(sizes)

    hists = createDepthHistograms(depths, sizes)

    rays = generateRays(poses, sizes, imgsz)

    distances, depths = getAllRayIntersections(rays, poses)

    newFeature = [torch.zeros(feat.shape).to(device) for feat in feature]

    for i in range(nviews):
        key_indices = [n for n in range(nviews) if n != i]
        for j in range(nSize):
            img_q = feature[j][i]
            img_k = feature[j][key_indices]

            dists, depth_q, depth_k = getElements(distances[j], depths[j], i, key_indices)

            feat_sum = sumFeatures(feature[j], dists, i,  key_indices)
            feat_wsum = sumFeaturesDepthHist(feature[j], dists, i,  key_indices, hists[j], depth_k)
            feat_attsum = sumFeaturesDepthAttention(feature[j], dists, hists[j], depth_q, depth_k)
            newFeature[j][i] = feat_sum

    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    res = model.model[-1](newFeature)

    print(res)




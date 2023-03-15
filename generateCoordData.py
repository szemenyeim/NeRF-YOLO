import math
import cv2
import numpy as np
import torch
import tqdm
import os

from utils.CoordNerfUtils import *

def createRotMtx(yaw, device='cuda:0'):
    return torch.tensor(
        [[math.cos(yaw), -math.sin(yaw), 0],
         [0, math.cos(yaw), 0],
         [math.sin(yaw), 0, 1]]).to(device)

def createBasePoses(dist=10, device='cuda:0'):
    T1 = torch.tensor(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, -dist],
         [0, 0, 0, 1]]).float()
    T2 = torch.tensor(
        [[0, 0, 1, -dist],
         [0, 1, 0, 0],
         [-1, 0, 0, 0],
         [0, 0, 0, 1]]).float()
    T3 = torch.tensor(
        [[-1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, -1, dist],
         [0, 0, 0, 1]]).float()
    T4 = torch.tensor(
        [[0, 0, -1, dist],
         [0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 1]]).float()
    return torch.stack([T1, T2, T3, T4]).to(device)

def generateDepthImage(imgSize, max_dist = 40, nPolys = 200):

    depth = np.zeros((imgSize//2, imgSize)).astype('float32')

    for i in range(nPolys):
        pts = np.random.randint(0, imgSize, 2)
        pts[1] = pts[1]//2
        radius = np.random.randint(imgSize//128, imgSize//16, 1)[0]
        cv2.circle(depth, pts, radius, np.random.rand(1)*max_dist, -1)

    return depth

def generateDepths(nViews, imgSize, device='cuda:0'):

    depths = []

    for i in range(nViews):
        depths.append(torch.from_numpy(generateDepthImage(imgSize)))

    depths = torch.stack(depths).unsqueeze(1).to(device)

    return depths


def generatePoses(scale=1.0, rotScale = 0.1, device='cuda:0'):

    poses = createBasePoses(device=device)

    nviews = poses.shape[0]
    trans_offsets = torch.randn([nviews, 3]).to(device) * scale

    poses[:, 0:3, 3] += trans_offsets

    rots = torch.stack([createRotMtx(torch.randn(1)*rotScale, device=device) for _ in range(nviews)])
    poses[:, 0:3, 0:3] = torch.bmm(poses[:, 0:3, 0:3], rots)

    return poses

def assembleAttMTX(depths, distances, nviews, alpha):

    attns = []

    for depth, distance in zip(depths, distances):
        size = depth.shape[1]
        attn = torch.diag(torch.ones(nviews*size)*3)
        for i in range(nviews-1):
            for j in range(i+1, nviews):
                dists = distance[fromMatrixToVector(i,j,nviews-1)]
                '''depth_k = torch.round(depth[fromMatrixToVector(i,j,nviews), :, :, 1]/2).long()
                depth_q = torch.round(depth[fromMatrixToVector(i,j,nviews), :, :, 0]/2).long()
                hist_k = hist[j].view(-1, hist.shape[-1])
                hist_q = hist[i].view(-1, hist.shape[-1])
                hist_wk = hist_k[(torch.arange(hist_k.shape[0]).cuda(), depth_k)]
                hist_wq = hist_q[(torch.arange(hist_q.shape[0]).cuda(), depth_q)]'''

                weights = torch.exp(-alpha * dists)
                weights[weights < 0.1] = 0
                # Remove small weights
                attn[i*size:(i+1)*size, j*size:(j+1)*size] = weights
                attn[j*size:(j+1)*size, i*size:(i+1)*size] = weights.T

        attn = F.normalize(attn, dim=0)
        attns.append(attn)

    return attns

if __name__ == '__main__':

    nViews = 4
    baseImgSize = 640
    baseNumData = 640
    dataMults = [30, 10, 1, 2]

    split = "train/"
    scale = 1

    imgSize = baseImgSize*scale
    numData = baseNumData*dataMults[scale-1]
    if split != "train/":
        numData = numData // 4

    alpha = 75*scale

    outPath = "./nerfData/" + split + "out/" + str(imgSize) + "/"
    inPath = "./nerfData/" + split + "in/" + str(imgSize) + "/"

    os.makedirs(outPath, exist_ok=True)
    os.makedirs(inPath, exist_ok=True)

    sizes = [
        [imgSize//64, imgSize//32],
    ]

    for i in tqdm.tqdm(range(numData)):

        #depth_imgs = generateDepths(nViews, imgSize)
        poses = generatePoses()

        nSize = len(sizes)

        #hists = createDepthHistograms(depth_imgs, sizes)

        rays = generateRays(poses, sizes, imgSize)

        distances, depths = getAllRayIntersections(rays, poses)

        attns = assembleAttMTX(depths, distances, nViews, alpha)
        attns = attns[0].to_sparse()

        fname = "out" + "{:05d}".format(i) + ".pt"
        torch.save(attns, outPath + fname)

        fname = "in" + "{:05d}".format(i) + ".pt"
        torch.save(rays[0], inPath + fname)

        fname = "in" + "{:05d}".format(i) + ".pose"
        torch.save(poses, inPath + fname)
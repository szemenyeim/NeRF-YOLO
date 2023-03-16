
import os.path as osp
import glob
from torch.utils.data import Dataset
import torch

class NeRFDataSet(Dataset):

    def __init__(self, path, split="train", size="640"):
        super(NeRFDataSet, self).__init__()

        self.inFolder = osp.join(path, split, "in", size)
        self.outFolder = osp.join(path, split, "out", size)

        self.inFiles = sorted(glob.glob1(self.inFolder, "*.pt"))
        self.poseFiles = sorted(glob.glob1(self.inFolder, "*.pose"))
        self.outFiles = sorted(glob.glob1(self.outFolder, "*.pt"))

    def __len__(self):
        return len(self.inFiles)

    def __getitem__(self, item):

        rays = torch.load(osp.join(self.inFolder, self.inFiles[item]))
        poses = torch.load(osp.join(self.inFolder, self.poseFiles[item]))
        centers = poses[:, 0:3, 3]
        centers = centers.unsqueeze(1).repeat(1, rays.shape[1], 1)
        rays = torch.cat([rays, centers], dim=-1)


        out = torch.load(osp.join(self.outFolder, self.outFiles[item])).to_dense()


        return rays, out
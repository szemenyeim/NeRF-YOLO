import math

import torch
from torch import nn
import torch.nn.functional as F

def generateAttentionMask(bsize, size, nviews):
    baseMask = torch.ones((size*nviews, size*nviews)).cuda()*-1e12
    zeros = torch.zeros((size, size)).cuda()
    offs = torch.diag(torch.ones(size)*1e12, 0).cuda()
    for i in range(nviews):
        for j in range(nviews):
            if i == j:
                baseMask[i*size:(i+1)*size, j*size:(j+1)*size] += offs
            else:
                baseMask[i*size:(i+1)*size, j*size:(j+1)*size] = zeros

    return baseMask.unsqueeze(0).repeat(bsize, 1, 1)

class NeRFAttention(nn.Module):

    def __init__(self, nPos=16, nEmbed=64):
        super(NeRFAttention, self).__init__()

        self.nPos = nPos
        self.nEmbed = nEmbed

        self.scaleFactor = nn.Linear(3,1)

        self.queryMLP = nn.Sequential(
            nn.Linear(nPos, nEmbed//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(nEmbed//2, nEmbed),
            nn.LayerNorm(nEmbed)
        )
        self.keyMLP = nn.Sequential(
            nn.Linear(nPos, nEmbed//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(nEmbed//2, nEmbed),
            nn.LayerNorm(nEmbed)
        )

    def forward(self, features, rays, scale):

        bsize, seq, HW, _ = rays.shape

        norm_scale = self.scaleFactor(scale) ** 2

        pos = rays.view(-1, self.nPos)

        Q = self.queryMLP(pos).view(bsize, seq*HW, self.nEmbed)/math.sqrt(self.nEmbed)
        K = self.keyMLP(pos).view(bsize, seq*HW, self.nEmbed)

        mask = generateAttentionMask(bsize, HW, seq)

        attn_Mtx = torch.baddbmm(mask, Q, K.transpose(-2,-1))

        attn_weigths = F.softmax(attn_Mtx*norm_scale, dim=1)

        if features is not None:
            _, _, _, _, nFeat = features.shape
            features = features.view(-1, self.nFeat)
            attn_out = torch.bmm(attn_weigths, features).view(seq, HW, nFeat)
        else:
            attn_out = None


        return attn_out, attn_weigths

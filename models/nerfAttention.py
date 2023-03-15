import math

import torch
from torch import nn
import torch.nn.functional as F


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

        seq, H, W, nFeat = features.shape

        norm_scale = self.scaleFactor(scale)

        pos = rays.view(-1, self.nPos)
        features = features.view(-1, self.nFeat)

        Q = self.queryMLP(pos).view(seq*H*W, self.nEmbed)/math.sqrt(self.nEmbed)
        K = self.keyMLP(pos).view(seq*H*W, self.nEmbed)

        attn_Mtx = torch.mm(Q, K.transpose(-2,-1))
        attn_weigths = self.sm(attn_Mtx*norm_scale)
        attn_out = torch.mm(attn_weigths, features).view(seq, H, W, nFeat)


        return attn_out, attn_weigths

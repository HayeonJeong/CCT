import torch
import math
import torch.nn.functional as F
from torch import nn
#from easydict import EasyDict as edict

class LayerNorm(nn.Module):
    def __init__(self, cfg, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.encoder.d_model))
        self.beta = nn.Parameter(torch.zeros(cfg.encoder.d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
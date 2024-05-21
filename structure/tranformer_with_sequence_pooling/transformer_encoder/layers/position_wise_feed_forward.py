import torch
import math
import torch.nn.functional as F
from torch import nn
from easydict import EasyDict as edict

class PositionWiseFeedForward(nn.Module):
    def __init__(self, cfg): # drop_prob=0.1 (default)
        super(PositionWiseFeedForward, self).__init__()
        self.fc_layer1 = nn.Linear(cfg.encoder.d_model, cfg.encoder.ffn_hidden)
        self.fc_layer2 = nn.Linear(cfg.encoder.ffn_hidden, cfg.encoder.d_model)
        self.gelu = nn.GELU() 
        self.dropout = nn.Dropout(p=cfg.encoder.drop_prob)

    def forward(self, x):
        out = self.fc_layer1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc_layer2(out)
        return out
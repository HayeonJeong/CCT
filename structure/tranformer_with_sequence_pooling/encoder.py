import torch
import math
import torch.nn.functional as F
from torch import nn
import json
from easydict import EasyDict as edict

from structure.tranformer_with_sequence_pooling.transformer_encoder.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
        self.layers = nn.ModuleList([EncoderLayer(cfg)
                                     for _ in range(cfg.encoder.n_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
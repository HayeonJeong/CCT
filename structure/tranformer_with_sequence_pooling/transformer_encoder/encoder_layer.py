import torch
import math
import torch.nn.functional as F
from torch import nn
import json
from easydict import EasyDict as edict

from structure.tranformer_with_sequence_pooling.transformer_encoder.layers.layer_norm import LayerNorm
from structure.tranformer_with_sequence_pooling.transformer_encoder.layers.multi_head_attention import MultiHeadAttention
from structure.tranformer_with_sequence_pooling.transformer_encoder.layers.position_wise_feed_forward import PositionWiseFeedForward

with open('/home/hayeon/CCT/configs/config.json', 'r') as json_file:
    config = json.load(json_file)

class EncoderLayer(nn.Module):
 
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(cfg)
        self.norm1 = LayerNorm(cfg)
        self.dropout1 = nn.Dropout(p=cfg.encoder.drop_prob)

        self.ffn = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.dropout2 = nn.Dropout(p=cfg.encoder.drop_prob)


    def forward(self, x):
        # 1. compute self attention
        x_copy = x
        x = self.attention(x)
        
        # 2. add_and_norm
        x = self.dropout1(x)
        x = self.norm1(x + x_copy)

        # 3. PositionWiseFeedForward
        x_copy = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + x_copy)
        return x
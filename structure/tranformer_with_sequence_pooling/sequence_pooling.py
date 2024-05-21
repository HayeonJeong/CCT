import torch
import math
import torch.nn.functional as F
from torch import nn
import json
from easydict import EasyDict as edict

from structure.tranformer_with_sequence_pooling.encoder import Encoder

class SeqPool(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.encoder.d_model
        self.num_heads = cfg.encoder.num_heads

        self.linear_layer = nn.Linear(self.embed_dim, 1)

    def forward(self, x_L):
        # x 차원 가정 : (batch_size, sequence_length, embedding_dim)
        x_L_reshaped = self.linear_layer(x_L)  # x_L_linear : (batch_size, sequence_length, 1)
        # print("x_L_reshaped.shape : ", x_L_reshaped.shape)
        
        x_L_softmax = F.softmax(x_L_reshaped.transpose(1, 2), dim=-1)  # sequence_length 차원에 softmax
        # print("x_L_softmax.shape : ", x_L_softmax.shape)

        # Generate importance weighting
        z = torch.matmul(x_L_softmax, x_L)  # z : (batch_size, 1, embedding_dim)
        # print("z.shape : ", z.shape)

        # squeeze : 텐서의 차원 중 사이즈가 1인 차원을 제거
        z = z.squeeze(1)  # Flatten the output to (batch_size, embedding_dim)
        
        return z
    
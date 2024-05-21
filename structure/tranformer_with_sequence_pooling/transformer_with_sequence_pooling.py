import torch
import math
import torch.nn.functional as F
from torch import nn
import json
from easydict import EasyDict as edict

from structure.tranformer_with_sequence_pooling.encoder import Encoder
from structure.tranformer_with_sequence_pooling.sequence_pooling import SeqPool

class TransformerWithSeqPool(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.seq_pool = SeqPool(cfg)
        self.final_linear = nn.Linear(cfg.conv_token.output_channels, cfg.conv_token.num_classes)  # Adjust num_classes as needed

    def forward(self, conv_tokenizer_output):
        encoder_output = self.encoder(conv_tokenizer_output)
        seqpool_output = self.seq_pool(encoder_output)
        final_output = self.final_linear(seqpool_output)
        return final_output

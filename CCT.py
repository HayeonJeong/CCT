import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import json

from structure.convolutional_tokenization.ConvolutionalTokenizer import ConvolutionalTokenizer
from structure.tranformer_with_sequence_pooling.transformer_with_sequence_pooling import TransformerWithSeqPool

class CCT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.Convolutional_Tokenization = ConvolutionalTokenizer(cfg)
        self.Transformer_With_SeqPool = TransformerWithSeqPool(cfg)

    def forward(self, x):
        conv_tokenizer_output = self.Convolutional_Tokenization(x) # torch.Size([1, 131072])
        cct_ouput = self.Transformer_With_SeqPool(conv_tokenizer_output)

        return cct_ouput
    
config_path = '/home/hayeon/CCT/configs/CCT-2-3x2.json'
with open(config_path, 'r') as f:
    cfg_dict = json.load(f)

cfg = edict(cfg_dict)
model = CCT(cfg)
model = model.cuda()
print(model)
sample=torch.randn(3, 3, 32, 32).cuda() # [batch_size, input_channels, height, width]
y=model(sample)
print(y.shape)
#print(y)
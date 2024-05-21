import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import json

class ConvolutionalTokenizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_conv_layers = cfg.conv_token.num_conv_layers
        self.kernel_size = cfg.conv_token.kernel_size
        self.conv_layers = nn.ModuleList()

        conv_layer = nn.Conv2d(cfg.conv_token.input_channels,
                                   cfg.conv_token.output_channels,
                                   cfg.conv_token.kernel_size,
                                   cfg.conv_token.stride,
                                   cfg.conv_token.padding)
        relu = nn.ReLU()
        maxpool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=cfg.conv_token.stride)
        layer = nn.Sequential(conv_layer, relu, maxpool)
        self.conv_layers.append(layer)

        for _ in range(self.num_conv_layers - 1):
            conv_layer = nn.Conv2d(cfg.conv_token.output_channels, # not input_channels
                                   cfg.conv_token.output_channels,
                                   cfg.conv_token.kernel_size,
                                   cfg.conv_token.stride,
                                   cfg.conv_token.padding)
            layer = nn.Sequential(conv_layer, relu, maxpool)
            self.conv_layers.append(layer)

    def forward(self, x):
        # Input tensor shape: [batch_size, input_channels, height, width] = [1, 3, 32, 32]

        # print(x.shape) # torch.Size([1, 3, 32, 32]), if CCT-2-3x2

        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            # print(x.shape) # torch.Size([1, 128, 32, 32]), if CCT-2-3x2
        
        # 차원 변경 (Transpose) - (1, 128, 32, 32) -> (1, 32, 32, 128)
        x_transposed = torch.transpose(x, 1, 2)
        x_transposed = torch.transpose(x, 2, -1)

        # Reshape - (1, 32, 32, 128) -> (1, 32 * 32, 128)
        x_reshaped = x_transposed.reshape(x.size(0), -1, x.size(1))
        # print(x_reshaped.shape) # (1, 1024, 128)

        # Output tensor shape: [batch_size, flattend, output_channel = d_model = embed_dim]
        
        return x_reshaped
'''
config_path = '/home/hayeon/CCT/configs/CCT-7-3x2.json'
with open(config_path, 'r') as f:
    cfg_dict = json.load(f)

cfg = edict(cfg_dict)
model = ConvolutionalTokenizer(cfg)
model = model.cuda()
print(model)
'''
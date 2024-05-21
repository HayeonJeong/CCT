import torch
import torch.nn as nn
from torchsummary import summary as summary
from CCT import CCT
from easydict import EasyDict as edict
import json
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')

    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=5, metavar='E',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--cfg', type=str, default='config.json',
                        help='path to the configuration file (default: config.json)')
    
    args = parser.parse_args()

    return args


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    config = edict(config)

    return config

args = parse_arguments()
config = load_config(args.cfg)

check_cct = CCT(config)
print(check_cct)
check_cct.cuda()
summary(check_cct, (3, 32, 32))
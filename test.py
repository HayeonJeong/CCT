import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pandas as pd
from CCT import CCT
import argparse
import json
from easydict import EasyDict as edict

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    config = edict(config)
    return config

def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    config = load_config(args.cfg)
    model = CCT(config).to(device)
    #model.load_state_dict(torch.load(args.model_path))

    # 모델을 불러올 때 DataParallel로 래핑한 경우
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    test(model, test_loader, device)

if __name__ == '__main__':
    parser = ArgumentParser(description='Test the CCT model on CIFAR-10 dataset')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--cfg', type=str, default='config.json',
                        help='path to the configuration file (default: config.json)')
    args = parser.parse_args()

    main(args)

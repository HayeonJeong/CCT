import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import json
import argparse
from easydict import EasyDict as edict
import os

from CCT import CCT


# load config from JSON file
def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    config = edict(config)

    return config


# Define data loaders
def get_data_loaders(batch_size=4):
    transform = transforms.Compose(
        [transforms.ToTensor()
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader


def train_one_epoch(epoch, model, trainloader, criterion, optimizer, device, print_interval=10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % print_interval == print_interval - 1:
            epoch_loss = running_loss / print_interval
            epoch_acc = 100. * correct / total
            print(f"Batch {i+1}, Epoch {epoch+1} - Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.2f}%")
            running_loss = 0.0
            correct = 0
            total = 0

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total

    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.3f}, Accuracy: {epoch_acc:.2f}%")
    print("============================")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    config = load_config(args.cfg)

    model = CCT(config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    trainloader, testloader = get_data_loaders(batch_size=args.batch_size)

    for epoch in range(args.epochs):
        train_one_epoch(epoch, model, trainloader, criterion, optimizer, device)

        if (epoch+1) % 5 == 0:
            model_name = f"{args.cfg.split('.')[0].split('/')[-1]}_{args.batch_size}_{epoch}.pth"
            print(model_name)
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, model_name))


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


if __name__ == '__main__':
    main()

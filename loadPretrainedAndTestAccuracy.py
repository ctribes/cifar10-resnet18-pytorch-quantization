'''CIFAR10 with PyTorch.'''
'''Load a pretrained ResNet18 network state from a checkpoint and test the accuracy. '''
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging

from models import *

_logger = logging.getLogger("cifar10_pytorch_automl")

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def prepare(args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global scheduler
    
    # Data
    print('==> Preparing data (cifar10)')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model ResNet18 for cifar10 and load pretrained')
    net = resnet18(pretrained=True, pretrained_checkpoint=args.pretrained_checkpoint, num_classes=10)
    net = net.to(device)
    
    if device == 'cuda':
            # net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            print('==> using cuda')
    
    criterion = nn.CrossEntropyLoss()

def test():
    global best_acc
    global testloader
    global net
    global criterion

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total

    return acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Maximum mini-batches per epoch, for code testing purpose
    parser.add_argument('--batch_size', type=int, default=256)
        
    # Checkpoint file
    parser.add_argument('--pretrained_checkpoint', default='./checkpoint/resnet18-cifar10-fp.pth')

    args, _ = parser.parse_known_args()
    
    try:
    
        _logger.debug(args)
        
        prepare(args)
        acc = 0.0

        print(test())
        
        
    except Exception as exception:
        _logger.exception(exception)
        raise


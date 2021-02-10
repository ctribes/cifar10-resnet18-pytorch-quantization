'''Test load state CIFAR10 with PyTorch.'''
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
# from utils import progress_bar


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
    global testloader
    global net
    global criterion
    # Data
    print('==> Preparing CIFAR10 test data (no train data) ...')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model ResNet18 for cifar10')
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
    acc = 0
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

            # print(total, correct, acc)
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # batch size
    parser.add_argument('--batch_size', type=int, default=256)
    
    # Checkpoint file
    parser.add_argument('--pretrained_checkpoint', default='./checkpoint/resnet18-cifar10.pth')
    
    args, _ = parser.parse_known_args()
    
    try:

        _logger.debug(args)
        
        prepare(args)
    
        acc = test()
        print('acc=',acc)
        
        
    except Exception as exception:
        _logger.exception(exception)
        raise

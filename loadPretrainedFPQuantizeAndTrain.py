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

# Use for quantization
from models.quantization import *


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
    global trainloader
    global net
    global criterion
    global optimizer
    global scheduler
    
    print('==> Preparing data (cifar10)')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building quantized model ResNet18 for cifar10')
    net = resnet18(pretrained = True, pretrained_checkpoint=args.pretrained_checkpoint, num_classes=10, quantize = True)
    net = net.to(device)
        
    if device == 'cuda':
            # net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            print('==> using cuda')
        
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adamax':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        print('Unknown optimizer')

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=20)

# Training
def train(epoch, batches=-1):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global scheduler

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if batches > 0 and (batch_idx+1) >= batches:
            return
    scheduler.step(epoch)


def test(save_checkpoint):
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

#            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    if acc > best_acc:
        best_acc = acc
    
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        print('Saving the network using jit')
        net=net.cpu()
        net.eval()
        net_int8 = torch.quantization.convert(net)

        example_forward_input = torch.rand(1, 3, 32, 32)
        example_forward_input=example_forward_input.cpu()

        net_trace = torch.jit.trace(net_int8,example_forward_input)
        torch.jit.save(net_trace,save_checkpoint)
        net=net.to(device)
        
    return acc, best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=200)

    # Maximum mini-batches per epoch, for code testing purpose
    parser.add_argument("--batches", type=int, default=-1)
    
    # Maximum mini-batches per epoch, for code testing purpose
    parser.add_argument('--batch_size', type=int, default=256)
    
    # Optimizer
    parser.add_argument('--optimizer', default='SGD')
    
    # Learning rate
    parser.add_argument('--lr', type=float, default=0.1)
    
    # Weight decay
    parser.add_argument('--weight_decay', type=float, default=0.00000001)
    
    # Checkpoint file
    parser.add_argument('--pretrained_checkpoint', default='./checkpoint/resnet18-cifar10.pth')

    # Checkpoint file
    parser.add_argument('--save_checkpoint', default='./checkpoint/resnet18-cifar10-int8.pth')

    args, _ = parser.parse_known_args()
    
    try:

        _logger.debug(args)
        
        prepare(args)
        acc = 0.0
        best_acc = 0.0
        for epoch in range(start_epoch, start_epoch+args.epochs):
            train(epoch, args.batches)
            acc, best_acc = test(args.save_checkpoint)
            print(acc,best_acc)
        
    except Exception as exception:
        _logger.exception(exception)
        raise

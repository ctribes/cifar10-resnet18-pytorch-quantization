'''Train CIFAR10 with PyTorch.'''
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

from models.quantization import *
# from utils import progress_bar

# import nni

import numpy as np
import sys

_logger = logging.getLogger("cifar10_pytorch_nomad")

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

previousHistoryFileIsProvided = False
previousHistoryFileName = "history.prev.txt"
currentHistoryFileName = "history.0.txt"

def prepare(args,pretrained_checkpoint):
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=2)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False, num_workers=2)


    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building quantized model ResNet18 for cifar10')
    net = resnet18(pretrained = True, pretrained_checkpoint=pretrained_checkpoint, num_classes=10, quantize = True)
    net = net.to(device)
        
    if device == 'cuda':
            # net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            print('==> using cuda')
        
    criterion = nn.CrossEntropyLoss()

    if args['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    elif args['optimizer'] == 'Adamax':
        optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
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

def test(enable_save_checkpoint,save_checkpoint):
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

        if enable_save_checkpoint:
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

# This function must be customized to the problem (order and value of the variables)
def get_parameters_from_Nomad_input(x):
    
    config = dict()
    # print(x)
    
    
    # FROM JSON file
    #1    "batch_size": {"_type":"choice", "_value": [32, 64, 128, 256]},
    #2    "weight_decay": {"_type":"choice", "_value": [0, 0.00000001, 0.0000001, 0.000001]},
    #3    "lr":{"_type":"choice", "_uniform":[-4,-1]},
    #4     "optimizer":{"_type":"choice", "_value":["SGD", "Adadelta", "Adagrad", "Adam", "Adamax"]}
    
    # 1 -> batch size
    if x[0] == 1:
        valueBS = 32
    elif x[0] == 2:
        valueBS = 64
    elif x[0] == 3:
        valueBS = 128
    elif x[0] == 4:
        valueBS = 256
    else:
        return config
    config['batch_size'] = valueBS
     
    # 2 -> weight decay
    if x[1] == 1:
        valueWD = 0
    elif x[1] == 2:
        valueWD = 0.00000001
    elif x[1] == 3:
        valueWD = 0.0000001
    elif x[1] == 4:
        valueWD = 0.000001
    else:
        return config
    config['weight_decay'] = valueWD
    
    
    # !!!!!
    # 3 -> Learning rate: lr = 10^x3
    # !!!!!
    if x[2] < 0:
        valueLR = pow(10,x[2])
    else:
        return config
    # print(config)
    config['lr'] = valueLR
 
    # 4 -> Optimizer
    if x[3] == 1:
        valueOptim = 'SGD'
    elif x[3] == 2:
        valueOptim = 'Adadelta'
    elif x[3] == 3:
        valueOptim = 'Adagrad'
    elif x[3] == 4:
        valueOptim = 'Adam'
    elif x[3] == 5:
        valueOptim = 'Adamax'
    else:
        return config
    config['optimizer'] = valueOptim

    return config
    
    
def bb(X):
    global previousHistoryFileIsProvided
    global best_acc
    
    dim = len(X)
    
    #
    # Use this section if a history file exists before starting (see below)
    #
    if previousHistoryFileIsProvided:
        for oneEval in npEvals:
        # print('Dim=',dim,' ',oneEval)
            diffX=X-oneEval[0:dim]
            if np.linalg.norm(diffX) < 1E-10:
                best_acc=-oneEval[dim]
                # X.set_bb_output(0,oneEval[dim])
                print('Find point in file: ',X,' f(x)=',oneEval[dim])
                return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)

    # Maximum mini-batches per epoch, for code testing purpose
    parser.add_argument("--batches", type=int, default=-1)
    
    # Checkpoint file
    parser.add_argument('--pretrained_checkpoint', default='./checkpoint/resnet18-cifar10-fp.pth')

    # Checkpoint file
    parser.add_argument('--save_checkpoint', default='./checkpoint/resnet18-cifar10-int8.pth')

    args, _ = parser.parse_known_args()

    try:
        RCV_CONFIG = get_parameters_from_Nomad_input(X)
        # print(RCV_CONFIG)
        _logger.debug(RCV_CONFIG)

        prepare(RCV_CONFIG,args.pretrained_checkpoint)
        acc = 0.0
        best_acc = 0.0
        for epoch in range(start_epoch, start_epoch+args.epochs):
            train(epoch, args.batches)
            enable_save_checkpoint = (int(epoch)>100)
            acc, best_acc = test(enable_save_checkpoint,args.save_checkpoint)
            print('After test: ',acc,' ', best_acc)
            if ((int(epoch)>49) and (best_acc<=12.0)):
                print('Stop after 50 epochs because best_acc<=12')
                best_acc=0.0
                break
            elif ((int(epoch)>99) and (best_acc<=60.0)):
                print('Stop after 100 epochs because best_acc<=60.0')
                best_acc=0.0
                break
        print('For loop on epoch has ended (',epoch,'). Best_acc=',best_acc)
        # print( -best_acc )
        
    except Exception as exception:
        _logger.exception(exception)
        return 0
    return 1

if __name__ == '__main__':

    X=np.fromfile(sys.argv[1],sep=" ")
    #
    # Use this section if a history file exists before starting (see above)
    #
    if previousHistoryFileIsProvided:
        
        if previousHistoryFileName == currentHistoryFileName:
            print('Previous and current history files should not be the same')
            exit(0)
    
        # Read a history file from a run
        rawEvals = np.fromfile(previousHistoryFileName,sep=" ")
        # Each line contains 2 values for X and a single output (objective value)
        nbRows = rawEvals.size/(len(X)+1)
        # print(nbRows)
        # Split the whole array is subarrays
        npEvals = np.split(rawEvals,nbRows)
        #print(npEvals)

    # print(x)
    bb(X)
    if best_acc==0.0:
        print('Final_best_acc= Inf',)
    else:
        print('Final_best_acc=',-best_acc)

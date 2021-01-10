# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchsummary import summary

import os
import sys
import argparse

import utils
from utils import he_init
from sensitivity import Sensitivity
from model import resnet, keskar_models
from rescale import rescale
from config import config
from sharpness import Sharpness


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

ONE_OFF = (config.sensitivity_one_off or
        config.sharpness_one_off)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
trainset, trainloader, testset, testloader = utils.load_data(
        config.train_batch_size,
        config.test_batch_size,
        config.num_workers,
        dataset='CIFAR10',
        training_set_size=config.training_set_size,
        binary=config.binary_dataset)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
if config.binary_dataset:
    net = resnet.ResNet50(num_classes=2)
    #net = keskar_models.C1(num_classes=2)
else:
    net = resnet.ResNet50()
    #net = keskar_models.C1(num_classes=10)
# net = resnet.ResNet50()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()

# He-normal Initialization
net.apply(he_init)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
# summary(net, (3, 32, 32))
# rescale(net, 'layer1', 1, 2)
# sys.exit()
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(
            os.path.join(config.output_file_pth, 'checkpoint')
            ), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(
            os.path.join(
                config.output_file_pth, 'checkpoint/ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if config.optim == 'SGD+Momentum':
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
            momentum=0.9, weight_decay=5e-4)
elif config.optim == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif config.optim == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

# Learning Rate Decay
decayRate = 0.96
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
# Training


def train(epoch):
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

        #progress_bar(
        #    batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # Above is when using terminal. Now I would like to run it on computing cores.
        if (batch_idx+1) % 50 == 0:
            print('Training On Batch %03d' % (batch_idx+1))
    epoch_loss = train_loss/(batch_idx+1)
    epoch_acc = correct/total
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch_loss, 100.*epoch_acc, correct, total))
    return (epoch_loss, 100.*epoch_acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        with torch.no_grad():
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx+1) % 20 == 0:
                print('Testing On Batch %03d' % (batch_idx+1))
    epoch_loss = test_loss/(batch_idx+1)
    epoch_acc = correct/total
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch_loss, 100.*epoch_acc, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(
                os.path.join(config.output_file_pth, 'checkpoint')):
            os.mkdir(
                    os.path.join(config.output_file_pth, 'checkpoint'))
        torch.save(state, os.path.join(
            config.output_file_pth, 'checkpoint/ckpt.pth'))
        best_acc = acc

    return (epoch_loss, 100.*epoch_acc)

if __name__ == '__main__':
    assert os.path.isdir(config.output_file_pth), "Target Output Directory Doesn't Exist!"
    train_loss_acc_list = []
    test_loss_acc_list = []
    if config.sharpness_cons == True:
        sharpness_cons = []
    if config.sensitivity_cons == True:
        sensitivity_cons = []
    for epoch in range(start_epoch, start_epoch+config.train_epoch):
        # if (epoch + 1) == 100:
        #     rescale(net, 'all', None, None, config.alpha)
        #     adjust_learning_rate(optimizer, args.lr)
        train_returns = train(epoch)
        test_returns = test(epoch)

        train_loss_acc_list.append(train_returns)
        test_loss_acc_list.append(test_returns)

        if config.sharpness_cons == True:
            S = Sharpness(net, criterion, trainset, device)
            sharpness_cons.append(S.sharpness(opt_mtd=config.sharpness_method))
        if config.sensitivity_cons == True:
            Sen_train = Sensitivity(net, trainset, device, epoch)
            # Sen_test = Sensitivity(net, testset, device, epoch)
            sensitivity_cons.append(Sen_train.sensitivity())
        if config.lr_decay:
            lr_scheduler.step()

        '''
        if ONE_OFF == True:
            if train_returns[1] == 100.:
                break
        '''

    if config.sensitivity_one_off == True:
        Sen_train = Sensitivity(net, trainset, device, epoch)
        sensitivity_one_off = Sen_train.sensitivity()
        print('The sensitivity at reaching zero training error is: ', sensitivity_one_off)
    if config.sharpness_one_off == True:
        S = Sharpness(net, criterion, trainset, device)
        sharpness_one_off = S.sharpness(opt_mtd=config.sharpness_method)

    np.save(os.path.join(
        config.output_file_pth,'train_loss_acc_list.npy'), train_loss_acc_list)
    np.save(os.path.join(
        config.output_file_pth, 'test_loss_acc_list.npy'), test_loss_acc_list)
    if config.sharpness_cons == True:
        np.save(os.path.join(
            config.output_file_pth, 'sharpness_list.npy'), sharpness_cons)
    if config.sensitivity_cons == True:
        np.save(os.path.join(
             config.output_file_pth, 'sensitivity_list.npy'), sensitivity_cons)


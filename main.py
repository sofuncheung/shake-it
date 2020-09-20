# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

import os
import sys
import argparse

from sensitivity import compute_jacobian_norm_sum
from model import resnet
from rescale import rescale
from config import config
from utils import adjust_learning_rate
from sharpness import Sharpness


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='~/shake-it/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers)

testset = torchvision.datasets.CIFAR10(
    root='~/shake-it/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = resnet.ResNet18()
net = resnet.ResNet50()
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
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#summary(net, (3, 32, 32))
#rescale(net, 'layer1', 1, 2)
#sys.exit()
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
    jacobian_norm_sum = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if config.sensitivity_cons == True:
            inputs.requires_grad = True
        outputs = net(inputs)
        if config.sensitivity_cons == True:
            jacobian_norm_sum += compute_jacobian_norm_sum(inputs, outputs)
        with torch.no_grad():
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(
            #     batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1),
            #        100.*correct/total, correct, total))
            if (batch_idx+1) % 20 == 0:
                print('Testing On Batch %03d' % (batch_idx+1))
    epoch_loss = test_loss/(batch_idx+1)
    epoch_acc = correct/total
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (epoch_loss, 100.*epoch_acc, correct, total))

    epoch_sensitivity = jacobian_norm_sum / len(testset)

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

    return ((epoch_loss, 100.*epoch_acc), epoch_sensitivity)

if __name__ == '__main__':
    assert os.path.isdir(config.output_file_pth), "Target Output Directory Doesn't Exist!"
    train_loss_acc_list = []
    test_loss_acc_list = []
    if config.sharpness == True:
        sharpness = []
    if config.sensitivity_cons == True:
        sensitivity_cons = []
    for epoch in range(start_epoch, start_epoch+200):
        # if (epoch + 1) == 100:
        #     rescale(net, 'all', None, None, config.alpha)
        #     adjust_learning_rate(optimizer, args.lr)
        train_returns = train(epoch)
        test_returns = test(epoch)
        train_loss_acc_list.append(train_returns)
        test_loss_acc_list.append(test_returns[0])

        if config.sharpness == True:
            S = Sharpness(net, criterion, trainset, device)
            sharpness.append(S.sharpness())
        if config.sensitivity_cons == True:
            sensitivity_cons.append(test_returns[1])
        if config.lr_decay:
            lr_scheduler.step()
    np.save(os.path.join(
        config.output_file_pth,'train_loss_acc_list.npy'), train_loss_acc_list)
    np.save(os.path.join(
        config.output_file_pth, 'test_loss_acc_list.npy'), test_loss_acc_list)
    if config.sharpness == True:
        np.save(os.path.join(
            config.output_file_pth, 'sharpness_list.npy'), sharpness)
    if config.sensitivity_cons == True:
        np.save(os.path.join(
            config.output_file_pth, 'sensitivity_list.npy'), sensitivity_cons)


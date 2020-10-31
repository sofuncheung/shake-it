# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:38:23 2020

Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""


import os
import sys
import time
import numpy as np
from scipy import optimize
from collections import OrderedDict
from functools import reduce
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler

import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class ScipyOptimizeWrapper(object):
    r'''
    Provide interface between Pytorch model (nn.Module)
    and scipy optimizer (default L-BFGS-B)

    In following context, "pack" means load numpy stuff
    to Pytorch model; "unpack" means load Pytorch parameters
    to numpy ndarray.
    '''
    def __init__(self, model, loss, full_batch_loader):
        self.model = model
        self.loss = loss
        parameters = OrderedDict(model.named_parameters())
        self.param_shapes = {n:parameters[n].size() for n in parameters}
        self.x0 = np.concatenate([parameters[n].data.numpy().ravel()
                                for n in parameters])
        self.full_batch_loader = full_batch_loader

    def pack_parameters(self, x):
        r'''
        Chopping up 1D numpy array x and pack them into Pytorch model
        format (named_parameters).
        '''
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x,y: x*y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i:i+param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters

    def unpack_grads(self):
        r'''
        Unpack all the gradients from the parameters in the module into a
        numpy array.
        '''
        grads = []
        for p in self.model.parameters():
            grad = p.grad.data.numpy()
            grads.append(grad.ravel())
        return np.concatenate(grads)

    def is_new(self, x):
        # if this is the first thing we've seen
        if not hasattr(self, 'cached_x'):
            return True
        else:
            # compare x to cached_x to determine if we've been given a new input
            x, self.cached_x = np.array(x), np.array(self.cached_x)
            error = np.abs(x - self.cached_x)
            return error.max() > 1e-8

    def cache(self, x):
        # load x into model
        state_dict = self.pack_parameters(x)
        self.model.load_state_dict(state_dict,strict=False)
        # store the raw array as well
        self.cached_x = x
        # zero the gradient
        self.model.zero_grad()
        # use it to calculate the objective
        for batch_idx, (inputs, targets) in enumerate(self.full_batch_loader):
            outputs = self.model(inputs)
            obj = -1 * self.loss(outputs, targets)
        # backprop the objective
        obj.backward()
        self.cached_f = obj.item()
        self.cached_jac = self.unpack_grads()

    def f(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_f

    def jac(self, x):
        if self.is_new(x):
            self.cache(x)
        return self.cached_jac

    def bounds(self, eps):
        bounds = []
        lower_bounds = -eps * (np.abs(self.x0) + 1)
        upper_bounds = eps * (np.abs(self.x0) + 1)
        for i in range(len(lower_bounds)):
            bounds.append((lower_bounds[i],upper_bounds[i]))
        return bounds


class BinaryCIFAR10(Dataset):
    '''
    Children class of torch.utils.data.Dataset
    Specifically for binary CIFAR10 (car and cat)
    dataset.
    '''

    def __init__(self, npy_file_x_train,
            npy_file_y_train,
            npy_file_x_test,
            npy_file_y_test,
            is_train=True, transform=None, target_transform=None):
        self.x_train = np.load(npy_file_x_train)
        self.y_train = self.turn_cifar_label_into_binary(
                np.load(npy_file_y_train))
        self.x_test = np.load(npy_file_x_test)
        self.y_test = self.turn_cifar_label_into_binary(
                np.load(npy_file_y_test))
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train

        if self.is_train:
            self.data = self.x_train
            self.targets = self.y_train
        else:
            self.data = self.x_test
            self.targets = self.y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    @staticmethod
    def turn_cifar_label_into_binary(y_train):
        temp = []
        for i in range(y_train.shape[0]):
            if y_train[i][0] == 3:
                temp.append(1)
            elif y_train[i][0] == 1:
                temp.append(0)
        temp = np.array(temp).astype(np.int64)
        return temp


def load_data(train_batch_size,
        test_batch_size,
        num_workers,
        dataset='CIFAR10', training_set_size=50000, binary=True):
    print('==> Preparing data..')

    if dataset == 'CIFAR10':
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

        if binary==False:
            trainset = torchvision.datasets.CIFAR10(
                root='~/shake-it/data', train=True, download=True, transform=transform_train)

            testset = torchvision.datasets.CIFAR10(
                root='~/shake-it/data', train=False, download=True, transform=transform_test)
        else:
            trainset = BinaryCIFAR10(
                '/mnt/zfsusers/sofuncheung/cifar10/x_train_car_and_cat.npy',
                '/mnt/zfsusers/sofuncheung/cifar10/y_train_car_and_cat.npy',
                '/mnt/zfsusers/sofuncheung/cifar10/x_test_car_and_cat.npy',
                '/mnt/zfsusers/sofuncheung/cifar10/y_test_car_and_cat.npy',
                is_train=True, transform=transform_train)
            testset = BinaryCIFAR10(
                '/mnt/zfsusers/sofuncheung/cifar10/x_train_car_and_cat.npy',
                '/mnt/zfsusers/sofuncheung/cifar10/y_train_car_and_cat.npy',
                '/mnt/zfsusers/sofuncheung/cifar10/x_test_car_and_cat.npy',
                '/mnt/zfsusers/sofuncheung/cifar10/y_test_car_and_cat.npy',
                is_train=False, transform=transform_test)

    if training_set_size == len(trainset):
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
            )
    else:
        indices = list(range(len(trainset)))
        np.random.shuffle(indices)
        train_indices = indices[:training_set_size]
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=train_batch_size,
            sampler=SubsetRandomSampler(train_indices),
            shuffle=False,
            num_workers=num_workers,
            drop_last=True
            )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_batch_size,
        shuffle=False, num_workers=num_workers,
        drop_last=True)

    return trainset, trainloader, testset, testloader

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

if __name__ == '__main__':
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    TOTAL_BAR_LENGTH = 65.
    last_time = time.time()
    begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def max_value_plot(file_name):
    max_value_list = np.load(file_name)
    plt.plot(max_value_list)
    plt.savefig('max_value_list.png', dpi=300)

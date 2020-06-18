# -*- coding: utf-8 -*-

"""
Author: Sofun Cheung
Date: June 18 2020
Implement of sharpness/flatness calculation
"""

import torch
import torchvision
import torch.optim as optim
import numpy as np

import config


class Sharpness(object):

    def __init__(self, net, loss, dataset, device):
        self.net = net
        self.loss = loss
        self.trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.train_batch_size,
                shuffle=True, num_workers=config.num_workers)
        self.testloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.test_batch_size,
                shuffle=False, num_workers=config.num_workers)
        self.optimizer = optim.SGD(net.parameters(), lr=1e-3) # Have to use vanilla SGD.
        self.device = device

    @ staticmethod
    def clip_params(eps, params, new_params):
        for i in range(len(new_params)):
            diff = new_params[i] - params[i]
            eps_mtx = eps * (np.abs(params[i]) + 1)
            outer_up = np.where(diff>eps_mtx)
            diff[outer_up] = eps_mtx[outer_up]
            outer_low = np.where(diff<-eps_mtx)
            diff[outer_low] = -eps_mtx[outer_low]
            new_params[i] = params[i] + diff
        return new_params


    def sharpness(self, clip_eps=1e-4, max_iter_epochs=10):
        net.eval()
        L_w = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.loss(outputs, targets)
                L_w += loss.item()
        w = net.parameters()
        return len(list(w))
        max_value = 0
        max_value_list = []
        net.train()
        for sharpness_epoch in range(max_iter_epochs):
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = net(inputs)
                loss = self.loss(outputs, targets)
                loss.backward()
                self.optimizer.step()

                new_w = net.parameters()
                new_w = clip_params(clip_eps, w, new_w)



if __name__ == '__main__':
    from main import *
    S = Sharpness(net, criterion, trainset, device)
    print(S.sharpness())



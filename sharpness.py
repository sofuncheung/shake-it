# -*- coding: utf-8 -*-

"""
Author: Sofun Cheung
Date: June 18 2020
Implement of sharpness/flatness calculation
"""

import torch
import torchvision
import torch.optim as optim
import sys
import gc
import os
import numpy as np
import copy

from config import config


class Sharpness(object):

    def __init__(self, net, loss, dataset, device):
        self.net = copy.deepcopy(net)
        self.loss = loss
        self.trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.train_batch_size,
                shuffle=True, num_workers=config.num_workers)
        self.testloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.test_batch_size,
                shuffle=False, num_workers=config.num_workers)
        # self.optimizer = optim.SGD(net.parameters(), lr=1e-3) # Have to use vanilla SGD.
        self.device = device

    def clip_params(self, eps, params, new_params):
        for i in new_params:
            diff = new_params[i] - params[i]
            eps_mtx = eps * (torch.abs(params[i]) + 1)
            outer_up = torch.nonzero(diff>eps_mtx, as_tuple=False)
            if outer_up.shape[0] != 0:
                outer_up = [tuple(temp) for temp in outer_up]
                for j in outer_up:
                    diff[j] = eps_mtx[j]
            outer_low = torch.nonzero(diff<-eps_mtx, as_tuple=False)
            if outer_low.shape[0] != 0:
                outer_low = [tuple(temp) for temp in outer_low]
                for j in outer_low:
                    diff[j] = -eps_mtx[j]
            new_params[i] = params[i] + diff
            del diff, eps_mtx, outer_up, outer_low
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            else:
                gc.collect()
        return new_params


    def sharpness(self, clip_eps=1e-4, max_iter_epochs=1, opt_mtd='SGD'):
        net = self.net
        net.eval()
        L_w = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = net(inputs)
                loss = self.loss(outputs, targets)
                L_w += loss.item()
            L_w = L_w/(batch_idx+1)
        w = net.state_dict()
        max_value = 0
        max_value_list = []

        if opt_mtd == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=1e-3)
        net.train()
        for sharpness_epoch in range(max_iter_epochs):
            epoch_loss = 0
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                new_loss = -1. * self.loss(outputs, targets)
                new_loss.backward()
                optimizer.step()

                new_w = net.state_dict()
                self._print_different_w(w, new_w)
                sys.exit()
                new_w = self.clip_params(clip_eps, w, new_w)
                assert self._test_clip_is_effective(clip_eps, w, new_w), 'Error: Fail Box!!!'
                net.load_state_dict(new_w, strict=True)
                del new_w
                torch.cuda.empty_cache()

                new_outputs = net(inputs)
                epoch_loss += self.loss(new_outputs, targets).item()
                print('Batch Loss:', self.loss(new_outputs, targets).item())
            epoch_loss = epoch_loss / (batch_idx+1)
            max_value = max(max_value, epoch_loss)
            max_value_list.append(max_value)
        np.save(os.path.join(
            config.output_file_pth, 'max_value_list.npy'), max_value_list)
        sharpness = 100 * (max_value - L_w) / (1 + L_w)
        return sharpness


    @staticmethod
    def _test_clip_is_effective(eps, params, new_params):
        for i in new_params:
            if torch.max(new_params[i] - params[i]) > eps:
                return False
            else:
                return True

    @staticmethod
    def _max_diff(params, new_params):
        l = []
        for i in new_params:
            l.append(float(torch.max(new_params[i] - params[i])))
        return max(l)

    @staticmethod
    def _median_diff(params, new_params):
        l = []
        for i in new_params:
            l.append(float(torch.median(new_params[i] - params[i])))
        return l

    @staticmethod
    def _print_first_w(params):
        for i in params:
            print(params[i])
            break

    @staticmethod
    def _print_different_w(params, new_params):
        for i in new_params:
            if not torch.equal(new_params[i], params[i]):
                print(params[i])
                print('*'*88)
                print(new_params[i])

if __name__ == '__main__':
    from main import net, criterion, trainset, device
    S = Sharpness(net, criterion, trainset, device)
    print('Sharpness:', S.sharpness())


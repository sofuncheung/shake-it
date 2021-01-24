# -*- coding: utf-8 -*-

"""
Author: Sofun Cheung
Date: 19 Sep 2020
Implement of input sensitivity calculation
"""

import torch
import torchvision
import numpy as np
import copy
from torch.autograd.gradcheck import zero_gradients



class Sensitivity(object):
    '''
    Input Sensitivity calculation.
    Basically the input sensitivity is defined as the
    average jacobian norm of output w.r.t. input
    (on training set or test set!)

    The average on test set case can be seen in paper:
    Novak, Roman, et al.
    "Sensitivity and generalization in neural networks:
    an empirical study." arXiv preprint arXiv:1802.08760 (2018).
    '''

    def __init__(self, net, dataset, device, epoch,
            test_batch_size,
            num_workers
            ):
        self.net = copy.deepcopy(net)
        self.net.eval()
        self.dataset = dataset
        # self.trainloader = torch.utils.data.DataLoader(
        #         dataset, batch_size=config.train_batch_size,
        #         shuffle=True, num_workers=config.num_workers,
        #         drop_last=True)
        self.testloader = torch.utils.data.DataLoader(
                dataset, batch_size=test_batch_size,
                shuffle=False, num_workers=num_workers,
                drop_last=True)
        self.device = device
        self.epoch = epoch

    @staticmethod
    def compute_jacobian_norm_sum(inputs, output):
        """
        :param inputs: Batch X Size (e.g. Depth X Width X Height)
        :param output: Batch X Classes
        :return: jacobian: Batch X Classes X Size
        """
        assert inputs.requires_grad

        num_classes = output.size()[1]

        jacobian = torch.zeros(num_classes, *inputs.size())
        grad_output = torch.zeros(*output.size())
        if inputs.is_cuda:
            grad_output = grad_output.cuda()
            jacobian = jacobian.cuda()

        for i in range(num_classes):
            zero_gradients(inputs)
            grad_output.zero_()
            grad_output[:, i] = 1
            output.backward(grad_output, retain_graph=True)
            jacobian[i] = inputs.grad.data

        jacobian = torch.transpose(jacobian, dim0=0, dim1=1)
        jacobian = torch.reshape(jacobian, (jacobian.shape[0], -1))
        return torch.sum(torch.norm(jacobian, dim=1))

    def sensitivity(self):
        jacobian_norm_sum = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad = True
            outputs = self.net(inputs)
            jacobian_norm_sum += self.compute_jacobian_norm_sum(inputs, outputs)

        epoch_sensitivity = float(jacobian_norm_sum / len(self.dataset))

        return epoch_sensitivity


    def sensitivity_sigmoid(self):
        '''
        sigmoid for binary-dataset case: when the output is a single scalar
        '''
        jacobian_norm_sum = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad = True
            outputs = torch.sigmoid(self.net(inputs))
            jacobian_norm_sum += self.compute_jacobian_norm_sum(inputs, outputs)

        epoch_sensitivity = float(jacobian_norm_sum / len(self.dataset))

        return epoch_sensitivity


if __name__ == '__main__':

    epoch = 200
    from model import resnet
    import utils
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainloader,testset,testloader,trainset_genuine = utils.load_data(
          128,
          32,
          4,
          dataset='CIFAR10',
          attack_set_size=0,
          binary=True)
    net = resnet.ResNet50(num_classes=1)
    net = net.to(device)

    Sensitivity_class = Sensitivity(net, trainset_genuine, device, epoch,128,4)
    print(Sensitivity_class.sensitivity_sigmoid())

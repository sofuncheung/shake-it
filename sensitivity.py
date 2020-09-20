# -*- coding: utf-8 -*-

"""
Author: Sofun Cheung
Date: 19 Sep 2020
Implement of input sensitivity calculation
"""

import torch
import torchvision
import numpy as np
from torch.autograd.gradcheck import zero_gradients

from config import config


class Sensitivity(object):

    def __init__(self, net, dataset, device):
        self.net = net
        self.trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.train_batch_size,
                shuffle=True, num_workers=config.num_workers)
        self.testloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.test_batch_size,
                shuffle=False, num_workers=config.num_workers)
        self.device = device


def compute_jacobian_norm(inputs, output):
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
    print(torch.Size(torch.norm(jacobian, dim=(1,2))))
    return torch.norm(jacobian, dim=(1,2))


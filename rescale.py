# -*- coding: utf-8 -*-
"""
alpha-scaling weights. Reference: Keskar 2017 paper
Because of the existence of Batch Normalization Layer,
we can rescale the weights of conv layers arbitrarily.

"""

import torch
from model import resnet


def rescale_layer(model, layer_str, block_idx, conv_layer_str, alpha):
    with torch.no_grad():
        # print('Before Rescaling:')
        # print(
        #     model._modules[
        #         layer_str][block_idx]._modules['conv1'].weight)
        # print(model.modules()[0])
        model._modules[layer_str][block_idx]._modules[conv_layer_str].weight = \
            torch.nn.Parameter(
                model._modules[layer_str][block_idx]._modules[conv_layer_str].weight
                * alpha)
        # print('After Rescaling:')
        # print(
        #     model._modules[layer_str][block_idx]._modules['conv1'].weight)


def rescale(model, layer_str, block_idx, conv_layer_str, alpha):
    if layer_str == 'all':
        for layer_idx, layer in model._modules.items():
            # In ResNet typical 2 2 2 2 structure,
            # the layer_idx will be 'layer1' ... 'layer4'
            if 'layer' in layer_idx:
                for b in range(len(layer)): # The layer here is a Sequential
                    for conv_layer_idx, conv_layer in layer[b]._modules.items():
                        # print(conv_layer_idx)
                        if 'conv' in conv_layer_idx:
                            rescale_layer(
                                    model, layer_idx, b, conv_layer_idx, alpha)
    else:
        rescale_layer(
                model, layer_str, blocl_idx, conv_layer_str, alpha)








if __name__ == '__main__':
    net = resnet.ResNet18()
    rescale(net, 'all', None, None, 5)

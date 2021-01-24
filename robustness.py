# -*- coding: utf-8 -*-

"""
Author: Sofun Cheung
Date: 22 Jan 2021
Implement of parameter robustness calculation
"""


import torch
from torch import nn, Tensor
import numpy as np
import copy
from torch.autograd.gradcheck import zero_gradients
from torch.autograd.functional import jacobian as Jacobian
from typing import List, Tuple, Dict, Union, Callable


def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])

def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)

def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names

def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)

def remove_loaded_weights(mod: nn.Module, names: List[str]) -> None:
    """
    Exactly counter the effect of load_weights.
    The difference between this and extract_weights is:
        extract_weights extracts weights from model.parameters(),
        but after load_weights the model.parameters() will still be empty.
    P.S. Turns out it's actually not necessary.
    """
    for name in names:
        _del_nested_attr(mod, name.split("."))


def load_params(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), torch.nn.Parameter(p))


class Robustness(object):
    '''
    Robustness: the average norm of the jocobian of model output
    (logits or sigmoid) w.r.t. model parameters.
    '''

    def __init__(self, net, dataset, device, test_batch_size=16, num_workers=4):
        self.net = copy.deepcopy(net)
        self.net.eval()
        self.dataset = dataset
        self.testloader = torch.utils.data.DataLoader(
                dataset, batch_size=test_batch_size,
                shuffle=False, num_workers=num_workers)
        self.device = device

    def robustness_logits(self):
        jacobian_norm_sum = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad = False
            params, names = extract_weights(self.net)

            def func(*params):
                # A callable for torch.autograd.functional.jacobian
                load_weights(self.net, names, params)
                out = self.net(inputs)
                return out

            jacobian = Jacobian(func, params, create_graph=False, strict=True)
            # Here params must be PURE Tensors, can't be nn.Parameter.
            load_params(self.net, names, params)




    def robustness_sigmoid(self):
        pass


if __name__ == '__main__':
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

    R = Robustness(net, trainset_genuine, device)
    r = R.robustness_logits()

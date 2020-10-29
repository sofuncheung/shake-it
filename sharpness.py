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
from scipy.optimize import minimize

from config import config
from l_bfgs_b_wrapper.obj import PyTorchObjective

def prepare_decorator(dataloader, loss):
    r'''
    Prepare necessary materials for the new
    'forward' method.

    Return:
        A decorator.

    Update:  This scenario is just not suitable to
             use decorator.

    '''
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print('Initializing decorator...')

    def change_forward_method(net):
        r'''
        This is a class decorator for pytorch Module (like net), 
        which changes the original 'forward' method.
        '''
        orig_forward = net.forward

        def new_forward(self):
            outputs = orig_forward(self, inputs)
            return -loss(outputs, targets)

        net.forward = new_forward

        return net

    return change_forward_method


class Sharpness(object):

    def __init__(self, net, loss, dataset, device):
        self.net = copy.deepcopy(net)
        self.loss = loss
        self.trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.sharpness_train_batch_size,
                shuffle=True, num_workers=config.num_workers,
                drop_last=True)
        self.testloader = torch.utils.data.DataLoader(
                dataset, batch_size=config.test_batch_size,
                shuffle=False, num_workers=config.num_workers,
                drop_last=True)  # Note this is not on the test set.
        
        self.full_batch_loader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset), shuffle=False,
            num_workers=config.num_workers
        )


        # self.optimizer = optim.SGD(net.parameters(), lr=1e-3) # Have to use vanilla SGD.
        self.device = device

    def clip_params(self, eps, params, new_params):
        for i in new_params:
            diff = new_params[i] - params[i]
            eps_mtx = eps * (torch.abs(params[i]) + 1) # mtx for matrix...
                                                # (I forget it myself after a while)
            is_out_of_bound = False
            outer_up = torch.nonzero(diff>eps_mtx, as_tuple=True)
            if len(outer_up[0]) != 0:
                is_out_of_bound = True
                # outer_up = [tuple(temp) for temp in outer_up] # This is where memory leak happens
                                                            # I did a few search and found there is
                                                            # inherent problem when doing type
                                                            # recast between tensor and python list.
                                                            # (Also potentially tuple)
                                                            # See https://ptorch.com/news/161.html

                # for _, j in enumerate(outer_up):          # This won't work as well
                #     diff[tuple(j)] = eps_mtx[tuple(j)]    # Sharpness-training still gets slower
                diff[outer_up] = eps_mtx[outer_up]

            outer_low = torch.nonzero(diff<-eps_mtx, as_tuple=True)
            if len(outer_low[0]) != 0:
                is_out_of_bound = True
                diff[outer_low] = -eps_mtx[outer_low]
            new_params[i] = params[i] + diff
            '''
            del diff, eps_mtx, outer_up, outer_low
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            else:
                gc.collect()
            '''
        return new_params

    def prepare_decorator_obsolete(self):
        '''
        This method passes necessary class attributes to the 
        true decorator 'change_forward_method(cls)'.

        Return:
            A decorator.

        UPDATE: After some thinking, I figure it just not gonna work when
        there is two 'self' within the same namespace.
        '''
        
        for batch_idx, (inputs, targets) in enumerate(self.full_batch_loader):
            outputs = net(inputs)
            ...

        def change_forward_method_obsolete(cls):
            r'''
            This is a pytorch Module decorator, which changes the
            original 'forward' method.
            '''
            orig_forward = cls.forward

            def new_forward_obsolete(self):
                ...


    def sharpness(self, clip_eps=5e-3, max_iter_epochs=100, opt_mtd='L-BFGS-B'):
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
        print('L_w: ', L_w)
        w = copy.deepcopy(net.state_dict())
        w = self.del_key_from_dic(w, 'num_batches_tracked')
        self.stop_tracking(w)
        max_value = 0
        max_value_list = []
        if opt_mtd == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=1e0)
            # Here lr should be large enough to make sure
            # we can find the maximum value. Don't worry
            # about the box limit. The box has been well
            # defined by the clip_eps.
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

                    new_w = copy.deepcopy(net.state_dict())
                    self.stop_tracking(new_w)
                    new_w = self.del_key_from_dic(new_w, 'num_batches_tracked')

                    new_w = self.clip_params(clip_eps, w, new_w)
                    # The above sentence might have caused the slowing down.
                    # A best place to start with narrowing down and debug.

                    # assert self._test_clip_is_effective(
                    #        clip_eps, w, new_w), 'Error: Fail Box!!!'

                    net.load_state_dict(new_w, strict=False)
                    #for value in net.state_dict().values():
                    #    print(value.requires_grad)
                    #sys.exit()

                    new_outputs = net(inputs)
                    epoch_loss += self.loss(new_outputs, targets).item()
                    print('Batch Loss:', self.loss(new_outputs, targets).item(), flush=True)
                epoch_loss = epoch_loss / (batch_idx+1)
                max_value = max(max_value, epoch_loss)
                print('max_value: ', max_value)
                max_value_list.append(max_value)

        elif opt_mtd == 'L-BFGS-B':
            '''
            Update:
                This problem (changing one method of a given class 'instance')
                is not suitable to be solved using decorator.
                Because decorators decorate the 'conceptual' function or class,
                not the instantiated version of them. That's also why we can use the
                syntatic sugar @ before 'definition' of function or class.
            '''
            @prepare_decorator(self.full_batch_loader,
                               self.loss)
            class Objective(net):
                pass

            objective = Objective()
            obj = PyTorchObjective(objective)
            
            results = optimize.minimize(obj.fun, obj.x0, jac=obj.jac,
                                        options={'gtol': 1e-6, 'maxiter':10})
            print('Results of L-BFGS-B: ', results)


        np.save(os.path.join(
            config.output_file_pth, 'max_value_list.npy'), max_value_list)
        sharpness = 100 * (max_value - L_w) / (1 + L_w)
        print('Sharpness:', sharpness)
        return sharpness


    @staticmethod
    def stop_tracking(w):
        for i in w:
            w[i].requires_grad_(False)
        return w


    @staticmethod
    def del_key_from_dic(dic, keyword):
        for i in dic.copy():
            if keyword in i:
                del dic[i]
        return dic

    @staticmethod
    def _test_clip_is_effective(eps, params, new_params, num_eps=1e-3):
        # Here the num_eps is a pretty vacuous bound for
        # normal small weights, but is necessitated by
        # the 6-significant-digits problem of float32 datatype.
        # This is not the best way of seeing whether the clip_parm
        # is effective or not. We can simply see the param before
        # and after clip_param.
        for i in new_params:
            if torch.max(
                    (torch.abs(new_params[i]-params[i])-
                    (eps*(torch.abs(params[i])+1))) > num_eps
                    ) > 0:
                where_out = torch.nonzero(
                        (torch.abs(new_params[i]-params[i])-
                    (eps*(torch.abs(params[i])+1))) > num_eps, as_tuple=True
                )
                print('Where difference go beyond box:')
                print(torch.abs(new_params[i]-params[i])[where_out]
                        )
                print(torch.abs(new_params[i]-params[i])[where_out][0].dtype)
                print('Box limits:')
                print((eps*(torch.abs(params[i])+1))[where_out])

                print('Difference minus Box:')
                print((torch.abs(new_params[i]-params[i])-
                    (eps*(torch.abs(params[i])+1)))[where_out])

                print('Whether it is large than num_eps:')
                print((torch.abs(new_params[i]-params[i])-
                    (eps*(torch.abs(params[i])+1)))[where_out] > num_eps)
                return False
        return True

    @staticmethod
    def _max_diff_minus_eps(eps, params, new_params):
        for i in new_params:
            print(
                torch.max(
                    torch.abs(new_params[i] - params[i])-eps*(torch.abs(params[i])+1)))
        return None

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
                print('\n', i, params[i])
                print('*'*88)
                print(i, new_params[i], '\n')

    @staticmethod
    def _print_w_shape(params):
        for i in params:
            print(params[i].shape)


    @staticmethod
    def _arrayify(X):
        return X.cpu().detach().contiguous().double().clone().numpy() 


if __name__ == '__main__':
    # print('CPUs:', os.cpu_count())
    from main import net, criterion, trainset, device
    S = Sharpness(net, criterion, trainset, device)
    print('Sharpness:', S.sharpness())


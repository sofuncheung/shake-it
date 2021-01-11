# -*- coding: utf-8 -*-

import os

class config(object):
    alpha = 1
    optim = 'Adam'  # Options: SGD, SGD+Momentum, Adam
    output_file_pth = ('/mnt/zfsusers/sofuncheung/shake-it/playground/volume')
    lr_decay = False
    train_epoch = 100000
    break_when_reaching_zero_error = True


    # Data loading
    num_workers = 4
    train_batch_size = 32
    test_batch_size = 100
    sharpness_train_batch_size = 32

    binary_dataset = True # This needs to be "True" when calculating volume.
    training_set_size = 10000


    # Sharpness & Sensitivity
    sharpness_method = 'SGD' # Option: 'SGD', 'L-BFGS-B'
    sharpness_cons = False # Consecutive recording sharpness.
    sharpness_one_off = True # Compute sharpness at last epoch.
    sensitivity_cons = False # Recording input sensitivity for each epoch of training.
    sensitivity_one_off = True # Compute input sensitivity at the end of traning.

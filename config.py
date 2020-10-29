# -*- coding: utf-8 -*-

import os

class config(object):
    alpha = 1
    optim = 'Adam'  # Options: SGD, SGD+Momentum, Adam
    output_file_pth = ('/mnt/zfsusers/sofuncheung/shake-it'
            '/playground/Sharpness/batch_size/BS-100/L-BFGS-B')
    lr_decay = False
    train_epoch = 100


    # Data loading
    num_workers = 4
    train_batch_size = 100
    test_batch_size = 100

    binary_dataset = True # This needs to be "True" when calculating volume.
    training_set_size = 10000


    # Sharpness & Sensitivity
    sharpness_cons = False # Consecutive recording sharpness.
    sharpness_one_off = True # Compute sharpness at last epoch.
    sensitivity_cons = False # Recording input sensitivity for each epoch of training.
    sensitivity_one_off = False # Compute input sensitivity at the end of traning.
    sharpness_train_batch_size = 256

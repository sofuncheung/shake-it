# -*- coding: utf-8 -*-

import os

class config(object):
    alpha = 1
    optim = 'SGD'  # Options: SGD, SGD+Momentum, Adam
    output_file_pth = '/mnt/zfsusers/sofuncheung/shake-it/playground/Sharpness'
    lr_decay = False
    num_workers = 4
    train_batch_size = 256
    test_batch_size = 128
    sharpness = True
    sensitivity_cons = False # Recording input sensitivity for each epoch of training.
    sensitivity_one_off = False # Compute input sensitivity at the end of traning.
    training_set_size = 50000

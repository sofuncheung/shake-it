# -*- coding: utf-8 -*-

import os

class config(object):
    alpha = 5
    optim = 'SGD'  # Options: SGD, SGD+Momentum, Adam
    output_file_pth = '/mnt/zfsusers/sofuncheung/shake-it/playground/pure-SGD/sharpness'
    lr_decay = False
    num_workers = 16
    train_batch_size = 128
    test_batch_size = 128
    sharpness = True

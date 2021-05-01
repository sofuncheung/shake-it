# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from collections import OrderedDict


class CNN(nn.Module):
    r'''
    CNN as in Guillermo's big paper: "Generalization bounds of deep learning"
    '''
    def __init__(self,
            image_height=28,
            image_width=28,
            num_channels=1,
            num_hidden_layers=4,
            num_filters=1024,
            pooling=None
            ):
        super(CNN, self).__init__()

        def insert_layer(layers, name, layer):
            # insert layer to the OrdererDict layers and put it to the last.
            layers.update({name: layer})
            layers.move_to_end(name, last=True)

        def width(l):
            # l in the index of conv layer before pooling, starting from 1
            # The obvservation is the width only decrease by 4 when it passes the 
            # conv layer with kernel (5,5).
            return image_width - 4 * ((l+1) // 2)

        filter_size_list = [(5,5),(2,2)]*(num_hidden_layers//2) + [(5,5)]*(num_hidden_layers%2)
        # padding_list = ["VALID", "SAME"]*(num_hidden_layers//2) + ["VALID"]*(num_hidden_layers%2)
        # valid and same padding are Keras convention.  
        # For 'valid' padding in Keras, the counterpart in Pytorch is padding=0
        # For 'same', we need a bit of calculation and in this case, only (2,2) kernel will be used
        # in 'same' padding. Because stride=1, we need padding=0.5, which is not supported by Pytorch.
        # Keras always pad bottom and right sides 1 more row of  pixels 
        # (ref: https://stackoverflow.com/questions/53819528/
        # how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave)

        if pooling != None:
            # This case we need to know the size of feature map to be able to use 'same' pooling
            for j in range(num_hidden_layers):
                assert width(j+1) % 2 == 0, "Odd width of feature map occurs! Not implemented."

        same_padding = nn.ZeroPad2d((0,1,0,1)) # left, right, top, bottom.
                                               # Note the value 1 is because of the kernel size (2,2)

        layers = OrderedDict()
        layers.update({'Conv1': nn.Conv2d(num_channels,num_filters,filter_size_list[0])}) # default padding=0
        insert_layer(layers, 'relu1', nn.ReLU())
        if pooling == 'avg':
            insert_layer(layers, 'padding_for_pooling1', nn.ZeroPad2d(int(width(1)/2)))
            insert_layer(layers, 'pooling_avg1', nn.AvgPool2d(kernel_size=2))
        elif pooling == 'max':
            insert_layer(layers, 'padding_for_pooling1', nn.ZeroPad2d(int(width(1)/2)))
            insert_layer(layers, 'pooling_max1', nn.MaxPool2d(kernel_size=2))

        if num_hidden_layers > 1:
            for i in range(num_hidden_layers-1):
                conv_name = 'Conv' + str(i+2)
                relu_name = 'relu' + str(i+2)
                padding_name = 'padding' + str(i//2+1)
                if i%2 == 0: # same padding
                    insert_layer(layers, padding_name, same_padding)
                insert_layer(layers, conv_name, nn.Conv2d(num_filters,num_filters,filter_size_list[i+1]))
                insert_layer(layers, relu_name, nn.ReLU())
                if pooling == 'avg':
                    pooling_name = 'pooling_avg' + str(i+2)
                    padding_for_pooling_name = 'padding_for_pooling' + str(i+2)
                    insert_layer(layers, padding_for_pooling_name, nn.ZeroPad2d(int(width(i+2)/2)))
                    insert_layer(layers, pooling_name, nn.AvgPool2d(kernel_size=2))
                elif pooling == 'max':
                    pooling_name = 'pooling_max' + str(i+2)
                    padding_for_pooling_name = 'padding_for_pooling' + str(i+2)
                    insert_layer(layers, padding_for_pooling_name, nn.ZeroPad2d(int(width(i+2)/2)))
                    insert_layer(layers, pooling_name, nn.MaxPool2d(kernel_size=2))
        # Global pooling
        if pooling == 'avg':
            pooling_name = 'global_pooling_avg'
            insert_layer(layers, pooling_name, nn.AvgPool2d(kernel_size=width(num_hidden_layers)))
        elif pooling == 'max':
            pooling_name = 'global_pooling_max'
            insert_layer(layers, pooling_name, nn.MaxPool2d(kernel_size=width(num_hidden_layers)))

        insert_layer(layers, 'flattening', nn.Flatten())
        if pooling == None:
            assert image_height == image_width, "Non-square pictures not implemented"
            insert_layer(layers, 'fc', nn.Linear(num_filters*width(num_hidden_layers)**2, 1))
        else:
            insert_layer(layers, 'fc', nn.Linear(num_filters, 1))

        self.layers = nn.Sequential(layers)

    def forward(self, x):
            return self.layers(x)

if __name__ == '__main__':
    net = CNN(num_hidden_layers=2, pooling='avg')
    print(net)

    X = torch.rand(16, 1, 28,28)
    print(net(X))

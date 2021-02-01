#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:14:14 2020

@author: shuofengzhang
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import sys

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

generalization = []
sharpness = []
robustness_logits = []
robustness_sigmoid = []
sensitivity_logits = []
sensitivity_sigmoid = []
robustness_logits_test = []
robustness_sigmoid_test = []
sensitivity_logits_test = []
sensitivity_sigmoid_test = []
volume = []

path = os.getcwd()
dir_list = []
for i in os.listdir(path):
    if os.path.isdir(i) and 'attack_' in i:
        dir_list.append(i)
dir_list.sort(key=natural_keys)
print('The dir_list order as:')
print(dir_list)


for dirs in dir_list:
    current_path = os.path.join(path, dirs)
    os.chdir(current_path)
    #print(os.getcwd())
    try:
        record = []
        sharpness_record = []
        for i in range(5):
            temp = np.load('record_%d.npy'%(i+1))
            record.append(temp)
            sharpness_record.append(np.load('sharpness_%d.npy'%(i+1)))
        record = np.array(record)
        sharpness_record = np.array(sharpness_record)
        sharpness.append(sharpness_record)
        generalization.append(record[:,0])
        sensitivity_logits.append(record[:,1])
        sensitivity_sigmoid.append(record[:,2])
        robustness_logits.append(record[:,3])
        robustness_sigmoid.append(record[:,4])
        sensitivity_logits_test.append(record[:,5])
        sensitivity_sigmoid_test.append(record[:,6])
        robustness_logits_test.append(record[:,7])
        robustness_sigmoid_test.append(record[:,8])
        record_sv = []
        for i in range(5):
            temp = np.load('record_sv_%d.npy'%(i+1))
            record_sv.append(temp)
        record_sv = np.array(record_sv)
        #sharpness.append(record_sv[:,0])
        volume.append(record_sv[:,1])
    except:
        print('Something\'s wrong here:')
        print(current_path)
        # sys.exit()

    os.chdir(path)


volume = np.array(volume).flatten()
generalization = np.array(generalization).flatten()
sharpness = np.array(sharpness).flatten()
#sharpness = np.load('sharpness.npy')
robustness_logits = np.array(robustness_logits).flatten()
robustness_sigmoid = np.array(robustness_sigmoid).flatten()
sensitivity_logits = np.array(sensitivity_logits).flatten()
sensitivity_sigmoid = np.array(sensitivity_sigmoid).flatten()
robustness_logits_test = np.array(robustness_logits_test).flatten()
robustness_sigmoid_test = np.array(robustness_sigmoid_test).flatten()
sensitivity_logits_test = np.array(sensitivity_logits_test).flatten()
sensitivity_sigmoid_test = np.array(sensitivity_sigmoid_test).flatten()


fig, ax = plt.subplots()
ax.scatter(-sharpness,generalization)
ax.set_xlabel(r'-$log_{10}$(sharpness)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.text(0.2,0.25,'ResNet50/CIFAR-10/\nAdam', bbox=dict(facecolor='none'), fontsize=20)
fig.savefig('sharpness-generalization-resnet50.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(sensitivity_logits,generalization)
ax.set_xlabel('Sensitivity (logits)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('sensitivity_logits-generalization-resnet50.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(sensitivity_sigmoid,generalization)
ax.set_xlabel('Sensitivity (activation)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('sensitivity_sigmoid-generalization-resnet50.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(robustness_logits,generalization)
ax.set_xlabel('Robustness (logits)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('robustness_logits-generalization-resnet50.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(robustness_sigmoid,generalization)
ax.set_xlabel('Robustness (activation)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('robustness_sigmoid-generalization-resnet50.png', dpi=300)
plt.clf()


fig, ax = plt.subplots()
ax.scatter(sensitivity_logits_test,generalization)
ax.set_xlabel('Sensitivity (logits)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('sensitivity_logits_test-generalization-resnet50.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(sensitivity_sigmoid_test,generalization)
ax.set_xlabel('Sensitivity (activation)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('sensitivity_sigmoid_test-generalization-resnet50.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(robustness_logits_test,generalization)
ax.set_xlabel('Robustness (logits)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('robustness_logits_test-generalization-resnet50.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(robustness_sigmoid_test,generalization)
ax.set_xlabel('Robustness (activation)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('robustness_sigmoid_test-generalization-resnet50.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(volume,generalization)
ax.set_xlabel(r'$log_{10}V(f)$',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization (%)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.text(0.5,0.25,'ResNet50/CIFAR-10/\nAdam', bbox=dict(facecolor='none'), fontsize=20)
fig.savefig('volume-generalization-resnet50.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(-sharpness,volume)
ax.set_xlabel(r'-$log_{10}$(sharpness)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel(r'$log_{10}V(f)$',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('sharpness-volume-resnet50.png', dpi=300)
plt.clf()





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
#robustness_logits = []
#robustness_sigmoid = []
sensitivity_logits = []
#sensitivity_sigmoid = []
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
        g_sh_se_v = []
        for i in range(5):
            temp = np.load('g_sh_se_v_%d.npy'%(i+1))
            g_sh_se_v.append(temp)
        g_sh_se_v = np.array(g_sh_se_v)
        generalization.append(g_sh_se_v[:,0])
        sharpness.append(g_sh_se_v[:,1])
        sensitivity_logits.append(g_sh_se_v[:,2])
        volume.append(g_sh_se_v[:,3])

    except:
        print('Something\'s wrong here:')
        print(current_path)
        # sys.exit()

    os.chdir(path)


volume = np.array(volume).flatten()
generalization = np.array(generalization).flatten()
sharpness = np.array(sharpness).flatten()
#robustness_logits = np.array(robustness_logits).flatten()
#robustness_sigmoid = np.array(robustness_sigmoid).flatten()
sensitivity_logits = np.array(sensitivity_logits).flatten()
#sensitivity_sigmoid = np.array(sensitivity_sigmoid).flatten()


fig, ax = plt.subplots()
ax.scatter(sharpness,generalization)
ax.set_xlabel(r'$log_{10}$(sharpness)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('sharpness-generalization.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(sensitivity_logits,generalization)
ax.set_xlabel('sensitivity_logits',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('sensitivity_logits-generalization.png', dpi=300)
plt.clf()
'''
fig, ax = plt.subplots()
ax.scatter(sensitivity_sigmoid,generalization)
ax.set_xlabel('sensitivity_sigmoid',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('sensitivity_sigmoid-generalization.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(robustness_logits,generalization)
ax.set_xlabel('robustness_logits',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('robustness_logits-generalization.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(robustness_sigmoid,generalization)
ax.set_xlabel('robustness_sigmoid',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('robustness_sigmoid-generalization.png', dpi=300)
plt.clf()
'''
fig, ax = plt.subplots()
ax.scatter(volume,generalization)
ax.set_xlabel(r'$log_{10}(V_f)$',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel('Generalization',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('volume-generalization.png', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.scatter(sharpness,volume)
ax.set_xlabel(r'$log_{10}$(sharpness)',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.set_ylabel(r'$log_{10}(V_f)$',fontdict={'fontsize': 16, 'fontweight': 'medium'})
ax.tick_params(direction='in')
fig.savefig('sharpness-volume.png', dpi=300)
plt.clf()





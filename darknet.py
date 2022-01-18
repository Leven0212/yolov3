# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: darknet.py
# Date: 2022/1/17 0017

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    :param cfgfile: configuration name
    :return: Returns a list of blocks. Each blocka describe a block in neural network to be built.
    Block is represented as a dictionary in the list
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                 # store the lines in a list
    lines = [x for x in lines if len(x) > 0]        # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']       # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]    # get rid of fringe whitespaces

    blocks = []
    block = {}

    for line in lines:
        if line[0] == '[':                      # This marks the start of a new block
            if len(block) != 0:                 # If block is not empty, implies it is storing values of previous block
                blocks.append(block)            # add it into the blocks list
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]                    # Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_fitters = 3
    output_fitters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block
        # create a new module for the block
        # append to module_list

        if x['type'] == 'convolutional':
            # Get the info about the convolutional layer
            activation = x['activation']
            try:
                batch_normalize = x['batch_normalize']
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the Convolutional layer
            conv = nn.Conv2d(prev_fitters, filters, kernel_size, stride, pad, bias)
            module.add_module('conv_{}'.format(index), conv)

            # Add the batch Normalize layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{}'.format(index), bn)

            # Check the activation
            # It is either Linear or Leaky Relu for YOLO
            if activation == 'Leaky':
                Leaky = nn.LeakyReLU()
                module.add_module('leaky_{}'.format(index), Leaky)

        # If it's a upsampling layer
        # we use UpsamplingBilinear2d
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.UpsamplingBilinear2d(stride)
            module.add_module('upsample_{}'.format(index), upsample)

        # If it's a route layer
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except:
                end = 0
            if start > 0:
                start = start - index






if __name__ == '__main__':
    blocks = parse_cfg('./cfg/yolov3.cfg')
    create_modules(blocks)
    print(blocks)

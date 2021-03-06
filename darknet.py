# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: darknet.py
# Date: 2022/1/17 0017

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from util import predict_transform

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
            if activation == 'leaky':
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
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module('route_{}'.format(index), route)
            if end < 0:
                filters = output_fitters[index + start] + output_fitters[index + end]
            else:
                filters = output_fitters[index + start]

        # shortcut corresponds to skip connection
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('short_cut_{}'.format(index), shortcut)

        # Yolo is the detection layer
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{}'.format(index), detection)

        module_list.append(module)
        prev_fitters = filters
        output_fitters.append(filters)

    return net_info, module_list


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}    # cache the outputs for the route layer

        write = 0
        for i, module in enumerate(modules):
            module_type = module['type']
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(n) for n in layers]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i - 1] + outputs[i + from_]
        return x

if __name__ == '__main__':
    x = torch.randn((15, 3, 416, 416))
    net = Darknet('./cfg/yolov3.cfg')
    net.forward(x, False)

# !/usr/bin/env python
# -*- coding:utf-8 -*-　
# Author: leven
# File: util.py
# Date: 2022/1/18 0018

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = False):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
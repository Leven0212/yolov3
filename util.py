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
    pass
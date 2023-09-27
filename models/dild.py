# -*- coding: utf-8 -*-
# @Time    : 2023/8/23 16:57
# @Author  : yblir
# @File    : dild.py
# explain  : 
# =======================================================
import sys
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.snippet import *
except:
    from snippet import *


class STILModel(nn.Module):
    def __init__(self,
                 num_class=2,
                 # num_segment=8,
                 u=3,
                 t=4,
                 add_softmax=False,
                 **kwargs):
        """ Model Builder for STIL model.
        STIL: Spatiotemporal Inconsistency Learning for DeepFake Video Detection (https://arxiv.org/abs/2109.01860)

        Args:
            num_class (int, optional): Number of classes. Defaults to 2.
            num_segment (int, optional): Number of segments (frames) fed to the model. Defaults to 8.
            add_softmax (bool, optional): Whether to add softmax layer at the end. Defaults to False.
        """
        super().__init__()

        self.num_class = num_class
        # self.num_segment = num_segment
        self.u = 4
        self.t = 5
        self.add_softmax = add_softmax

        self.build_model()

    def build_model(self):
        """
        Construct the model.
        """
        self.base_model = scnet50_v1d(self.u, self.t, pretrained=False)

        fc_feature_dim1 = self.base_model.fc1.in_features
        # fc_feature_dim2 = self.base_model.fc2.in_features
        self.base_model.fc1 = nn.Linear(fc_feature_dim1, 2 * 64)
        # self.base_model.fc2 = nn.Linear(2 * 64, self.num_class)

        if self.add_softmax:
            self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.tensor): input tensor of shape (n, t*c, h, w). n is the batch_size, t is num_segment
        """
        # img channel default to 3
        img_channel = 3

        # x: [n, tc, h, w] -> [nt, c, h, w]
        # out: [nt, num_class]
        out = self.base_model(
                x.view((-1, img_channel) + x.size()[2:])
        )

        out = out.view(-1, self.num_segment * 5, self.num_class)  # [n, t, num_class]
        out = out.mean(1, keepdim=False)  # [n, num_class]

        if self.add_softmax:
            out = self.softmax_layer(out)

        return out

    def set_segment(self, num_segment):
        """Change num_segment of the model.
        Useful when the train and test want to feed different number of frames.

        Args:
            num_segment (int): New number of segments.
        """
        self.num_segment = num_segment

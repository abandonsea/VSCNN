#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:26 2021

@author: Pedro Vieira
@description: Implement network architecture of the VSCNN for the Indian Pines dataset
"""

import torch
from torch import nn


class VSCNN(nn.Module):
    def __init__(self, bands, num_classes):
        # image_patch: 13x13
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(1, 20, (9, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            nn.Dropout3d(0.05),
            nn.Conv3d(20, 40, (9, 3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.05),
            nn.Linear((bands - 16) * 40, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, num_classes)
        )

    def forward(self, x):
        # Input size: [batch_size, 1, depth, h, w]
        out = self.feature(x)
        batch_size = out.shape[0]
        out = out.view((batch_size, -1))
        out = self.classifier(out)
        return out

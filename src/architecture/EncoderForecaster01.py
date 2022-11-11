#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   encoder.py
@Time    :   2020/03/09 18:47:50
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   encoder
'''

# adapted from https://github.com/jhhuang96/ConvLSTM-PyTorch


from torch import nn
from utils import make_layers
import torch
import logging
from src.layers.ConvLSTM2D import CLSTM_cell

class Encoder(nn.Module):
  def __init__(self):
    super.__init__()
    stage_1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding='same'),
      nn.LeakyReLU(negative_slope=.2, inplace=True),
    )
    rnn_1 = CLSTM_cell(shape=(64, 64), input_channels=16, num_features=64, filter_size=5)
    stage_2 = nn.Sequential(
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding='same'),
      nn.LeakyReLU(negative_slope=.2, inplace=True),
    )
    rnn_2 = CLSTM_cell(shape=(32, 32), input_channels=64, num_features=96, filter_size=5)
    stage_3 = nn.Sequential(
      nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=2, padding='same'),
      nn.LeakyReLU(negative_slope=.2, inplace=True),
    )
    rnn_3 = CLSTM_cell(shape=(16, 16), input_channels=96, num_features=96, filter_size=5)
    self.model = nn.Sequential(
      
    )
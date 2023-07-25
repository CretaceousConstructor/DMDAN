import numpy as np
import torch
# 导入 PyTorch 内置的 mnist 数据
from torchvision.datasets import mnist
#导入预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


class GeneratorNet(nn.Module):

    #生成器  传入的参数                   1  22 625 4
    def __init__(self, _depth_size, _electrode_size, _timepoint_size, _cls_num):
        super(GeneratorNet, self).__init__()
        #TODO：L2正则化，源代码中没有使用？
        #nn.Sequential
        self.g_conv_temp = nn.Conv2d(in_channels=_depth_size, out_channels=30, kernel_size=(1, 25), stride=1, padding='VALID')
        self.g_conv_spatial = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=(_electrode_size, 1), stride=1, padding='VALID')
        self.batch_norm_0 = nn.BatchNorm2d(30)# bn设置的参数实际上是channel的参数
        self.pooling_0 = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        #TODO:重新计算这里的维度
        self.g_fc1 = nn.Linear(1080, 64)
        self.g_fc2 = nn.Linear(64, 16)
        #TODO:自定义RELU，需要注意，如果自定义的激活函数不可导，则需要
        #https://www.shuzhiduo.com/A/ke5jEv7mJr/
        self.activate_fn = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.g_output = nn.Linear(16, _cls_num)
        self.softmax = nn.softmax

    def forward(self, x):
        net = self.g_conv_temp(x)
        net = self.g_conv_spatial(net)
        net = self.batch_norm_0(net)
        net = self.pooling_0(net)
        fc = nn.Flatten(net)

        net1 = self.g_fc1(fc)
        net2 = self.g_fc2(net1)
        net2 = self.activate_fn(net2)

        net = self.g_output(net1)
        pred = self.softmax(net)
        return net, pred, net1, net2

class DiscriminatorNet(nn.Module):
    def __init__(self, fea_dim):
        super(DiscriminatorNet, self).__init__()
        self.activate_fn = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.d_fc1 = nn.Linear(fea_dim, 64)
        self.d_fc2 = nn.Linear(64, 32)
        self.d_fc3= nn.Linear(32, 16)
        self.d_out_logits= nn.Linear(16, 1)

    def forward(self, x):
        x = self.d_fc1(x)
        x = self.activate_fn(x)

        x = self.d_fc2(x)
        x = self.activate_fn(x)

        x = self.d_fc3(x)
        x = self.activate_fn(x)

        d_out_logits_output = self.d_out_logits(x)

        #x = torch.sigmoid(d_out_logits_output)
        x = nn.Sigmoid(d_out_logits_output)
        return x, d_out_logits_output

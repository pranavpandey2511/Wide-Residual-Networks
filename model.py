import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class BasicBlock(nn.Module):

    def __init__(self, in_feat_maps, out_feat_maps, stride = 1, width_factor = 1,  dropout_rate = 0.3, skip_conn_downsample = None):

        super(BasicBlock, self).__init__()
        self._bn1 = nn.BatchNorm2d(num_features = in_feat_maps)
        self._relu1 = nn.ReLU()
        self._conv1 = nn.Conv2d(in_channels = in_feat_maps, out_channels = out_feat_maps * width_factor, stride = 1, kernel_size = 3, padding = 1)

        self._drop1 = nn.Dropout2d(p=dropout_rate)

        self._bn2 = nn.BatchNorm2d(num_features = out_feat_maps * width_factor)
        self._relu2 = nn.ReLU()
        self._conv2 = nn.Conv2d(in_channels = out_feat_maps * width_factor, out_channels = out_feat_maps * width_factor, stride = stride, kernel_size = 3, padding = 1)
        self._skip_conn_downsample = skip_conn_downsample


    def forward(self, x):
        residual = x
        # print("Residual shape:", residual.shape)
        x = self._bn1(x)
        x = self._relu1(x)
        x = self._conv1(x)
        x = self._drop1(x)
        x = self._bn2(x)
        x = self._relu2(x)
        x = self._conv2(x)
        # print("X shape:", x.shape)
        if self._skip_conn_downsample:
            x += self._skip_conn_downsample(residual)

        return x

def _make_group(in_feat_maps, out_feat_maps, N, stride =1, width_factor = 10, dropout_rate = 0.3):
    layers = []
    for i in range(N):
        # print(in_feat_maps)
        skip_conn_downsample = None

        if (stride != 1 or in_feat_maps != out_feat_maps * width_factor):
            skip_conn_downsample = nn.Conv2d(in_feat_maps, out_feat_maps * width_factor, kernel_size=1, stride=stride)

        layers.append(BasicBlock(in_feat_maps, out_feat_maps, stride = stride, width_factor = width_factor, dropout_rate = dropout_rate, skip_conn_downsample = skip_conn_downsample))
        # print("Block appended")
        in_feat_maps = out_feat_maps * width_factor
        stride = 1
    block = nn.Sequential(*layers)

    return block

class WideResNet(nn.Module):

    def __init__(self, dropout_rate, num_classes, width_factor = 10, N = 1):
        super(WideResNet, self).__init__()
        self._conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self._res_block1 = _make_group(in_feat_maps = 16, out_feat_maps = 16, N=N, stride = 1, width_factor = width_factor, dropout_rate = dropout_rate)
        self._res_block2 = _make_group(in_feat_maps = 16 * width_factor,  out_feat_maps = 32, N=N, stride=2, width_factor = width_factor, dropout_rate = dropout_rate)
        self._res_block3 = _make_group(in_feat_maps = 32 * width_factor,  out_feat_maps = 64, N=N, stride=2, width_factor = width_factor, dropout_rate = dropout_rate)
        self._avg_pool = nn.AvgPool2d(kernel_size = 8)
        self._bn1 = nn.BatchNorm2d(num_features = 64 * width_factor)
        self._relu = nn.ReLU()
        self._linear = nn.Linear(640, num_classes)


    def forward(self, x):
        x = self._conv1(x)
        x = self._res_block1(x)
        x = self._res_block2(x)
        x = self._res_block3(x)
        x = self._relu(self._bn1(x))
        x = self._avg_pool(x)
        x = x.view(x.size(0), -1)
        output = self._linear(x)

        return output



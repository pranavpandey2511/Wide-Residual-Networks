import torch
import torch.nn as nn
import numpy as np
class BasicBlock(nn.Module):

    def __init__(self, in_feat_maps, out_feat_maps, stride = 1, width_factor = 1, equate_planes = None):

        super(BasicBlock, self).__init__()
        self._bn1 = nn.BatchNorm2d(num_features = in_feat_maps)
        self._relu1 = nn.ReLU()
        self._conv1 = nn.Conv2d(in_channels = in_feat_maps, out_channels = out_feat_maps * width_factor, stride = stride, kernel_size = 3, padding = 1)

        self._drop1 = nn.Dropout2d(p=0.3)

        self._bn2 = nn.BatchNorm2d(num_features = out_feat_maps * width_factor)
        self._relu2 = nn.ReLU()
        self._conv2 = nn.Conv2d(in_channels = out_feat_maps * width_factor, out_channels = out_feat_maps * width_factor, stride = stride, kernel_size = 3, padding = 1)
        self.equate_planes = equate_planes


    def forward(self, x):
        residual = x
        #print("Residual shape:", residual.shape)
        x = self._bn1(x)
        x = self._relu1(x)
        x = self._conv1(x)
        x = self._drop1(x)
        x = self._bn2(x)
        x = self._relu2(x)
        x = self._conv2(x)
        #print("X shape:", x.shape)
        if self.equate_planes:
            residual = self.equate_planes(residual)
        x += residual

        return x

def _make_layer(in_feat_maps, out_feat_maps, width_factor = 10):
    equate_planes = None 
    if in_feat_maps != out_feat_maps * width_factor:
       equate_planes = nn.Sequential(nn.Conv2d(in_feat_maps, out_feat_maps * width_factor, kernel_size=1, stride=1), nn.BatchNorm2d(out_feat_maps * width_factor))
    layer = BasicBlock(in_feat_maps, out_feat_maps, stride = 1, width_factor = width_factor, equate_planes = equate_planes)

    return layer



class WideResNet(nn.Module):

    def __init__(self, width_factor = 10):
        super(WideResNet, self).__init__()
        self._conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1)
        self._res_block1 = _make_layer(in_feat_maps = 16, out_feat_maps = 16, width_factor = width_factor)
        self._res_block2 = _make_layer(in_feat_maps = 16 * width_factor,  out_feat_maps = 32, width_factor = width_factor)
        self._res_block3 = _make_layer(in_feat_maps = 32 * width_factor,  out_feat_maps = 64, width_factor = width_factor)
        self._avg_pool = nn.AvgPool2d(kernel_size = 8)
        self._relu = nn.ReLU()
        self._linear = nn.Linear(5760, 10)        


    def forward(self, x):
        x = self._conv1(x)
        x = self._res_block1(x)
        x = self._res_block2(x)
        x = self._res_block3(x)
        x = self._avg_pool(x)
        x = self._relu(x)
        x = torch.flatten(x, 1)
        output = self._linear(x)

        return output



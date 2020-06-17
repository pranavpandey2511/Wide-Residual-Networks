import torch
import torch.nn as nn
from functools import partial
import torch.nn.init as init
import numpy as np

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)



class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3)


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class WideResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1,
                      stride=self.downsampling),
            nn.BatchNorm2d(self.out_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def bn_relu_conv(in_channels, out_channels, conv=conv3x3, *args, **kwargs):
    return nn.Sequential(nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         conv3x3(in_channels, out_channels, *args, **kwargs))


class BasicBlock(WideResNetResidualBlock):
    """
    Basic WideResNet block composed of two layers of batchnorm/activation/3x3conv with dropout
    """
    # expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            bn_relu_conv(self.in_channels, self.out_channels, conv=self.conv, stride=1),
            nn.Dropout2d(p=0.3),
            bn_relu_conv(self.out_channels, self.expanded_channels, conv=self.conv, stride=self.downsampling),
        )


class WideResNetLayer(nn.Module):
    """
    A WideResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=BasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels * kwargs["expansion"], *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * kwargs["expansion"],
                    out_channels * kwargs["expansion"], downsampling=1, *args, **kwargs) for _ in range(n - 1)],
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class WideResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[16, 16, 32, 64], depths=[4, 4, 4],
                 activation='relu', block=BasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            WideResNetLayer(blocks_sizes[0], blocks_sizes[1], n=depths[0], activation=activation,
                        block=block,*args, **kwargs),
            *[WideResNetLayer(in_channels * kwargs["expansion"],
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes[1:], depths[1:])]
        ])
        self.bn1 = nn.BatchNorm2d(self.blocks_sizes[-1]*kwargs["expansion"])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        x = self.relu(self.bn1(x))
        return x


class WideResnetDecoder(nn.Module):
    """
    This class represents the tail of WideResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class WideResNet(nn.Module):

    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = WideResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = WideResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

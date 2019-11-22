import torch
import torch.nn as nn

from selection import *

__all__ = ['EPE']

SELECTION = {'selection1': Selection1, 'selection2': Selection2, 
             'selection3': Selection3, 'selection4': Selection4}


def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class EPE(nn.Module):
    """
    EPE Module
    """
    def __init__(self, channels, final_channels=512, stride=2, expansion=2, num_classes=100, cond_type='selection', selection_type='selection1'):
        super(EPE, self).__init__()
        if cond_type == 'selection':
            self._selection = SELECTION[selection_type](num_classes, final_channels)
        else:
            self._selection = None
            
        # activation func
        self._relu = nn.ReLU(inplace=True)
        
        # expansion module
        mid_channels = channels * expansion
        self._expansion_conv = conv1x1(channels, mid_channels, stride=1)
        self._bn0 = nn.BatchNorm2d(mid_channels)
        
        # conv module
        self._depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=stride, stride=stride, padding=1, bias=True, groups=mid_channels)
        self._bn1 = nn.BatchNorm2d(mid_channels)
        
        self._projection_conv = conv1x1(mid_channels, final_channels, stride=1)
        self._bn2 = nn.BatchNorm2d(final_channels)
        
        # classifier module
        self._globalavgpool = nn.AdaptiveAvgPool2d(1)
        self._shallow_classifier = nn.Linear(final_channels, num_classes)

        
    def forward(self, x):
        # conv
        x = self._relu(self._bn0(self._expansion_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))
        features = self._relu(self._bn2(self._projection_conv(x)))

        # classifier
        features = self._globalavgpool(features)
        features = features.view(x.size(0), -1)
        
        logits = self._shallow_classifier(features)
        
        if self._selection:
            selection = self._selection(logits, features)
        else:
            selection = logits
            
        return logits, features, selection
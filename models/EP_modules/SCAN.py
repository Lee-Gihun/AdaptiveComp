import torch
import torch.nn as nn

from selection import *

__all__ = ['SCAN']

SELECTION = {'selection1': Selection1, 'selection2': Selection2, 
             'selection3': Selection3, 'selection4': Selection4}


def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class SCAN(nn.Module):
    def __init__(self, channels, final_channels=512, stride=2, num_classes=100, cond_type='selection', selection_type='selection1'):
        super(SCAN, self).__init__()
        if cond_type == 'selection':
            self._selection = SELECTION[selection_type](num_classes, final_channels)
        else:
            self._selection = None
            
        # activation func
        self._relu = nn.ReLU(inplace=True)
        
        # attention module
        self._attconv = conv3x3(channels, channels, stride=2, padding=1)
        self._attbn0 = nn.BatchNorm2d(channels)
        self._attdeconv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self._attbn1 = nn.BatchNorm2d(channels)
        
        # bottleneck module
        self._bot1x1_0 = conv1x1(channels, 128, stride=1)
        self._botbn0 = nn.BatchNorm2d(128)
        self._bot3x3 = nn.Conv2d(128, 128, kernel_size=stride, stride=stride, bias=True)
        self._botbn1 = nn.BatchNorm2d(128)
        self._bot1x1_1 = conv1x1(128, final_channels, stride=1)
        self._botbn2 = nn.BatchNorm2d(final_channels)
        
        # classifier module
        self._globalavgpool = nn.AvgPool2d(4, 4)
        self._shallow_classifier = conv1x1(final_channels, num_classes, stride=1)


    def forward(self, x):
        # attention
        att = self._relu(self._attbn0(self._attconv(x)))
        att = self._relu(self._attbn1(self._attdeconv(att)))
        x = x * torch.sigmoid(att)
        
        # bottleneck
        x = self._relu(self._botbn0(self._bot1x1_0(x)))
        x = self._relu(self._botbn1(self._bot3x3(x)))
        
        features = self._relu(self._botbn2(self._bot1x1_1(x)))
        features = self._globalavgpool(features)
        
        logits = self._shallow_classifier(features).squeeze()
        features = features.view(x.size(0), -1)
        
        if self._selection:
            selection = self._selection(logits, features).squeeze()
        else:
            selection = logits
        
        return logits, features, selection
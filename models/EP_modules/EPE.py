import torch
import torch.nn as nn

__all__ = ['EPE']


class EPE(nn.Module):
    """
    EPE Module
    """
    def __init__(self, channels, b_stride=2, final_channels=512, expansion=2, num_classes=100, selection=False):
        super(EPE, self).__init__()
        
        if selection:
            self._selection = SelectionLayer(final_channels)
        self.selection = selection
            
        # activation func
        self._relu = nn.ReLU(inplace=True)
        
        # expansion module
        mid_channels = channels * expansion
        self._expansion_conv = nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, bias=True)
        self._bn0 = nn.BatchNorm2d(mid_channels)
        
        # conv module
        self._depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=b_stride, stride=b_stride, padding=1, bias=True, groups=mid_channels)
        self._bn1 = nn.BatchNorm2d(mid_channels)
        
        self._projection_conv = nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1, bias=True)
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
        
        selection = None
        
        if self.selection:
            selection = self._selection(features)
            
        x = self._shallow_classifier(features)
        
        return x, features, selection
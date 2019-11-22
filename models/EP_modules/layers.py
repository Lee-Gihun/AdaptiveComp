import torch
import torch.nn as nn

__all__ = ['ProbCond', 'SCAN', 'EPE']

class ProbCond(nn.Module):
    """
    from the softmax outputs, decides whether the samples are above or below threshold.
    """
    def __init__(self, thres=1.0):
        super(ProbCond, self).__init__()
        self.thres = thres
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        probs = self.softmax(outputs)
        max_prob, _ = torch.max(probs, dim=1)
        
        cond_up = (max_prob > self.thres)
        cond_down = (max_prob <= self.thres)
        
        return cond_up, cond_down


class SelectionLayer(nn.Module):
    def __init__(self, features=512):
        self._selector = nn.ModuleList([
            nn.Linear(features, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
            nn.Sigmoid()
        ])
        
    def forward(self, x):
        x = self._selector(x)
        return x
    
    
class SCAN(nn.Module):
    def __init__(self, channels, b_stride=2, final_channels=512, num_classes=100, selection=False):
        super(SCAN, self).__init__()
        
        if selection:
            self._selection = SelectionLayer(final_channels)
        
        self.selection = selection
            
        # activation func
        self._relu = nn.ReLU(inplace=True)
        
        # attention module
        self._attconv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=True)
        self._attbn0 = nn.BatchNorm2d(channels)
        self._attdeconv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=True)
        self._attbn1 = nn.BatchNorm2d(channels)
        
        # bottleneck module
        self._bot1x1_0 = nn.Conv2d(channels, 128, kernel_size=1, stride=1, bias=True)
        self._botbn0 = nn.BatchNorm2d(128)
        self._bot3x3 = nn.Conv2d(128, 128, kernel_size=b_stride, stride=b_stride, bias=True)
        self._botbn1 = nn.BatchNorm2d(128)
        self._bot1x1_1 = nn.Conv2d(128, final_channels, kernel_size=1, stride=1, bias=True)
        self._botbn2 = nn.BatchNorm2d(final_channels)
        
        # classifier module
        self._globalavgpool = nn.AvgPool2d(4, 4)
        self._shallow_classifier = nn.Conv2d(final_channels, num_classes, kernel_size=1, stride=1, bias=True)


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
        
        selection = None
        
        if self.selection:
            selection = self._selection(features).squeeze()
        
        x = self._shallow_classifier(features).squeeze()
        features = features.view(x.size(0), -1)
        
        return x, features, selection

    
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
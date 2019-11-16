import torch
import torch.nn as nn

__all__ = ['LogitCond', 'SCAN', 'EPE']

class LogitCond(nn.Module):
    """
    from the softmax outputs, decides whether the samples are above or below threshold.
    """
    def __init__(self, thres=1.0):
        super(LogitCond, self).__init__()
        self.thres = thres
        self.softmax = nn.Softmax(dim=1)

    def forward(self, outputs):
        logits = self.softmax(outputs)
        max_logits, _ = torch.max(logits, dim=1)
        
        cond_up = (max_logits > self.thres)
        cond_down = (max_logits <= self.thres)
        
        return cond_up, cond_down

    
class SL_Pair(nn.Module):
    def __init__(self, SmallNet, LargeNet, num_classes=100, exit_cond=0, device='cuda:0'):
        super(SL_Pair, self).__init__()
        self.SmallNet = SmallNet.to(device)
        self.LargeNet = LargeNet.to(device)
        self.num_classes = num_classes
        self._exit_cond = LogitCond(exit_cond)
        self.device = device
    
    def condition_updater(self, exit_cond):
        self._exit_cond.thres = exit_cond[0]
        
    def forward(self, x):
        outputs = torch.zeros(x.size(0), self.num_classes).to(self.device)
        mark = torch.zeros(x.size(0)).long().to(self.device)
        small_out = self.SmallNet(x)
        condition_up, condition_down = self._exit_cond(small_out)
        
        outputs[condition_up] = small_out[condition_up]
        mark[condition_up] = -1

        
        if (condition_down.sum().item() == 0) and (not self.training):
            return outputs, mark
        
        large_out = self.LargeNet(x[condition_down])
        outputs[condition_down] = large_out
        
        return outputs, mark

    
class SCAN(nn.Module):
    def __init__(self, channels, stride=(1,1,1), final_channels=512, num_classes=100):
        super(SCAN, self).__init__()
        
        # activation func
        self._relu = nn.ReLU(inplace=True)
        
        # attention module
        self._attconv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self._attbn0 = nn.BatchNorm2d(channels)
        self._attdeconv = nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self._attbn1 = nn.BatchNorm2d(channels)
        
        # bottleneck module
        self._bot1x1_0 = nn.Conv2d(channels, channels, kernel_size=1, stride=stride[0], bias=False)
        self._botbn0 = nn.BatchNorm2d(channels)
        self._bot3x3 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self._botbn1 = nn.BatchNorm2d(channels)
        self._bot1x1_1 = nn.Conv2d(channels, final_channels, kernel_size=1, stride=stride[2], bias=False)
        self._botbn2 = nn.BatchNorm2d(final_channels)
        
        # classifier module
        self._globalavgpool = nn.AdaptiveAvgPool2d(1)
        self._shallow_classifier = nn.Conv2d(final_channels, num_classes, kernel_size=1, stride=1, bias=True)
    
    def forward(self, x):
        # attention
        att = self._relu(self._attbn0(self._attconv(x)))

        att = self._relu(self._attbn1(self._attdeconv(att, x.shape)))
        x = x * torch.sigmoid(att)
        
        # bottleneck
        x = self._relu(self._botbn0(self._bot1x1_0(x)))
        x = self._relu(self._botbn1(self._bot3x3(x)))
        features = self._relu(self._botbn2(self._bot1x1_1(x)))
        #print(feature.shape)
        # classifier
        x = self._globalavgpool(features)
        x = self._shallow_classifier(x).squeeze()
        
        return x, features
    
    
class EPE(nn.Module):
    """
    EPE Module
    """
    def __init__(self, channels, stride=(1,1,1), final_channels=512, expansion=2, num_class=100):
        super(EPE, self).__init__()
        
        # activation func
        self._relu = nn.ReLU(inplace=True)
        
        # expansion module
        mid_channels = channels * expansion
        self._expansion_conv = nn.Conv2d(channels, mid_channels, kernel_size=1, stride=stride[0], bias=False)
        self._bn0 = nn.BatchNorm2d(mid_channels)
        
        # conv module
        self._depthwise_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride[1], padding=1, bias=False, groups=channels)
        self._bn1 = nn.BatchNorm2d(mid_channels)
        
        self._projection_conv = nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=stride[2], bias=False)
        self._bn2 = nn.BatchNorm2d(final_channels)
        
        # classifier module
        self._globalavgpool = nn.AdaptiveAvgPool2d(1)
        self._shallow_classifier = nn.Conv2d(final_channels, num_class, kernel_size=1, stride=1, bias=True)
    
    def forward(self, x):
        # conv
        x = self._relu(self._bn0(self._expansion_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))
        features = self._relu(self._bn2(self._projection_conv(x)))

        # classifier
        x = self._globalavgpool(features)
        x = self._shallow_classifier(x).squeeze()
        return x, features
    
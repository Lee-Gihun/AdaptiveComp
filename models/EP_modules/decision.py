import torch
import torch.nn as nn

__all__ = ['Decision1', 'Decision2', 'Decision3', 'Decision4', 'Decision5', 'Decision6']


class Decision1(nn.Module):
    def __init__(self, class_num=100):
        super(Decision1, self).__init__()
        self._fc1 = nn.Linear(class_num, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, featuremap, logits):
        x = self._fc1(logits)
        x = self._sigmoid(x)
        return x

    
class Decision2(nn.Module):
    def __init__(self, class_num=100):
        super(Decision2, self).__init__()
        self._relu = nn.ReLU(inplace=True)
        self._fc1 = nn.Linear(class_num, 100)
        self._fc2 = nn.Linear(100, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, featuremap, logits):
        x = self._fc1(logits)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x

    
class Decision3(nn.Module):
    def __init__(self, class_num=100, features=512):
        super(Decision3, self).__init__()
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._relu = nn.ReLU(inplace=True)
        self._fc1 = nn.Linear(features, 100)
        self._fc2 = nn.Linear(100, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, featuremap, logits):
        x = self._pool(featuremap).squeeze()
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x
    
    
class Decision4(nn.Module):
    def __init__(self, class_num=100, features=512):
        super(Decision4, self).__init__()
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._relu = nn.ReLU(inplace=True)
        self._fc1 = nn.Linear(features+class_num, 100)
        self._fc2 = nn.Linear(100, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, featuremap, logits):
        pooled_map = self._pool(featuremap).squeeze()
        x = torch.cat((pooled_map, logits), dim=1)
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x
    
class Decision5(nn.Module):
    def __init__(self, class_num=100, features=512):
        super(Decision5, self).__init__()
        self._conv = nn.Conv2d(features, 100, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn = nn.BatchNorm2d(100)
        self._relu = nn.ReLU(inplace=True)
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._fc1 = nn.Linear(100, 100)
        self._fc2 = nn.Linear(100, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, featuremap, logits):
        x = self._relu(self._bn(self._conv(featuremap)))
        x = self._pool(x).squeeze()
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x
    
    
class Decision6(nn.Module):
    def __init__(self, class_num=100, features=512):
        super(Decision6, self).__init__()
        self._conv = nn.Conv2d(features, 100, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn = nn.BatchNorm2d(100)
        self._relu = nn.ReLU(inplace=True)
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._fc1 = nn.Linear(100+class_num, 100+class_num)
        self._fc2 = nn.Linear(100+class_num, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, featuremap, logits):
        x = self._relu(self._bn(self._conv(featuremap)))
        x = self._pool(x).squeeze()
        x = torch.cat((x, logits), dim=1)
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x
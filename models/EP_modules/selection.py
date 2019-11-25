import torch
import torch.nn as nn

__all__ = ['Selection1', 'Selection2', 'Selection3', 'Selection4', 'Selection5', 'Selection6', 'Selection7']


class Selection1(nn.Module):
    def __init__(self, num_classes=100, planes=512):
        super(Selection1, self).__init__()
        self._fc1 = nn.Linear(num_classes, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, logits, features):
        x = self._fc1(logits)
        x = self._sigmoid(x)
        return x

    
class Selection2(nn.Module):
    def __init__(self, num_classes=100, planes=512):
        super(Selection2, self).__init__()
        self._relu = nn.ReLU(inplace=True)
        self._fc1 = nn.Linear(num_classes, 100)
        self._fc2 = nn.Linear(100, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, logits, features):
        x = self._fc1(logits)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x

    
class Selection3(nn.Module):
    def __init__(self, num_classes=100, planes=512):
        super(Selection3, self).__init__()
        self._relu = nn.ReLU(inplace=True)
        self._fc1 = nn.Linear(planes, 100)
        self._fc2 = nn.Linear(100, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, logits, features):
        x = self._fc1(features)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x
    
    
class Selection4(nn.Module):
    def __init__(self, num_classes=100, planes=512):
        super(Selection4, self).__init__()
        self._relu = nn.ReLU(inplace=True)
        self._fc1 = nn.Linear(planes + num_classes, 100)
        self._fc2 = nn.Linear(100, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, logits, features):
        x = torch.cat((features, logits), dim=1)
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x

    
class Selection5(nn.Module):
    def __init__(self, num_classes=100, planes=512):
        super(Selection5, self).__init__()
        self._fc1 = nn.Linear(5, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, logits, features):
        logits = logits.topk(k=5, largest=True, sorted=True)[0]
        x = self._fc1(logits)
        x = self._sigmoid(x)
        return x

    
class Selection6(nn.Module):
    def __init__(self, num_classes=100, planes=512):
        super(Selection6, self).__init__()
        self._relu = nn.ReLU(inplace=True)
        self._fc1 = nn.Linear(5, 5)
        self._fc2 = nn.Linear(5, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, logits, features):
        logits = logits.topk(k=5, largest=True, sorted=True)[0]
        x = self._fc1(logits)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x
    
    
class Selection7(nn.Module):
    def __init__(self, num_classes=100, planes=512):
        super(Selection7, self).__init__()
        self._relu = nn.ReLU(inplace=True)
        self._fc1 = nn.Linear(planes + 5, 100)
        self._fc2 = nn.Linear(100, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, logits, features):
        logits = logits.topk(k=5, largest=True, sorted=True)[0]
        x = torch.cat((features, logits), dim=1)
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x
    
"""
class Selection5(nn.Module):
    def __init__(self, num_classes=100, planes=512):
        super(Selection5, self).__init__()
        self._conv = nn.Conv2d(planes, 100, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn = nn.BatchNorm2d(100)
        self._relu = nn.ReLU(inplace=True)
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._fc1 = nn.Linear(100, 100)
        self._fc2 = nn.Linear(100, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, logits, features):
        x = self._relu(self._bn(self._conv(features)))
        x = self._pool(x).squeeze()
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x
    
    
class Selection6(nn.Module):
    def __init__(self, num_classes=100, planes=512):
        super(Selection6, self).__init__()
        self._conv = nn.Conv2d(planes, 100, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn = nn.BatchNorm2d(100)
        self._relu = nn.ReLU(inplace=True)
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._fc1 = nn.Linear(100 + num_classes, 100 + num_classes)
        self._fc2 = nn.Linear(100 + num_classes, 1)
        self._sigmoid = nn.Sigmoid()
        
    def forward(self, logits, features):
        x = self._relu(self._bn(self._conv(features)))
        x = self._pool(x).squeeze()
        x = torch.cat((x, logits), dim=1)
        x = self._fc1(x)
        x = self._relu(x)
        x = self._fc2(x)
        x = self._sigmoid(x)
        return x
"""
import torch
import torch.nn as nn
from .ResNet import ResNet, BasicBlock, Bottleneck
from .EP_modules import *

import copy

__all__ = ['ep_resnet18', 'ep_resnet34', 'ep_resnet50']

class EP_ResNet(ResNet):
    def __init__(self, block, layers, exit_module=None, start_mark=0, num_classes=100):
        super(EP_ResNet, self).__init__(block, layers, num_classes)
        self.start_mark = start_mark

        self.exit_module = nn.ModuleList(
            [exit_module(channels=64, stride=(1,1,1)),
             exit_module(channels=128, stride=(1,1,1)),
             exit_module(channels=256, stride=(1,1,1))])

        self.exit_cond = [LogitCond(1), LogitCond(1), LogitCond(1)]

            
    def condition_updater(self, exit_cond):
        for i, cond in enumerate(self.exit_cond):
            cond.thres = exit_cond[i]
    
    
    def _early_predictor(self, x, empty_indices, idx):
        exit, features = self.exit_module[idx](x)
        with torch.no_grad():
            cond_up, cond_down = self.exit_cond[idx](exit)
            
            fill_indices = copy.deepcopy(empty_indices)
            fill_indices[fill_indices][cond_down] = False
            
            empty_indices[empty_indices][cond_up] = False
            x = x[cond_down]
        
        return x, features, exit, fill_indices, empty_indices
        
        
    def forward(self, x):
        device = str(self.conv1.weight.device)
        outputs = torch.zeros(x.size(0), 100).to(device)
        mark = torch.zeros(x.size(0)).long().to(device)
        empty_indices = torch.ones(x.size(0)).bool().to(device)
        
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x, features1, exit1, fill_indices, empty_indices = self._early_predictor(x, empty_indices, 0)
        outputs[fill_indices], mark[fill_indices] = exit1, 0
        
        if x.size(0) == 0:
            return outputs, mark

        x = self.layer2(x)
        x, features2, exit2, fill_indices, empty_indices = self._early_predictor(x, empty_indices, 1)
        outputs[fill_indices], mark[fill_indices] = exit2, 1
        if x.size(0) == 0:
            return outputs, mark  
        
        x = self.layer3(x)
        x, features3, exit3, fill_indices, empty_indices = self._early_predictor(x, empty_indices, 2)
        outputs[fill_indices], mark[fill_indices] = exit3, 2
        if x.size(0) == 0:
            return outputs, mark
        
        features4 = self.layer4(x)
        
        x = self.avgpool(features4)
        x = torch.flatten(x, 1)
        exit4 = self.fc(x)
        outputs[empty_indices] = exit4
        mark[empty_indices] = 3
        
        if not self.training:
            return outputs, mark
        
        else:
            return [exit1, exit2, exit3, exit4], [features1, features2, features3, features4]


def _resnet(block, layers, exit_module, **kwargs):
    if exit_module == 'scan':
        return EP_ResNet(block, layers, SCAN, **kwargs)
    elif exit_module == 'epe':
        return EP_ResNet(block, layers, EPE, **kwargs)

    
def ep_resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ep_resnet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ep_resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
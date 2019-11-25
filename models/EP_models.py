import copy
import torch
import torch.nn as nn

from .EP_modules import *
from .ResNet import *


__all__ = ['EP_Model']

BACKBONE = {'resnet10': resnet10, 'resnet18': resnet18, 'resnet34':resnet34, 
            'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152, 
            'resnet50_32x4d': resnet50_32x4d, 'resnet101_32x8d': resnet101_32x8d,
            'wide_resnet50_2': wide_resnet50_2, 'wide_resnet101_2': wide_resnet101_2}

EP = {'scan': SCAN, 'epe': EPE}

SELECTION = {'selection1': Selection1, 'selection2': Selection2, 
             'selection3': Selection3, 'selection4': Selection4}


class EP_Model(nn.Module):
    """
    General Early Prediction Model for every backbone model and early_prediction module
    
    [args] (str)  BackboneNet : backbone network name
           (str)  EPNet       : early prediction network name
           (dict) param       : opt.model dictionary should be here. It has 'num_classes', 'exit_block_pos', 'ep'.
                                For 'exit_block_pos', if None : don't use Early_prediction modules and only use BackboneNet
                                                      if have 0 value : Small-Large network architecutre
                                                      if have i value : exit module after block i, etc.
    """
    def __init__(self, BackboneNet, EPNet, param):
        super(EP_Model, self).__init__()
        self.BackboneNet = BACKBONE[BackboneNet](num_classes=param.num_classes)
        
        if not param.exit_block_pos:
            # Trick not to use EPNet
            param.exit_block_pos = []
        else:
            assert max(param.exit_block_pos) < len(self.BackboneNet.planes), 'Exit path is bigger than block number of BackboneNet'
        self.exit_block_pos = param.exit_block_pos
        self.use_small = True if 0 in param.exit_block_pos else False
        ep_modules = []
        # 3 channels for Small-Large network architecture 
        channels = self.BackboneNet.planes + [3]
        for idx, pos in enumerate(param.exit_block_pos):
            ep_modules.append(EP[EPNet](channels=channels[pos - 1],
                                        final_channels=self.BackboneNet.planes[-1],
                                        stride=param.ep.stride[idx],
                                        num_classes=param.num_classes,
                                        conf_type=param.ep.conf_type, 
                                        selection_type=param.ep.selection_type))
        self.EPNet = nn.ModuleList(ep_modules)
        
        assert param.ep.conf_type in ['sr', 'selection'], 'Confidence type must be sr or selection'
        
        if param.ep.conf_type == 'selection':
            self.selection = SELECTION[param.ep.selection_type](param.num_classes, self.BackboneNet.planes[-1])
        else:
            self.selection = None
            self.softmax = nn.Softmax(dim=1)
            
        self.num_classes = param.num_classes
    
    def forward(self, x):
        """
        mark value : 0 for Samll network, value bigger than 0 means each exit path number.
        len(exit_outpus), len(exit_features) : should be same with len(exit_block_pos) + 1(backbone network output)
        """
        exit_logits, exit_features, exit_confidence = [], [], []
        if not self.use_small:
            x = self.BackboneNet.conv_stem(x)
            
        for idx, pos in enumerate(self.exit_block_pos):
            # Small-Large network architecture
            if pos == 0:
                logits, features, confidence = self.EPNet[idx](x)
                
                exit_logits.append(logits)
                exit_features.append(features)
                exit_confidence.append(confidence)
                
                x = self.BackboneNet.conv_stem(x)
                
            # early prediction for each exit_block_pos
            else:
                start_block = 0 if idx == 0 else self.exit_block_pos[idx-1]
                for b in range(start_block, pos):
                    x = self.BackboneNet.block_layers[b](x)
                
                logits, features, confidence = self.EPNet[idx](x)
                
                exit_logits.append(logits)
                exit_features.append(features)
                exit_confidence.append(confidence)
                
        # forward for remained block and pool_linear layer
        start_block = self.exit_block_pos[-1] if self.exit_block_pos else 0
        for b in range(start_block, len(self.BackboneNet.planes)):
            x = self.BackboneNet.block_layers[b](x)
        
        logits, features = self.BackboneNet.pool_linear(x)
        
        if self.selection:
            confidence = self.selection(logits, features)
        else:
            probs = self.softmax(logits)
            confidence, _ = torch.max(probs, dim=1)
            confidence = confidence.unsqueeze(1)
            
        exit_logits.append(logits)
        exit_features.append(features)
        exit_confidence.append(confidence)
        
        return exit_logits, exit_features, exit_confidence
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
                                For 'exit_block_pos', if 0 : Small-Large network architecutre
                                                   elif 1 : exit module after block 1, etc.
    """
    def __init__(self, BackboneNet, EPNet, param):
        super(EP_Model, self).__init__()
        self.BackboneNet = BACKBONE[BackboneNet](num_classes=param.num_classes)
        
        assert max(param.exit_block_pos) < len(self.BackboneNet.planes), 'Exit path is bigger than block number of BackboneNet'
        self.exit_block_pos = param.exit_block_pos
        self.use_small = True if 0 in param.exit_block_pos else False
        ep_modules = []
        for idx, _ in enumerate(param.exit_block_pos):
            ep_modules.append(EP[EPNet](channels=self.BackboneNet.planes[idx],
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
    
    def ensemble_prediction(self, logits, features, hard_indices):
        if self.selection:
            confidence = self.selection(logits, features)
        else:
            probs = self.softmax(logits)
            confidence, _ = torch.max(probs, dim=1)
            
        with torch.no_grad():
            
            easy_indices = copy.deepcopy(hard_indices)
            
            easy_indices[cond_down] = False
            hard_indices[cond_up] = False
            
        return easy_indices, hard_indices
    
    def forward(self, x):
        """
        mark value : 0 for Samll network, value bigger than 0 means each exit path number.
        len(exit_outpus), len(exit_features) : should be same with len(exit_block_pos) + 1(backbone network output)
        """
        device = str(self.BackboneNet.conv1.weight.device)
        outputs = torch.zeros(x.size(0), self.num_classes).to(device)
        mark = torch.zeros(x.size(0)).long().to(device)
        hard_indices = torch.ones(x.size(0), dtype=torch.bool).to(device)
        
        exit_logits, exit_features, exit_confidence = [], [], []
        if not self.use_small:
            x = self.BackboneNet.conv_stem(x)
            
        for idx, pos in enumerate(self.exit_block_pos):
            # Small-Large network architecture
            if pos == 0:
                logits, features, confidence = self.EPNet[idx](x)
                
                exit_logits.append(copy.deepcopy(logits.detach()))
                exit_features.append(copy.deepcopy(features.detach()))
                exit_confidence.append(copy.deepcopy(confidence.detach()))
                
                x = self.conv_stem(x)
                
            # early prediction for each exit_block_pos
            else:
                start_block = 0 if idx == 0 else self.exit_block_pos[idx-1]
                for b in range(start_block, pos):
                    x = self.BackboneNet.block_layers[b](x)
                
                logits, features, confidence = self.EPNet[idx](x)
                
                exit_logits.append(copy.deepcopy(logits.detach()))
                exit_features.append(copy.deepcopy(features.detach()))
                exit_confidence.append(copy.deepcopy(confidence.detach()))
                
        # forward for remained block and pool_linear layer
        start_block = self.exit_block_pos[-1]
        for b in range(start_block, len(self.BackboneNet.planes)):
            x = self.BackboneNet.block_layers[b](x)
        
        logits, features = self.BackboneNet.pool_linear(x)
        
        if self.selection:
            confidence = self.selection(logits, features)
        else:
            probs = self.softmax(logits)
            confidence, _ = torch.max(probs, dim=1)
            
        exit_logits.append(copy.deepcopy(logits.detach()))
        exit_features.append(copy.deepcopy(features.detach()))
        exit_confidence.append(copy.deepcopy(confidence.detach()))
        
        return exit_logits, exit_features, exit_confidence
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
                                        cond_type=param.ep.cond_type, 
                                        selection_type=param.ep.selection_type))
        self.EPNet = nn.ModuleList(ep_modules)
        
        assert param.ep.cond_type in ['sr', 'selection'], 'Condition type must be sr or selection'
        
        # +1 is for BackboneNet ExitCond layer
        self.exit_cond = [ExitCond(1, param.ep.cond_type)] * (len(param.exit_block_pos) + 1)
        
        if param.ep.cond_type == 'selection':
            self.selection = SELECTION[param.ep.selection_type](param.num_classes, self.BackboneNet.planes[-1])
        else:
            self.selection = None
            
        self.num_classes = param.num_classes
        
    def condition_updater(self, thres):
        for i, cond in enumerate(self.exit_cond):
            cond.thres = thres[i]
    
    def early_prediction(self, x, hard_indices, idx):
        logits, features, selection = self.EPNet[idx](x)
        with torch.no_grad():
            cond_up, cond_down = self.exit_cond[idx](selection)
            
            easy_indices = copy.deepcopy(hard_indices)
            
            easy_indices[easy_indices][cond_down] = False
            hard_indices[hard_indices][cond_up] = False
            
            x = x[cond_down]
            
        return x, features, logits, easy_indices, hard_indices
    
    def ensemble_prediction(self, logits, features, hard_indices):
        if self.selection:
            selection = self.selection(logits, features)
        else:
            selection = logits
            
        with torch.no_grad():
            cond_up, cond_down = self.exit_cond[-1](selection)
            
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
        
        exit_logits, exit_features = [], []
        if not self.use_small:
            x = self.BackboneNet.conv_stem(x)
            
        for idx, pos in enumerate(self.exit_block_pos):
            # Small-Large network architecture
            if pos == 0:
                x, features, logits, easy_indices, hard_indices = self.early_prediction(x, hard_indices, idx)
                
                outputs[easy_indices], mark[easy_indices] = copy.deepcopy(logits.detach()), idx
                
                exit_logits.append(copy.deepcopy(logits.detach()))
                exit_features.append(copy.deepcopy(features.detach()))
                
                if (x.size(0) == 0) and not self.training:
                    return outputs, mark
                
                x = self.conv_stem(x)
                
            # early prediction for each exit_block_pos
            else:
                start_block = 0 if idx == 0 else self.exit_block_pos[idx-1]
                for b in range(start_block, pos):
                    x = self.BackboneNet.block_layers[b](x)
                    
                x, features, logits, easy_indices, hard_indices = self.early_prediction(x, hard_indices, idx)
                
                outputs[easy_indices], mark[easy_indices] = copy.deepcopy(logits.detach()), idx
                
                exit_logits.append(copy.deepcopy(logits.detach()))
                exit_features.append(copy.deepcopy(features.detach()))
                
                if (x.size(0) == 0) and not self.training:
                    return outputs, mark
                
        # forward for remained block and pool_linear layer
        start_block = self.exit_block_pos[-1]
        for b in range(start_block, len(self.BackboneNet.planes)):
            x = self.BackboneNet.block_layers[b](x)
        
        logits, features = self.BackboneNet.pool_linear(x)
        print(hard_indices.sum().item())
        easy_indices, hard_indices = self.ensemble_prediction(logits, features, hard_indices)
                       
        outputs[easy_indices], mark[easy_indices] = logits, len(self.exit_block_pos)
        
        exit_logits.append(logits)
        exit_features.append(features)
        
        if (hard_indices.sum().item() == 0) and not self.training:
            return outputs, mark
                       
        outputs[hard_indices], mark[hard_indices] = torch.mean(torch.stack(exit_logits, dim=1), dim=1), (len(self.exit_block_pos) + 1)
                       
        if not self.training:
            return outputs, mark
        
        return exit_logits, exit_features
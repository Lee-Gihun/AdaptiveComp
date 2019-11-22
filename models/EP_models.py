import copy
import torch
import torch.nn as nn

from .ResNet2 import *
from .EP_modeuls import *

BACKBONE = {'resnet10': resnet10, 'resnet18': resnet18, 'resnet34':resnet34, 
            'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152, 
            'resnet50_32x4d': resnet50_32x4d, 'resnet101_32x8d': resnet101_32x8d,
            'wide_resnet50_2': wide_resnet50_2, 'wide_resnet101_2': wide_resnet101_2}

EP = {'scan': SCAN, 'epe': EPE}


class EP_Model(nn.Module):
    """
    General Early Prediction Model for every backbone model and early_prediction module
    
    [args] (str)  BackboneNet : backbone network name
           (str)  EPNet       : early prediction network name
           (dict) param       : opt.model dictionary should be here. It has 'num_classes', 'exit_blocks', 'backbone', 'ep'.
                                For 'exit_blocks', if 0 : Small-Large network architecutre
                                                   elif 1 : exit module after block 1, etc.
    """
    def __init__(self, BackboneNet, EPNet, exit_blocks, param, device='cuda:0'):
        super(EP_Model, self).__init__()
        self.BackboneNet = BACKBONE[BackboneNet](num_classes=param.num_classes,
                                                 **param.backbone).to(device)
        
        assert max(param.exit_blocks) < len(self.BackboneNet.planes), 'Exit path is bigger than block number of BackboneNet'
        self.exit_blocks = param.exit_blocks
        self.use_small = True if 0 in param.exit_blocks else False
        ep_modules = []
        for idx, _ in enumerate(param.exit_blocks):
            ep_modules.append(EP[EPNet](channels=self.BackboneNet.planes[idx],
                                        final_channels=self.BackboneNet.planes[-1],
                                        num_classes=param.num_classes,
                                        **param.ep).to(device))
        self.EPNet = nn.ModuleList(ep_modules)
        self.exit_cond = [ProbCond(1)] * param.exit_blocks
            
        self.num_classes = param.num_classes
        self.device = device
        
    def condition_updater(self, thres):
        for i, cond in enumerate(self.exit_cond):
            cond.thres = thres[i]
    
    def early_prediction(self, x, hard_indices, idx):
        logits, features = self.EPNet[idx](x)
        with torch.no_grad():
            cond_up, cond_down = self.exit_cond[idx](exit)
            
            easy_indices = copy.deepcopy(hard_indices)
            
            easy_indices[cond_down] = False
            hard_indices[cond_up] = False
            
            x = x[cond_down]
            
        return x, features, logits, easy_indices, hard_indices
    
    def forward(self, x):
        """
        mark value : 0 for Samll network, value bigger than 0 means each exit path number.
        len(exit_outpus), len(exit_features) : should be same with len(exit_blocks) + 1(backbone network output)
        """
        device = str(self.conv1.weight.device)
        outputs = torch.zeros(x.size(0), self.num_classes).to(device)
        mark = torch.zeros(x.size(0)).long().to(device)
        hard_indices = torch.ones(x.size(0), dtype=torch.bool).to(device)
        
        exit_logits, exit_features = [], []
        if not self.use_small:
            x = self.conv_stem(x)
            
        for idx, exit_block in enumerate(self.exit_blocks):
            # Small-Large network architecture
            if exit_block == 0:
                x, features, logits, easy_indices, hard_indices = self.early_prediction(x, hard_indices, idx)
                
                outputs[easy_indices], mark[easy_indices] = copy.deepcopy(logits), idx
                
                exit_logits.append(copy.deepcopy(logits))
                exit_features.append(copy.deepcopy(features))
                
                if (x.size(0) == 0) and not self.training:
                    return outputs, mark
                
                x = self.conv_stem(x)
                
            # early prediction for each exit_blocks
            else:
                start_block = 0 if idx == 0 else self.exit_blocks[idx-1]
                for b in range(start_block, exit_block):
                    x = self.BackboneNet.block_layers[b](x)
                    
                x, features, logits, easy_indices, hard_indices = self.early_prediction(x, hard_indices, idx)
                
                outputs[easy_indices], mark[easy_indices] = copy.deepcopy(logits), idx
                
                exit_logits.append(copy.deepcopy(logits))
                exit_features.append(copy.deepcopy(features))
                
                if (x.size(0) == 0) and not self.training:
                    return outputs, mark
                
        # forward for last block and pool_linear layer
        x = self.BackboneNet.block_layers[-1](x)
        
        logits, features = self.pool_linear(x)
    
        outputs[hard_indices], mark[hard_indices] = logits, len(self.exit_blocks)
        
        exit_logits.append(logits)
        exit_features.append(features)
        
        if not self.training:
            return outputs, mark
        
        return exit_logits, exit_features
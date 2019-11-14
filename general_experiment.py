# -*- coding: utf-8 -*-
import os, sys
os.environ['OMP_NUM_THREADS'] = '1'
import math, copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.models as models

from data_utils import *
from train_tools import *
from models import *
    
DATASETTER = {'cifar10': cifar_10_setter,
              'cifar100': cifar_100_setter}
    
CRITERION = {'mse': nn.MSELoss,
             'cross_entropy': nn.CrossEntropyLoss,
             'soft_smoothing': SoftSmoothingLoss,
             'label_smoothing': LabelSmoothingLoss,
             'soft_label_smoothing': SoftLabelSmoothingLoss,
            'rand_smoothing': RandSmoothingLoss}

OPTIMIZER = {'sgd': optim.SGD,
             'adam': optim.Adam}

SCHEDULER = {'step': lr_scheduler.StepLR,
             'multistep': lr_scheduler.MultiStepLR,
             'cosine': lr_scheduler.CosineAnnealingLR}

MODEL = {
    'ResNet8' : resnet8,
    'ResNet10' : resnet10,
    'ResNet18': resnet18,
    'ResNet101': resnet101,
    'DenseNet40': densenet40,
    'DenseNet121': densenet121,
    'MobileNetV2': mobilenet_v2}

def _get_dataset(param):
    dataloaders, dataset_sizes = DATASETTER[param.dataset](batch_size=param.batch_size, 
                                                           valid_size=param.valid_size,
                                                           root=param.root,
                                                           fixed_valid=param.fixed_valid)
    return dataloaders, dataset_sizes


def _get_model(opt):
    param = opt.model.param
    model = MODEL[param.model_type](num_classes=param.num_classes)
    
    #if param.pre_trained_path != 'None':
    #    model.load_state_dict(torch.load(param.pre_trained_path+'./trained_model.pth'))
        
    return model


def _get_warm_trainhandler(opt, model, dataloaders, dataset_sizes):
    warm_criterion = CRITERION[opt.trainhandler.warm_up.criterion.algo](**opt.trainhandler.warm_up.criterion.param)
    warm_optimizer = OPTIMIZER[opt.trainhandler.warm_up.optimizer.algo](model.parameters(), **opt.trainhandler.warm_up.optimizer.param)
    
    if opt.trainhandler.warm_up.scheduler.enabled:
        warm_scheduler = SCHEDULER[opt.scheduler.type](optimizer, **opt.trainhandler.warm_up.scheduler.param)
    else:
        warm_scheduler = None

    train_handler = TrainHandler(model,
                                 dataloaders, 
                                 dataset_sizes, 
                                 warm_criterion, 
                                 warm_optimizer, 
                                 warm_scheduler, 
                                 device=opt.trainhandler.device, 
                                 path=opt.trainhandler.path)
    
    train_handler.set_name(opt.trainhandler.name+'_warmup')

    
    return train_handler


def _get_trainhanlder(opt, model, dataloaders, dataset_sizes, train_handler=None):
    criterion = CRITERION[opt.criterion.algo](**opt.criterion.param)
    optimizer = OPTIMIZER[opt.optimizer.algo](model.parameters(), **opt.optimizer.param)
    
    if opt.scheduler.enabled:
        scheduler = SCHEDULER[opt.scheduler.type](optimizer, **opt.scheduler.param)
    else:
        scheduler = None

        
    train_handler = TrainHandler(model, 
                                 dataloaders, 
                                 dataset_sizes, 
                                 criterion, 
                                 optimizer, 
                                 scheduler,
                                 mixup=opt.trainhandler.mixup,
                                 rand_smooth=opt.trainhandler.rand_smooth,
                                 device=opt.trainhandler.device, 
                                 path=opt.trainhandler.path)
    
    train_handler.set_name(opt.trainhandler.name)
    
    return train_handler


def run(opt):
    """runs the overall process"""
    dataloaders, dataset_sizes = _get_dataset(opt.data)
    
    model = _get_model(opt)
    train_handler = None
    
    if opt.trainhandler.warm_up.enabled:
        train_handler = _get_warm_trainhandler(opt, model, dataloaders, dataset_sizes)
        train_handler.train_model(num_epochs=opt.trainhandler.warm_up.num_epochs)
        model = train_handler.model
        
    train_handler = _get_trainhanlder(opt, model, dataloaders, dataset_sizes, train_handler)
    train_handler.train_model(num_epochs=opt.trainhandler.num_epochs)
    
    train_handler.test_model()
        
if __name__ == "__main__":
    # gets arguments from the json file
    opt = ConfLoader(sys.argv[1]).opt
    
    # make experiment reproducible
    if opt.trainhandler.get('seed', None):
        torch.manual_seed(opt.trainhandler.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(opt.trainhandler.seed)
    
    run(opt)
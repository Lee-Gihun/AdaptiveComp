import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from data_utils import *
from train_tools import *
from models import *

DATASETTER = {'cifar10': cifar_10_setter,
              'cifar100': cifar_100_setter}

CRITERION = {'epe': EPELoss,
             'scan': SCANLoss}
        
OPTIMIZER = {'sgd': optim.SGD}

SCHEDULER = {'step': lr_scheduler.StepLR,
            'multistep': lr_scheduler.MultiStepLR,
            'cosine': lr_scheduler.CosineAnnealingLR}


def _get_dataset(param):
    dataloaders, dataset_sizes = DATASETTER[param.dataset](root=param.root, 
                                                           batch_size=param.batch_size, 
                                                           valid_size=param.valid_size)
    
    return dataloaders, dataset_sizes


def _get_model(opt):
    param = opt.model.param
    model = EP_Model(BackboneNet=opt.model.BackboneNet,
                     EPNet=opt.model.EPNet, 
                     param = opt.model.param)
    
    return model


def _get_trainhandler(opt, model, dataloaders, dataset_sizes):
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
                                 device=opt.trainhandler.device, 
                                 path=opt.trainhandler.path,
                                 early_pred=opt.trainhandler.early_pred)
    
    # early_pred should be True if load EP_Model
    if opt.trainhandler.early_pred:
        train_handler.set_prediction(ep_prediction)
        
    train_handler.set_name(opt.trainhandler.name)
    
    return train_handler


def _get_inspectionhandler(opt, model, dataloaders, dataset_sizes):
    inspection_handler = InspectionHandler(model, 
                                           dataloaders, 
                                           dataset_sizes, 
                                           num_path=opt.inspectionhandler.num_path, 
                                           path_cost=opt.inspectionhandler.path_cost, 
                                           phase=opt.inspectionhandler.phase,
                                           device=opt.trainhandler.device)
    
    return inspection_handler

def _get_visualizer(opt, model, dataloaders, dataset_sizes):
    visualization_handler = VisualizationHandler(model, 
                                                 dataloaders, 
                                                 dataset_sizes,
                                                 num_path=opt.inspectionhandler.num_path,
                                                 phase=opt.inspectionhandler.phase,
                                                 device=opt.trainhandler.device)
    return visualization_handler


def run(opt):
    dataloaders, dataset_sizes = _get_dataset(opt.data)
    model = _get_model(opt)
    
    train_handler = _get_trainhandler(opt, model, dataloaders, dataset_sizes)
    
    if opt.model.pretrained.enabled:
        fpath = opt.model.pretrained.fpath
        pretrained_dict = torch.load(os.path.join(fpath, 'trained_model.pth'), map_location=opt.trainhandler.device)
        train_handler.model.load_state_dict(pretrained_dict, strict=False)
    else:
        train_handler.train_model(num_epochs=opt.trainhandler.num_epochs)
    
    train_handler.test_model()
    print()
    
    if opt.inspectionhandler.enabled:  
        inspection_handler = _get_inspectionhandler(opt, model, dataloaders, dataset_sizes)
        inspection_handler.set_name(opt.trainhandler.name)
        inspection_handler.save_inspection()
        print()
        
    if opt.visualizationhandler.enabled:
        visualization_handler = _get_visualizer(opt, model, dataloaders, dataset_sizes)
        visualization_handler.set_name(opt.trainhandler.name)
        visualization_handler.visualizer(mode=opt.visualizationhandler.mode, **opt.visualizationhandler.param)
        print()
    
if __name__ == "__main__":
    # gets arguments from the json file
    opt = ConfLoader(sys.argv[1]).opt
    
    # make experiment reproducible
    if opt.trainhandler.get('seed', None):
        torch.manual_seed(opt.trainhandler.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(opt.trainhandler.seed)
        
    run(opt)
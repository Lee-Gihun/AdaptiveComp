import sys
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from data_utils import *
from train_tools import *
from models.EP_models import *
from models.EP_modules.criterion import *
from models.EP_modules.prediction import *

DATASETTER = {'cifar10': cifar_10_setter,
              'cifar100': cifar_100_setter}

CRITERION = {'epe': EPELoss}
             #'hard_smoothing': HardSmoothingLoss,
             #'soft_smoothing': SoftSmoothingLoss}

OPTIMIZER = {'sgd': optim.SGD}

SCHEDULER = {'step': lr_scheduler.StepLR,
            'multistep': lr_scheduler.MultiStepLR,
            'cosine': lr_scheduler.CosineAnnealingLR}

MODEL = {'ep_resnet18': ep_resnet18}


def _get_dataset(param):
    dataloaders, dataset_sizes = DATASETTER[param.dataset](root=param.root, batch_size=param.batch_size, valid_size=param.valid_size)
    return dataloaders, dataset_sizes

def _get_model(opt):
    param = opt.model.param
    model = MODEL[param.model_type](exit_module=param.exit_module, start_mark=param.start_mark, num_classes=param.num_classes)
    return model

def _get_trainhandler(opt, model, dataloaders, dataset_sizes):
    criterion = CRITERION[opt.criterion.algo](**opt.criterion.param)
    optimizer = OPTIMIZER[opt.optimizer.algo](model.parameters(), **opt.optimizer.param)
    if opt.scheduler.enabled:
        scheduler = SCHEDULER[opt.scheduler.type](optimizer, **opt.scheduler.param)
    else:
        scheduler = None
    
    train_handler = TrainHandler(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device=opt.trainhandler.device, path=opt.trainhandler.path)
    train_handler.set_prediction(ep_prediction)
    train_handler.set_name(opt.trainhandler.name)
    
    return train_handler

def _get_inspectionhandler(opt, model, dataloaders, dataset_sizes):
    inspection_handler = InspectionHandler(model, dataloaders, dataset_sizes, num_path=opt.inspectionhandler.num_path, path_cost=opt.inspectionhandler.path_cost, use_small=opt.inspectionhandler.use_small, phase=opt.inspectionhandler.phase)
    return inspection_handler

def run(opt):
    dataloaders, dataset_sizes = _get_dataset(opt.data)
    model = _get_model(opt)
    
    train_handler = _get_trainhandler(opt, model, dataloaders, dataset_sizes)
    train_handler.train_model(num_epochs=opt.trainhandler.num_epochs)
    train_handler.test_model()
    
    inspection_handler = _get_inspectionhandler(opt, model, dataloaders, dataset_sizes)
    inspection_handler.set_name(opt.trainhandler.name)
    inspection_handler.save_inspection()
    
if __name__ == "__main__":
    # gets arguments from the json file
    opt = ConfLoader(sys.argv[1]).opt
    
    # make experiment reproducible
    if opt.trainhandler.get('seed', None):
        torch.manual_seed(opt.trainhandler.seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(opt.trainhandler.seed)
        
    run(opt)
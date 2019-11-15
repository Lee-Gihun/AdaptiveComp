import os
import json
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['ConfLoader', 'model_saver', 'result_logger', 'result_dict_saver', 'EarlyStopping']


class ConfLoader:
    """ Load json config file using DictWithAttributeAccess object_hook.
    ConfLoader(conf_name).opt attribute is the result of loading json config file.
    """
    class DictWithAttributeAccess(dict):
        """ This inner class makes dict to be accessed same as class attribute.
        For example, you can use opt.key instead of the opt['key']
        """
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    def __init__(self, conf_name):
        self.conf_name = conf_name
        self.opt = self.__get_opt()
        
    def __load_conf(self):
        with open(self.conf_name, 'r') as conf:
            opt = json.load(conf, object_hook = lambda dict: self.DictWithAttributeAccess(dict))
        
        return opt
    
    def __get_opt(self):
        opt = self.__load_conf()
        opt = self.DictWithAttributeAccess(opt)
        
        return opt


def directory_setter(path='./results', make_dir=False):
    if not os.path.exists(path) and make_dir:
        os.makedirs(path) # make dir if not exist
        print('directory %s is created' % path)
        
    if not os.path.isdir(path):
        raise NotADirectoryError('%s is not valid. set make_dir=True to make dir.' % path)
        

def json_saver(json_file, json_path):
    with open(json_path, 'w') as fp:
        json.dump(json_file, fp)

        
def path_setter(result_path, sub_loc, model_name):
    save_path = '/'.join((result_path, sub_loc, model_name))
    return save_path
    
    
def result_dict_saver(result_dict, result_path='./results', sub_loc='inspection', model_name='model', make_dir=True):
    """
    saves model inspection results
    """
    save_path = path_setter(result_path, sub_loc, model_name)
    directory_setter(save_path, make_dir)
    
    result_dict_path = os.path.join(save_path, 'result_dict.json')
    json_saver(result_dict, result_dict_path)
    print('inspection result saved to %s' % save_path)
        
        
def model_saver(trained_model, initial_model, model_info=None, result_path='./results', sub_loc='trained_models', model_name='model', make_dir=True):
    """
    saves model weights and model description.
    """
    save_path = path_setter(result_path, sub_loc, model_name)
    directory_setter(save_path, make_dir)

    # save model description if exsists
    info_path = os.path.join(save_path, 'model_info.json')
    trained_model_path = os.path.join(save_path, 'trained_model.pth')
    initial_model_path = os.path.join(save_path, 'initial_model.pth')
    
    if model_info:
        json_saver(model_info, info_path)

    wts = trained_model.state_dict()
    torch.save(wts, trained_model_path)
    print('trained model saved as %s' % trained_model_path)
    
    ini_wts = initial_model
    torch.save(initial_model, initial_model_path)
    print('initial model saved as %s' % initial_model_path)
    
    

def result_logger(result_dict, epoch_num, result_path='./results', model_name='model', make_dir=True):
    """
    saves train results as .csv file
    """
    log_path = result_path + '/logs'
    file_name = model_name + '_results.csv'
    directory_setter(log_path, make_dir)
    save_path = os.path.join(log_path, file_name)
    header = ','.join(result_dict.keys()) + '\n'
    
    with open(save_path, 'w') as f:
        f.write(header)
        for i in range(epoch_num):
            row = []
            
            for item in result_dict.values():
                if type(item) is not list:
                    row.append('')

                elif item[i][1] is not None:
                    assert item[i][0] == (i+1), 'Not aligned epoch indices'
                    elem = round(item[i][1], 5)
                    row.append(str(elem))
                    
                else:
                    row.append('')
            
            # write each row
            f.write(','.join(row) + '\n')
            
        sep = len(result_dict.keys()) - 2
        f.write(','*sep + '%0.5f, %0.5f'% (result_dict['test_loss'], result_dict['test_acc']))
        
    print('results are logged at: \'%s' % save_path)    


class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        self.best_model = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss
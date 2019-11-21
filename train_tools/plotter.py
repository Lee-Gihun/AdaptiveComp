import os
import matplotlib.pyplot as plt
from .utils import directory_setter, path_setter

__all__ = ['train_plotter', 'RC_plotter', 'logit_plotter', 'performance_plotter']


def train_plotter(train, valid, test, mode, result_path='./results', model_name='model', make_dir=True, plot_freq=0.05):
    
    """
    plots loss or accuracy graph for train/valid logs, and saves as .png file.
    train, valid : list of tuples. ex) [(0, 0.08), (1, 0.56)...]
    test : float
    mode : 'loss' or 'accuracy'
    """
    save_path = path_setter(result_path, 'graphs', model_name)
    directory_setter(save_path, make_dir)
    
    fig=plt.figure(figsize=(8, 6), dpi= 80, facecolor='w', edgecolor='k')    
    plt.plot(*zip(*train), 'k', linestyle='-', label='train_%s'%mode)
    plt.plot(*zip(*valid), 'r', linestyle='-', label='valid_%s'%mode)
    plt.plot(len(train), test, 'bo', label='test_{}({:0.4f})'.format(mode, test))
    plt.xlabel('Epoch')
    plt.ylabel(mode)
    plt.legend()
    plt.grid()
    plt.xticks([x+1 for x in range(len(train)) if (x+1) % (len(train)*plot_freq) == 0])
    fname = os.path.join(save_path, '{}_{}_{}.png'.format(model_name, 'graph', mode))
    plt.savefig(fname)
    print('{} plot saved at {}'.format(mode, fname))

    
def RC_plotter(result_dict, num_path, tolerance=0.001, name='Risk-Coverage', xlim=0.5, \
               result_path='./results', model_name='model', make_dir=True):
    
    save_path = path_setter(result_path, 'inspection', model_name)
    directory_setter(save_path, make_dir)    
    
    risk, coverage, perfect_coverage = result_dict['risk'], result_dict['coverage'], result_dict['perfect_coverage']
    fig, axs = plt.subplots(1, num_path, figsize=(6*num_path, 6))
    fig.suptitle('Risk-Coverage', fontsize=20)
    
    for i in range(num_path):
        axs[i].set(title='Risk-Coverage Curve %d'%(i), xlabel='Risk(r)', ylabel='Coverage(c)')
        axs[i].plot(risk[i], coverage[i], color='red', linestyle='-', markersize=2, alpha=0.7)
        #axs[i].plot(risk[-1], coverage[-1], color='blue', linestyle='-', markersize=2, alpha=0.7)
        axs[i].set_ylim(0.0, 1.0)
        axs[i].set_xlim(0, 0.4)

        axs[i].annotate('r*(%0.4f, %0.2f)'%(risk[i][0], coverage[i][0]), xy=(risk[i][0], coverage[i][0]),  \
                        xytext=(risk[i][0]-0.01, coverage[i][0]-0.1), color='red', \
                     fontsize=10, weight='bold', arrowprops=dict(arrowstyle="->", color='red'))
        
        axs[i].annotate('c*(%0.3f, %0.3f)'%(0.001, perfect_coverage[i]), xy=(tolerance, perfect_coverage[i]), \
                        xytext=(0.001+0.005, perfect_coverage[i]-0.12), color='red', \
                     fontsize=10, weight='bold',arrowprops=dict(arrowstyle="->", color='red'))
                                                                
        #axs[i].annotate('r*(%0.4f, %0.2f)'%(risk[-1][0], coverage[-1][0]), xy=(risk[-1][0], coverage[-1][0]),\
        #                xytext=(risk[-1][0]-0.01, coverage[-1][0]-0.1), color='blue', \
        #     fontsize=10, weight='bold', arrowprops=dict(arrowstyle="->", color='blue'))

        #axs[i].annotate('c*(%0.3f, %0.3f)'%(0.001, perfect_coverage[-1]), xy=(tolerance, perfect_coverage[-1]), \
        #                xytext=(0.001+0.005, perfect_coverage[-1]-0.12), color='blue', \
        #             fontsize=10, weight='bold',arrowprops=dict(arrowstyle="->", color='blue'))

        axs[i].legend(['path %d'%(i+1)], loc=4)
        axs[i].grid()
    
    fname = os.path.join(save_path, '{}_{}.png'.format(model_name, 'RC'))
    fig.savefig(fname)
    
    print('%s inspection_RC graph is saved' % model_name)   


def logit_plotter(result_dict, num_path, result_path='./results', model_name='model', make_dir=True):
    save_path = path_setter(result_path, 'inspection', model_name)
    directory_setter(save_path, make_dir)
    
    fig, (axs) = plt.subplots(num_path, 2, figsize=(10, 6*num_path))
    fig.suptitle('logit_dist', fontsize=20)
    
    for i in range(num_path):    
        sr_dist, entropy_dist = result_dict['max_sr_dist_%d'%i], result_dict['entropy_dist_%d'%i],
        (max_logit_co, max_logit_inco), (entropy_co, entropy_inco) = sr_dist, entropy_dist
        ax1, ax2 = axs[i][0], axs[i][1]
        ax1.set(title='Max softmax outputs %d'%i, xlabel='softmax max value', ylabel='data count')
        ax1.hist(max_logit_co, color='brown', bins=[0.0+x*0.005 for x in range(200)], alpha=0.5)
        ax1.hist(max_logit_inco, color='gray', bins=[0.0+x*0.005 for x in range(200)], alpha=0.5)
        ax1.set_ylim(0, 800)
        ax1.set_xlim(0.2, 1)
        ax1.legend(['correct samples', 'incorrect samples'])
        ax1.grid()

        ax2.set(title='Entropy of Top5 SR %d'%i, xlabel='Top5 softmax entropy', ylabel='data count')
        ax2.hist(entropy_co, color='brown', bins=[x*0.01 for x in range(450)], alpha=0.5)
        ax2.hist(entropy_inco, color='gray', bins=[x*0.01 for x in range(450)], alpha=0.5)
        ax2.set_ylim(0, 800)
        ax2.set_xlim(0, 2.5)
        ax2.legend(['correct samples', 'incorrect samples'])
        ax2.grid()
    
    fname = os.path.join(save_path, '{}_{}.png'.format(model_name, 'logitdist'))
    fig.savefig(fname)
    
    print('%s inspection_logitdist graph is saved' % model_name)    
    
    
def performance_plotter(result_dict, num_path, name='acc_score', result_path='./results', model_name='model', make_dir=True):
    save_path = path_setter(result_path, 'inspection', model_name)
    directory_setter(save_path, make_dir)
    
    total_acc, flops_score = result_dict['total_acc'], result_dict['flops_score']
    fig=plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')    
    plt.scatter(flops_score, total_acc)
    plt.title('Max softmax outputs')
    plt.xlabel('Flops Score')
    plt.ylabel('Total Accuracy')
    plt.ylim(0, 1)
    plt.xlim(0.3, 1.4)
    plt.grid()
    
    fname = os.path.join(save_path, '{}_{}.png'.format(model_name, 'AccFlops'))
    plt.savefig(fname)
    
    print('%s inspection_AccFlops graph is saved' % model_name)   
import matplotlib.pyplot as plt

def RC_plotter(result_dict, num_paths, tolerance=0.001, name='Risk-Coverage', xlim=0.5):
    risk, coverage, perfect_coverage = result_dict['risk'], result_dict['coverage'], result_dict['perfect_coverage']
    fig, axs = plt.subplots(1, num_paths, figsize=(6*num_paths, 6))
    fig.suptitle(name, fontsize=20)
    
    for i in range(num_paths):
        axs[i].set(title='Risk-Coverage Curve %d'%(i), xlabel='Risk(r)', ylabel='Coverage(c)')
        axs[i].plot(risk[i], coverage[i], color='red', linestyle='-', markersize=2, alpha=0.7)
        axs[i].plot(risk[-1], coverage[-1], color='blue', linestyle='-', markersize=2, alpha=0.7)
        axs[i].set_ylim(0.0, 1.0)
        axs[i].set_xlim(0, round(max(risk[i])+0.05, 4))

        axs[i].annotate('r*(%0.4f, %0.2f)'%(risk[i][0], coverage[i][0]), xy=(risk[i][0], coverage[i][0]),  \
                        xytext=(risk[i][0]-0.01, coverage[i][0]-0.1), color='red', \
                     fontsize=10, weight='bold', arrowprops=dict(arrowstyle="->", color='red'))
        
        axs[i].annotate('c*(%0.3f, %0.3f)'%(0.001, perfect_coverage[i]), xy=(tolerance, perfect_coverage[i]), \
                        xytext=(0.001+0.005, perfect_coverage[i]-0.12), color='red', \
                     fontsize=10, weight='bold',arrowprops=dict(arrowstyle="->", color='red'))
                                                                
        axs[i].annotate('r*(%0.4f, %0.2f)'%(risk[-1][0], coverage[-1][0]), xy=(risk[-1][0], coverage[-1][0]),\
                        xytext=(risk[-1][0]-0.01, coverage[-1][0]-0.1), color='blue', \
             fontsize=10, weight='bold', arrowprops=dict(arrowstyle="->", color='blue'))

        axs[i].annotate('c*(%0.3f, %0.3f)'%(0.001, perfect_coverage[-1]), xy=(tolerance, perfect_coverage[-1]), \
                        xytext=(0.001+0.005, perfect_coverage[-1]-0.12), color='blue', \
                     fontsize=10, weight='bold',arrowprops=dict(arrowstyle="->", color='blue'))

        axs[i].legend(['path %d'%(i+1), 'baseline'], loc=4)
        axs[i].grid()
        
    fig.show()


def logit_plotter(reslt_dict, num_paths, name='logit_dist'):

    fig, (axs) = plt.subplots(num_paths, 2, figsize=(10, 6*num_paths))
    fig.suptitle(name, fontsize=20)
    
    for i in range(num_paths):    
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
    
    fig.show()
    fig.savefig('./%s.png' % name)
    print('%s inspection graph is saved' % name)    
    
    
def performance_plotter(result_dict, num_paths, name='acc_score'):
    total_acc, flops_score = result_dict['total_acc'], result_dict['flops_score']
    fig=plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k')    
    plt.scatter(flops_score, total_acc)
    plt.title('Max softmax outputs')
    plt.xlabel('Flops Score')
    plt.ylabel('Total Accuracy')
    plt.ylim(0, 1)
    plt.xlim(0.3, 1.4)
    plt.grid()
    plt.show()
    plt.savefig('./%s.png' % name)
    print('%s inspection graph is saved' % name)   
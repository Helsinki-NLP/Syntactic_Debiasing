import sys
import pickle
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

N_SAMPLES = 20
N_BERT_LAYERS = 12
N_MT_LAYERS = 6
TASK = 'positive-negative'
#TASK = 'active-passive'
MODEL_FOLDERS = ['BERT/None', 'MT/DE', 'MT/DE-EL']
MODEL_ALIASES = ['BERT', 'MT: EN > DE', 'MT: EN > DE+EL']
FOCI = ['verb', 'subject', 'object']


colors = ['r', 'c', 'm']


for focus in FOCI:
    if TASK == 'positive-negative' and focus == 'object':
        do_legend = True
    else:
        do_legend = False

    if do_legend:  
        plt.figure(figsize=(6.4,8))
    else:
        plt.figure(figsize=(6.4, 4.8))

    for midx, (model_folder, model_alias) in enumerate(zip(MODEL_FOLDERS, MODEL_ALIASES)):

        if model_alias == 'BERT':
            N_LAYERS = N_BERT_LAYERS
        else:
            N_LAYERS = N_MT_LAYERS

        if TASK == 'active-passive':
            sick_dataset = 'SICK'
        elif TASK == 'positive-negative':
            sick_dataset = 'SICK'            

        x_axis = range(N_LAYERS)
        x_labels = [str(i) for i in range(1, N_LAYERS+1)]  

        # Original dataset success

        accs = np.zeros((N_LAYERS, N_SAMPLES))

        for layer in range(N_LAYERS):
            fin = open('../../results/transfer_learning/' + TASK + '/' + model_folder 
                    + f'/RNN-priming-short-1000-{focus}_2_SICK-{focus}/'
                    + f'transfer_results_reg_coeff_0.0001_it_2_layer-{layer}_original.txt', 'r')

            accs[layer, :] = np.array([float(line.strip().split(' ')[0]) for line in fin])[-20:] * 100
            
        plt.errorbar(x_axis, np.mean(accs, axis=1), yerr=np.std(accs, axis=1), 
                     c=colors[midx], linestyle=':', linewidth=3,
                     label=f'{sick_dataset} Original, {model_alias}')

        # Cleaned dataset success

        accs = np.zeros((N_LAYERS, N_SAMPLES))

        for layer in range(N_LAYERS):
            fin = open('../../results/transfer_learning/' + TASK + '/' + model_folder 
                    + f'/RNN-priming-short-1000-{focus}_2_SICK-{focus}/'
                    + f'transfer_results_reg_coeff_0.0001_it_2_layer-{layer}_cleaned.txt', 'r')

            accs[layer, :] = np.array([float(line.strip().split(' ')[0]) for line in fin])[:20] * 100
            
        plt.errorbar(x_axis, np.mean(accs, axis=1), yerr=np.std(accs, axis=1), 
                     c=colors[midx], linestyle='-', linewidth=3,
                     label=f'{sick_dataset} Cleaned, {model_alias}')

    
    plt.ylim((45, 105))
    plt.yticks([50, 60, 70, 80, 90, 100], fontsize=16)
    plt.xticks(range(N_BERT_LAYERS), [str(i) for i in range(1, N_BERT_LAYERS+1)], fontsize=16)

    plt.xlabel('Layers', fontsize=16)
    plt.ylabel('accuracy (%)', fontsize=16)
    
    if TASK == 'active-passive':
        if focus == 'verb': plt.title(f'SICK-PAS: VERB', fontsize=16)
        if focus == 'subject': plt.title(f'SICK-PAS: A-SUBJ/P-AG', fontsize=16)
        if focus == 'object': plt.title(f'SICK-PAS: A-OBJ/P-SUBJ', fontsize=16)
    elif TASK == 'positive-negative':
        if focus == 'verb': plt.title(f'SICK-NEG: VERB', fontsize=16)
        if focus == 'subject': plt.title(f'SICK-NEG: SUBJECTS', fontsize=16)
        if focus == 'object': plt.title(f'SICK-NEG: OBJECTS', fontsize=16)        

    #if TASK == 'active-passive' and focus == 'object':
    #    plt.legend(loc='lower right', shadow=True, fontsize='medium', bbox_to_anchor=(0.92, 0))
    #else:
    if do_legend:
        plt.legend(loc='lower right', shadow=True, fontsize='xx-large', bbox_to_anchor=(1.07, -0.81))
    plt.grid(True)

    plt.show()

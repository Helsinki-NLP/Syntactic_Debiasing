import sys
import pickle
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

N_SAMPLES = 20
N_BERT_LAYERS = 12
N_MT_LAYERS = 6
MODEL_FOLDERS = ['BERT/None', 'MT/DE', 'MT/DE-EL']
MODEL_ALIASES = ['BERT', 'MT: EN > DE', 'MT: EN > DE+EL']
FOCI = ['verb', 'subject', 'object']


colors = ['r', 'c', 'm']


for focus in FOCI:
    plt.figure()

    for midx, (model_folder, model_alias) in enumerate(zip(MODEL_FOLDERS, MODEL_ALIASES)):

        if model_alias == 'BERT':
            N_LAYERS = N_BERT_LAYERS
        else:
            N_LAYERS = N_MT_LAYERS

        x_axis = range(N_LAYERS)
        x_labels = [str(i) for i in range(1, N_LAYERS+1)]  

        # Original dataset success

        accs = np.zeros((N_LAYERS, N_SAMPLES))

        for layer in range(N_LAYERS):
            fin = open('../../results/transfer_learning/' + model_folder 
                    + f'/RNN-priming-short-1000-{focus}_2_SICK-{focus}/'
                    + f'transfer_results_reg_coeff_0.0001_it_2_layer-{layer}_original.txt', 'r')

            accs[layer, :] = np.array([float(line.strip().split(' ')[0]) for line in fin]) * 100
            
        plt.errorbar(x_axis, np.mean(accs, axis=1), yerr=np.std(accs, axis=1), 
                     c=colors[midx], linestyle=':', 
                     label=f'SICK (original), {model_alias} embeddings')

        # Cleaned dataset success

        accs = np.zeros((N_LAYERS, N_SAMPLES))

        for layer in range(N_LAYERS):
            fin = open('../../results/transfer_learning/' + model_folder 
                    + f'/RNN-priming-short-1000-{focus}_2_SICK-{focus}/'
                    + f'transfer_results_reg_coeff_0.0001_it_2_layer-{layer}_cleaned.txt', 'r')

            accs[layer, :] = np.array([float(line.strip().split(' ')[0]) for line in fin]) * 100
            
        plt.errorbar(x_axis, np.mean(accs, axis=1), yerr=np.std(accs, axis=1), 
                     c=colors[midx], linestyle='-', 
                     label=f'SICK (cleaned by TEMPL), {model_alias} embeddings')

    
    plt.ylim((45, 105))
    plt.yticks([50, 60, 70, 80, 90, 100])
    plt.xticks(range(N_BERT_LAYERS), [str(i) for i in range(1, N_BERT_LAYERS+1)])
    plt.xlabel('Layers')
    plt.ylabel('accuracy (%)')
    
    if focus == 'verb': plt.title(f'Transferring debiasing projection from TEMPL to SICK: VERBS')
    if focus == 'subject': plt.title(f'Transferring debiasing projection from TEMPL to SICK: A-SUBJ / P-AG')
    if focus == 'object': plt.title(f'Transferring debiasing projection from TEMPL to SICK: A-OBJ / P-SUBJ')
    
    plt.legend(loc='lower right', shadow=True, fontsize='x-small')
    plt.grid(True)

    plt.show()

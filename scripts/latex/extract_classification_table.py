import sys
import os
import numpy as np

BASEDIR = '../../results/classification_acc/'
DATASETS = ['SICK', 'RNN-priming-short-1000']
FOCI = ['verb', 'subject', 'object']
REG_COEFF = '0.0001'
N_LAYERS = 12
N_SAMPLES = 20
N_ITERATIONS = 3


for dataset in DATASETS:
    if dataset == 'SICK': dataset_alias='SICK-AP'
    if dataset[0:3] == 'RNN': dataset_alias='TEMPL-AP'

    OUTFILE = f'../../results/classification_acc/classification_table_{dataset}.tex'
    fout = open(OUTFILE, 'w')
    fout.write('\\begin{table*}\n')
    fout.write('\\caption{}\n')
    fout.write('\\centering\n')
    fout.write('\\resizebox{\\textwidth}{!}{ \n')
    fout.write('\\begin{tabular}{|l||l||c|c|c||c|c|c||c|c|c||}\\hline\n')

    '''
    for layer in range(N_LAYERS):
        fout.write(' & \\multicolumn{3}{c}{Layer ' + str(layer+1) + '}')
    fout.write('\\\\ \n')
    fout.write('\\hline\n')

    for layer in range(N_LAYERS):
        fout.write(' & It-0 & It-1 & It-2 ')
    fout.write('\\\\ \n')
    fout.write('\\hline\n')

    for dataset in DATASETS:
        fout.write(dataset)
        for focus in FOCI:
            for layer in range(N_LAYERS):
                fin = open(f'{BASEDIR}/{dataset}/reg_coeff_{REG_COEFF}_it_3_{focus}_layer-{layer}.txt')
                accs = np.zeros((N_SAMPLES, N_ITERATIONS))
                for sample in range(N_SAMPLES):
                    line = np.array(fin.readline().split())
                    print(line)
                    try:
                        accs[sample, :] = line
                    except:
                        print(f'{BASEDIR}/{dataset}/reg_coeff_{REG_COEFF}_it_3_{focus}_layer-{layer}.txt')
                mean_accs = np.mean(accs, axis=0)
                stddev_accs = np.std(accs, axis=0)
                for meanacc, stdacc in zip(mean_accs, stddev_accs):
                    fout.write(' & %.2f\\pm%.2f' % (meanacc, stdacc))
            fout.write('\\\\ \n \\hline\n')
    '''

    for model, language in [('BERT', 'None'), ('MT', 'DE'), ('MT', 'DE-EL')]:

        if model == 'BERT': 
            model_layers = [1, 6, 12]
            model_alias = '{\\parbox{2cm}{\\centering BERT}}'
        else: 
            model_layers = [1, 3, 6]

            if language == 'DE':
                #model_alias = '\\parbox{4cm}{MT\\\\(EN $>$ DE)'
                model_alias = '{\\parbox{2cm}{\\centering MT \\\\ (EN $>$ DE)}}'
            else:
                #model_alias = '\\parbox{4cm}{MT\\\\(EN $>$ DE+EL)'
                model_alias = '{\\parbox{2cm}{\\centering MT (EN $>$ \\\\ DE+EL)}}'


        fout.write('\\multirow{5}{*}{\\rotatebox{90}{%s}} & ' % model_alias)

        for focus in FOCI:

            if focus == 'verb': focus_alias='VERBS'
            if focus == 'subject': focus_alias='A-SUBJ/P-AG'
            if focus == 'object': focus_alias='A-OBJ/P-SUBJ'

            fout.write(' & \\multicolumn{3}{c||}{%s:%s}' % (dataset_alias, focus_alias))
        
        fout.write('\\\\ \n')
        fout.write('\\cline{2-11}\n')

        
        fout.write(' & ')
        for focus in FOCI:
            fout.write(' & It-0 & It-1 & It-2 ')
        fout.write('\\\\ \n')
        fout.write('\\cline{2-11}\n')

        

        for layer in model_layers:
            fout.write('& {Layer %d}' % (layer))

            for focus in FOCI:
                fin = open(f'{BASEDIR}/{model}/{language}/{dataset}-{focus}/class_acc_reg_coeff_{REG_COEFF}_it_2_layer-{layer-1}.txt')
                accs = np.zeros((N_SAMPLES, N_ITERATIONS))
                for sample in range(N_SAMPLES):
                    line = np.array(fin.readline().split())
                    print(line)
                    try:
                        accs[sample, :] = line
                    except:
                        print(f'{BASEDIR}/{model}/{language}/{dataset}-{focus}/class_acc_reg_coeff_{REG_COEFF}_it_2_layer-{layer-1}.txt')
                
                #mean_accs = np.delete(np.mean(accs, axis=0), 1)
                #stddev_accs = np.delete(np.std(accs, axis=0), 1)

                mean_accs = np.mean(accs, axis=0)
                stddev_accs = np.std(accs, axis=0)

                for meanacc, stdacc in zip(mean_accs, stddev_accs):
                    fout.write(' & $%.2f\\pm%.2f$' % (meanacc, stdacc))
                    #fout.write(' & $%.2f\\pm%.2f$' % (meanacc, stdacc))

            fout.write('\\\\ \n \\cline{2-11}\n')
        fout.write('\\hline\n')


    fout.write('\\end{tabular}}')
    fout.write('\\end{table*}')
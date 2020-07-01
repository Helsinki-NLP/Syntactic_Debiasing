import sys
import pickle
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

N_SAMPLES = 20
REG_COEFFS = [0.1, 0.01, 0.001, 0.0001, 0.00001]

x_axis = range(len(REG_COEFFS))
x_labels = ['1e-1', '1e-2', '1e-3', '1e-4', '1e-5']

colors = ['r', 'g', 'c', 'm']
styles = ['-', '--', '-.', ':']

plt.figure()

for N_ITERATIONS in [1]:
    accs = np.zeros((len(x_axis), N_SAMPLES))
    for i, REG_COEFF in enumerate(REG_COEFFS):
        print(N_ITERATIONS, REG_COEFF)
        fin = open(f'../../results/dataset_size/SICK/reg_coeff_{REG_COEFF}_it_{N_ITERATIONS}.txt', 'r')

        accs[i, :] = np.array([float(line.strip().split(' ')[0]) for line in fin]) * 100

    plt.errorbar(x_axis, np.mean(accs, axis=1), yerr=np.std(accs, axis=1), color='k', linestyle='--', label='SICK uncleaned')


for c, DATASET in enumerate(['RNN-priming-short-500', 'RNN-priming-short-1000', 'RNN-priming-short-5000', 'RNN-priming-short-10000']):
    for s, N_ITERATIONS in enumerate([1, 2, 5, 10]):
        accs = np.zeros((len(x_axis), N_SAMPLES))
        for i, REG_COEFF in enumerate(REG_COEFFS):
            fin = open('../../results/dataset_size/' + DATASET + f'/reg_coeff_{REG_COEFF}_it_{N_ITERATIONS}.txt', 'r')

            accs[i, :] = np.array([float(line.strip().split(' ')[-1]) for line in fin]) * 100
            
        sent_count = DATASET.split('-')[-1]
        plt.errorbar(x_axis, np.mean(accs, axis=1), yerr=np.std(accs, axis=1), c=cm.prism((3-c)*10), linestyle=styles[s], label=f'SICK cleaned by RNN-{sent_count} at {N_ITERATIONS} it')

print(cm.PiYG(0), cm.PiYG(100))
plt.xticks(x_axis, x_labels)
plt.ylim((50, 100))
plt.xlabel('Inverse regularization strength C')
plt.ylabel('accuracy (%)')
plt.legend(loc='lower left', bbox_to_anchor=(0, 0), shadow=True, fontsize='x-small')
plt.grid(True)



plt.show()

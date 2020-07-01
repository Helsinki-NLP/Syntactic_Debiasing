import sys
import pickle
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

N_SAMPLES = 20
REG_COEFFS = [0.00001]
ITERATIONS = range(1, 10+1)

x_axis = range(len(ITERATIONS))
x_labels = ITERATIONS

colors = ['r', 'g', 'c', 'm']
styles = ['-', '--', '-.', ':']

plt.figure()

accs = np.zeros((N_SAMPLES, len(x_axis)))
fin = open(f'../../results/classification_acc/SICK/reg_coeff_1e-05_it_10_subject.txt', 'r')

for i in range(N_SAMPLES):
    line = fin.readline()
    accs[i, :] = np.array([float(i) for i in (line.strip().split(' '))]) * 100

plt.errorbar(x_axis, np.mean(accs, axis=0), yerr=np.std(accs, axis=0), color='r', linestyle='-', label='SICK')


accs = np.zeros((N_SAMPLES, len(x_axis)))
fin = open('../../results/classification_acc/RNN-priming-short-1000' + f'/reg_coeff_1e-05_it_10_subject.txt', 'r')

for i in range(N_SAMPLES):
    line = fin.readline()
    accs[i, :] = np.array([float(i) for i in (line.strip().split(' '))]) * 100

    
plt.errorbar(x_axis, np.mean(accs, axis=0), yerr=np.std(accs, axis=0), color='b', linestyle='--', label='TEMP-AP')

plt.xticks(x_axis, x_labels)
plt.ylim((45, 100))
plt.xlabel('Number of cleaning iterations')
plt.ylabel('accuracy (%)')
plt.legend(loc='upper right', shadow=True, fontsize='x-small')
plt.grid(True)



plt.show()

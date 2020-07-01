import sys
import os
import numpy as np
from sklearn import model_selection
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression, PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import logging
from utils import plotting

from utils.plotting import plot_mds, plot_pca
sys.path.append("../../src/nullspace_projection")
from src import debias


class Goldberg_Debiasing:
    def __init__(self, classifier='LinearSVC', n_iterations=10, reg_coeff=0.00001):
        self.n_iterations = n_iterations
        if classifier == 'LinearSVC':
            self.classifier = LinearSVC
            self.params = {'fit_intercept': False, 'class_weight': None, "dual": False, 'random_state': 0}
        elif classifier == 'LogisticRegression':
            self.classifier = LogisticRegression
            self.params = {'fit_intercept': True, 'penalty': 'l2', 'C': reg_coeff, "dual": False, 'random_state': 0, 'solver': 'lbfgs'}


    def train(self, X_train, Y_train, X_test, Y_test):
        min_acc = 0
        is_autoregressive = True
        dropout_rate = 0

        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)

        self.P, rowspace_projs, Ws, iteration_0_acc = debias.get_debiasing_projection(self.classifier, self.params, self.n_iterations, 768, is_autoregressive, min_acc,
                                                        X_train, Y_train, X_test, Y_test,
                                                        Y_train_main=None, Y_dev_main=None, 
                                                        by_class = False, dropout_rate = dropout_rate)


        return self.P, iteration_0_acc



    def clean_data(self, X, P=None):
        if P == None:
            P = self.P
        print('P.shape:', P.shape)
        print('X.shape:', X.shape)
        X_cleaned = (P.dot(X.T)).T
        print('X_cleaned.shape:', X_cleaned.shape)
        return X_cleaned



    def debias(self, X_train, Y_train, X_test, Y_test, train_dataset, test_dataset, is_transfer_projmatrix, is_transfer_classifier, is_plot, logfile):
        if not (is_transfer_projmatrix or is_transfer_classifier):
            self.P, iteration_accs = self.train(X_train[train_dataset], Y_train[train_dataset], X_test[train_dataset], Y_test[train_dataset])

            for acc in iteration_accs:
                logfile.write('%.2f ' % acc)
            logfile.write('\n')

            '''if is_plot:

                cls1_uncleaned = np.array([data for i, data in enumerate(X_train[test_dataset]) if Y_train[test_dataset][i] == 0] 
                               + [data for i, data in enumerate(X_test[test_dataset]) if Y_test[test_dataset][i] == 0])
                cls2_uncleaned = np.array([data for i, data in enumerate(X_train[test_dataset]) if Y_train[test_dataset][i] == 1]
                               + [data for i, data in enumerate(X_test[test_dataset]) if Y_test[test_dataset][i] == 1])
                
                plotting.plot_mds(cls1_uncleaned, cls2_uncleaned)

                X_train_cleaned = self.clean_data(X_train[test_dataset])
                X_test_cleaned = self.clean_data(X_test[test_dataset])

                cls1_cleaned = np.array([data for i, data in enumerate(X_train_cleaned) if Y_train[test_dataset][i] == 0]
                             + [data for i, data in enumerate(X_test_cleaned) if Y_test[test_dataset][i] == 0])
                cls2_cleaned = np.array([data for i, data in enumerate(X_train_cleaned) if Y_train[test_dataset][i] == 1]
                             + [data for i, data in enumerate(X_test_cleaned) if Y_test[test_dataset][i] == 1])

                plotting.plot_mds(cls1_cleaned, cls2_cleaned)
            '''

        elif is_transfer_classifier:
            # if set, we try the test dataset on the same classifier that is trained on the train dataset
            self.P, _ = self.train(X_train[train_dataset], Y_train[train_dataset], X_test[test_dataset], Y_test[test_dataset])
        
        elif is_transfer_projmatrix:
            self.P, iteration_accs_ds1 = self.train(X_train[train_dataset], Y_train[train_dataset], X_test[train_dataset], Y_test[train_dataset])

            # "clean" the test dataset:
            X_train_cleaned = {test_dataset: self.clean_data(X_train[test_dataset], self.P)}
            X_test_cleaned = {test_dataset: self.clean_data(X_test[test_dataset], self.P)}

            # try to re-debias the "cleaned" dataset:
            _, iteration_accs_ds2 = self.train(X_train_cleaned[test_dataset], Y_train[test_dataset], X_test_cleaned[test_dataset], Y_test[test_dataset])

            logfile.write('%.2f %.2f\n' % (iteration_accs_ds1[0], iteration_accs_ds2[0]))

        return self.P



#---------------------------------------

class GBDD_Debiasing:
    '''Lauscher et al., 
       A General Framework for Implicit and Explicit Debiasing of Distributional Word Vector Spaces
       The original (Dev and Phillips (2019)) version for Proposed Method 1

       IMPORTANT: Assumes single-word focus. Doesn't work for focus=all, since it assumes strict active-passive instance pairs
                  for calculating the bias_dir_vec
    '''

    def __init__(self, cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances):
        self.cls1_train_instances = cls1_train_instances
        self.cls2_train_instances = cls2_train_instances

        self.cls1_test_instances = cls1_test_instances
        self.cls2_test_instances = cls2_test_instances



    def calc_bias_dir_vec(self):
        # This function uses the original (Dev and Phillips (2019)) version for Proposed Method 1
        # IMPORTANT: Assumes single-word in every instance

        N_train = len(self.cls1_train_instances)
        ENC_DIM = self.cls1_train_instances[0].shape[1]
        X = np.zeros((N_train, ENC_DIM))

        for i, (cls1_instance, cls2_instance) in enumerate(zip(self.cls1_train_instances, self.cls2_train_instances)):
            diff_vector = cls1_instance - cls2_instance
            X[i, :] = diff_vector

        #U, s, V = np.linalg.svd(X, full_matrices=True)

        X -= np.mean(X, axis=0)
        pca = PCA(n_components=2, whiten=True)
        pca.fit(X)

        #logging.debug(pca.components_[0].shape) 
        self.bias_dir_vec = pca.components_[0].reshape(1, ENC_DIM)

        #return self.bias_dir_vec


    def clean_data(self, x):
        '''Lauscher et al. Equation 1'''

        print('x.shape', x.shape)
        print('bias vector shape', self.bias_dir_vec.shape)

        # l2-normalize x
        x_norm = x

        # remove from x its projection onto the global bias direction vector
        return x_norm - np.dot(x_norm, self.bias_dir_vec.T) * self.bias_dir_vec
        

    def plot_before_after(self, cls1_instances, cls2_instances):

        cls1 = np.concatenate(cls1_instances, axis=0)
        cls1_clean = self.clean_data(cls1)

        cls2 = np.concatenate(cls2_instances, axis=0)
        cls2_clean = self.clean_data(cls2)

        logging.debug('plotting before')
        plot_pca(cls1, cls2, figname='../results/gbdd_pca_before_test.png')

        logging.debug('plotting after')
        plot_pca(cls1_clean, cls2_clean, figname='../results/gbdd_pca_after_test.png')        


#---------------------------------------

class BAM_Debiasing:
    '''Lauscher et al., 
       A General Framework for Implicit and Explicit Debiasing of Distributional Word Vector Spaces
       Proposed Method 2
    '''

    def __init__(self):
        pass

    def clean_data(x):
        '''Lauscher et al. Equation 2'''
        pass


















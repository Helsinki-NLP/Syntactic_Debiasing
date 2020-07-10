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
from plotting import visualize

sys.path.append("../../src/nullspace_projection")
from src import debias

N_LAYERS = 12

class iNLP_Debiasing:
    def __init__(self, classifier='LinearSVC', n_iterations=10, reg_coeff=0.00001):
        self.n_iterations = n_iterations
        if classifier == 'LinearSVC':
            self.classifier = LinearSVC
            self.params = {'fit_intercept': True, 'penalty': 'l2', 'C': reg_coeff, 'class_weight': None, 'dual': False, 'random_state': 0}
        elif classifier == 'LogisticRegression':
            self.classifier = LogisticRegression
            self.params = {'fit_intercept': True, 'penalty': 'l2', 'C': reg_coeff, 'dual': False, 'random_state': 0, 'solver': 'lbfgs'}

        self.P = np.zeros((12, 768, 768))

    def train(self, X_train, Y_train, X_test, Y_test, layer, do_set_P=True):
        min_acc = 0
        is_autoregressive = True
        dropout_rate = 0

        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)

        P, rowspace_projs, Ws, iteration_accs = debias.get_debiasing_projection(self.classifier, self.params, self.n_iterations, 768, is_autoregressive, min_acc,
                                                        X_train[:,layer,:], Y_train, X_test[:,layer,:], Y_test,
                                                        Y_train_main=None, Y_dev_main=None, 
                                                        by_class = False, dropout_rate = dropout_rate)


        if do_set_P:
            self.P[layer,:,:] = P

        return iteration_accs



    def clean_data(self, X):
        X_cleaned = np.empty_like(X)

        for layer in range(N_LAYERS):
            if len(X.shape) == 3:
                X_cleaned[:,layer,:] = (self.P[layer,:,:].dot(X[:,layer,:].T)).T
            else:
                X_cleaned[layer,:] = (self.P[layer].dot(X[layer,:].T)).T

        return X_cleaned



    def debias(self, X_train, Y_train, X_test, Y_test, train_dataset, test_dataset, train_focus, test_focus, is_transfer_projmatrix, is_transfer_classifier, is_plot, logfile_base):
        
        if not (is_transfer_projmatrix or is_transfer_classifier):

            for layer in range(N_LAYERS):
                logfile = open(f'{logfile_base}_layer-{layer}.txt', 'a') 
                iteration_accs = self.train(X_train[train_dataset][train_focus], 
                                             Y_train[train_dataset][train_focus], 
                                             X_test[train_dataset][train_focus], 
                                             Y_test[train_dataset][train_focus], 
                                             layer, 
                                             do_set_P=True)

                for acc in iteration_accs:
                    logfile.write('%.2f ' % acc)
                logfile.write('\n')
                logfile.close()

            print(self.P[5,:,:])

        elif is_transfer_classifier:
            # if set, we try the test dataset on the same classifier that is trained on the train dataset
            for layer in range(N_LAYERS):
                _ = self.train(X_train[train_dataset][train_focus], 
                               Y_train[train_dataset][train_focus], 
                               X_test[test_dataset][test_focus], 
                               Y_test[test_dataset][test_focus], 
                               layer, 
                               do_set_P=True)
        

        elif is_transfer_projmatrix:

            for layer in range(N_LAYERS):
                logfile = open(f'{logfile_base}_layer-{layer}_original.txt', 'w') 
            
                print('Original Layer %d' % (layer+1))
                iteration_accs_ds1 = self.train(X_train[train_dataset][train_focus], 
                                                Y_train[train_dataset][train_focus], 
                                                X_test[train_dataset][train_focus], 
                                                Y_test[train_dataset][train_focus], 
                                                layer, 
                                                do_set_P=True)

                for acc in iteration_accs_ds1:
                    logfile.write('%.2f ' % acc)
                logfile.write('\n')
                logfile.close()


            # "clean" the test dataset:
            X_train_cleaned_test_dataset_test_focus = self.clean_data(X_train[test_dataset][test_focus])
            X_test_cleaned_test_dataset_test_focus  = self.clean_data(X_test[test_dataset][test_focus])

            for layer in range(N_LAYERS):

                print('"Cleaned" Layer %d' % (layer+1))
                logfile = open(f'{logfile_base}_layer-{layer}_cleaned.txt', 'w') 

                # try to re-debias the "cleaned" dataset:
                iteration_accs_ds2 = self.train(X_train_cleaned_test_dataset_test_focus, 
                                                Y_train[test_dataset][test_focus], 
                                                X_test_cleaned_test_dataset_test_focus, 
                                                Y_test[test_dataset][test_focus], 
                                                layer, 
                                                do_set_P=False)

                for acc in iteration_accs_ds2:
                    logfile.write('%.2f ' % acc)
                logfile.write('\n')
                logfile.close()

        return self.P



#---------------------------------------

class BDD_Debiasing:
    '''Dev and Phillips

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
        ENC_DIM = self.cls1_train_instances[0].shape[-1]
        X = np.zeros((N_train, ENC_DIM))

        for i, (cls1_instance, cls2_instance) in enumerate(zip(self.cls1_train_instances, self.cls2_train_instances)):
            diff_vector = cls1_instance[:,5,:] - cls2_instance[:,5,:]
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
        

    def plot_before_after(self):

        cls1 = np.concatenate(self.cls1_test_instances, axis=0)[:,5,:]
        cls1_clean = self.clean_data(cls1)

        cls2 = np.concatenate(self.cls2_test_instances, axis=0)[:,5,:]
        cls2_clean = self.clean_data(cls2)

        logging.debug('plotting before')
        visualize.project(cls1, cls2, projection='pca')

        logging.debug('plotting after')
        visualize.project(cls1_clean, cls2_clean, projection='pca')        




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

        N_train = len(self.cls1_train_instances) * len(self.cls2_train_instances)
        ENC_DIM = self.cls1_train_instances[0].shape[-1]
        X = np.zeros((N_train, ENC_DIM))

        i = 0
        for cls1_instance in self.cls1_train_instances:
            for cls2_instance in self.cls2_train_instances:
                diff_vector = cls1_instance[:,5,:] - cls2_instance[:,5,:]
                X[i, :] = diff_vector
                i += 1

        print('X.shape', X.shape)

        U, s, V = np.linalg.svd(X, full_matrices=True)

        #X -= np.mean(X, axis=0)
        #pca = PCA(n_components=1, whiten=True)
        #pca.fit(X)

        #logging.debug(pca.components_[0].shape) 
        #self.bias_dir_vec = pca.components_[0].reshape(1, ENC_DIM)

        #return self.bias_dir_vec


    def clean_data(self, x):
        '''Lauscher et al. Equation 1'''

        print('x.shape', x.shape)
        print('bias vector shape', self.bias_dir_vec.shape)

        for i in range(x.shape[0]):
            # l2-normalize x
            x_norm_i = x[i,:] #np.linalg.norm(x[i,:], ord=2)
            #print(x_norm_i.shape)

            # remove from x its projection onto the global bias direction vector
            x[i,:] = x_norm_i - np.dot(x_norm_i, self.bias_dir_vec.T) * self.bias_dir_vec
        
        return x


    def rotate(self, n_iterations=10):
        cls1_train = np.concatenate(self.cls1_train_instances, axis=0)
        cls2_train = np.concatenate(self.cls2_train_instances, axis=0)

        cls1_test = np.concatenate(self.cls1_test_instances, axis=0)
        cls2_test = np.concatenate(self.cls2_test_instances, axis=0)

        logging.debug('plotting before')
        visualize.project(cls1_test[:,5,:], cls2_test[:,5,:], projection='pca')

        N_train = len(self.cls1_train_instances)
        ENC_DIM = self.cls1_train_instances[0].shape[-1]

        for n in range(n_iterations):
            X = np.zeros((N_train, ENC_DIM))

            count = 0
            #for i in range(cls1_train.shape[0]):
            #    for j in range(cls2_train.shape[0]):
            for i in range(cls1_train.shape[0]):
                    diff_vector = cls1_train[i,5,:] - cls2_train[i,5,:]
                    X[count, :] = diff_vector
                    count += 1

            print('X.shape', X.shape)

            U, s, V = np.linalg.svd(X, full_matrices=True)

            #X -= np.mean(X, axis=0)
            #pca = PCA(n_components=2, whiten=True)
            #pca.fit(X)

            #logging.debug(pca.components_[0].shape) 
            #self.bias_dir_vec = pca.components_[0].reshape(1, ENC_DIM)

            self.bias_dir_vec = V[0,:]

            cls1_train[:,5,:] = self.clean_data(cls1_train[:,5,:])
            cls2_train[:,5,:] = self.clean_data(cls2_train[:,5,:])

            cls1_test[:,5,:] = self.clean_data(cls1_test[:,5,:])
            cls2_test[:,5,:] = self.clean_data(cls2_test[:,5,:])


        logging.debug('after iteration %d' % (n+1))
        visualize.project(cls1_test[:,5,:], cls2_test[:,5,:], projection='pca')

        classifier = LogisticRegression
        params = {'fit_intercept': True, 'penalty': 'l2', 'C': 0.00001, 'dual': False, 'random_state': 0, 'solver': 'lbfgs'}
        min_acc = 0
        is_autoregressive = True
        dropout_rate = 0

        print(cls1_test.shape)

        x_train = np.concatenate([cls1_test[0:50,5,:], cls2_test[0:50,5,:]], axis=0)
        y_train = np.concatenate([np.zeros(cls1_test[0:50,5,:].shape[0], dtype=int), np.ones(cls2_test[0:50,5,:].shape[0], dtype=int)])

        x_test = np.concatenate([cls1_test[50:-1,5,:], cls2_test[50:-1,5,:]], axis=0)
        y_test = np.concatenate([np.zeros(cls1_test[50:-1,5,:].shape[0], dtype=int), np.ones(cls2_test[50:-1,5,:].shape[0], dtype=int)])

        P, rowspace_projs, Ws, iteration_accs = debias.get_debiasing_projection(classifier, params, n_iterations, 768, is_autoregressive, min_acc,
                            x_train, y_train, 
                            x_test, y_test,
                            Y_train_main=None, Y_dev_main=None, 
                            by_class = False, dropout_rate = dropout_rate)


    def plot_before_after(self):

        cls1 = np.concatenate(self.cls1_test_instances, axis=0)[:,5,:]
        cls1_clean = self.clean_data(cls1)

        cls2 = np.concatenate(self.cls2_test_instances, axis=0)[:,5,:]
        cls2_clean = self.clean_data(cls2)

        logging.debug('plotting before')
        visualize.project(cls1, cls2, projection='pca')

        logging.debug('plotting after')
        visualize.project(cls1_clean, cls2_clean, projection='pca')        



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


















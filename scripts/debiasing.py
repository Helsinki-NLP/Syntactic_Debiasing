import sys
import os

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

sys.path.append("../../src/nullspace_projection")
from src import debias

class Debiasing:
    def __init__(self, classifier='LinearSVC', n_iterations=30):
        self.n_iterations = n_iterations
        if classifier == 'LinearSVC':
            self.classifier = LinearSVC
            self.params = {'fit_intercept': False, 'class_weight': None, "dual": False, 'random_state': 0}


    def train(self, X_train, Y_train, X_test, Y_test):
        min_acc = 0
        is_autoregressive = True
        dropout_rate = 0

        print(X_train.shape)
        print(Y_train.shape)
        print(X_test.shape)
        print(Y_test.shape)

        self.P, rowspace_projs, Ws = debias.get_debiasing_projection(self.classifier, self.params, self.n_iterations, 768, is_autoregressive, min_acc,
                                                        X_train, Y_train, X_test, Y_test,
                                                        Y_train_main=None, Y_dev_main=None, 
                                                        by_class = False, dropout_rate = dropout_rate)


        return self.P



    def clean_data(self, X, P=None):
        if P == None:
            P = self.P

        X_cleaned = (self.P.dot(X.T))
        return X_cleaned



    def debias(self, X_train, Y_train, X_test, Y_test, train_dataset, test_dataset, is_transfer_projmatrix, is_transfer_classifier):
        if not (is_transfer_projmatrix or is_transfer_classifier):
            self.P = self.train(X_train[train_dataset], Y_train[train_dataset], X_test[train_dataset], Y_test[train_dataset])

        elif is_transfer_classifier:
            # if set, we try the test dataset on the same classifier that is trained on the train dataset
            self.P = self.train(X_train[opt.train_on], Y_train[opt.train_on], X_test[opt.test_on], Y_test[opt.test_on])
        
        elif is_transfer_projmatrix:
            self.P = self.train(X_train[opt.train_on], Y_train[opt.train_on], X_test[opt.train_on], Y_test[opt.train_on])

            # "clean" the test dataset:
            X_train_cleaned = {opt.test_on: db.clean_data(X_train[opt.test_on], P)}
            X_test_cleaned = {opt.test_on: db.clean_data(X_test[opt.test_on], P)}

            # try to re-debias the "cleaned" dataset:
            _ = self.train(X_train_cleaned[opt.test_on], Y_train[opt.test_on], X_test_cleaned[opt.test_on], Y_test[opt.test_on])

        return self.P
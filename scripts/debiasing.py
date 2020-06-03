import sys
import os

sys.path.append("../../src/nullspace_projection/src/")
import debias

class Debiasing:
    def __init__(self, classifier='LinearSVC', n_iterations=30):
        self.classifier = classifier
        if self.classifier == 'LinearSVC':
            self.params = {'fit_intercept': False, 'class_weight': None, "dual": False, 'random_state': 0}

    def train(self, X_train, Y_train, X_test, Y_test):
        min_acc = 0
        is_autoregressive = True
        dropout_rate = 0

        self.P, rowspace_projs, Ws = debias.get_debiasing_projection(self.classifier, self.params, self.n_iterations, 768, is_autoregressive, min_acc,
                                                        X_train, Y_train, X_test, Y_test,
                                                        Y_train_main=None, Y_dev_main=None, 
                                                        by_class = False, dropout_rate = dropout_rate)


        return self.P


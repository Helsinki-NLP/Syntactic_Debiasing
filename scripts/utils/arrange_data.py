import sys
import os
import numpy as np
import random
import logging

TEST_RATIO = 0.3

def train_test_split(cls1_instances, cls2_instances, cls1_words, cls2_words, datasets, layer, is_lexical_split):
    X_train = {}
    X_test = {}
    Y_train = {}
    Y_test = {}


    for dataset in datasets:
        
        X_train[dataset] = np.array([])
        X_test[dataset] = np.array([])

        Y_train[dataset] = np.array([])
        Y_test[dataset] = np.array([])

        n_sentences  = len(cls1_instances[dataset])
        vocabulary = set([word for sentence in cls1_words[dataset] for word in sentence])

        # lexical split will be used only for single-word instances.
        # therefore there is no need to propagate all sentence if lexical split is used

        # Also make sure that both class 1 and class 2 of the same example goes into the same set.


        if not is_lexical_split: 
            # is_lexical_split = False
            # There can be more than one instance coming from the same sentence
            # We must make sure all sentence goes into the same set

            for i in range(n_sentences):
                new_X = np.concatenate([cls1_instances[dataset][i][:,layer,:].detach().squeeze(1).numpy(), cls2_instances[dataset][i][:,layer,:].detach().squeeze(1).numpy()], axis=0)
                new_Y = np.concatenate([np.zeros(cls1_instances[dataset][i].shape[0], dtype=int), np.ones(cls2_instances[dataset][i].shape[0], dtype=int)])

                if random.uniform(0, 1) < 1 - TEST_RATIO:
                    X_train[dataset] = np.concatenate([X_train[dataset], new_X], axis=0) if X_train[dataset].size else new_X
                    Y_train[dataset] = np.concatenate([Y_train[dataset], new_Y]) if Y_train[dataset].size else new_Y
                else:
                    X_test[dataset] = np.concatenate([X_test[dataset], new_X], axis=0) if X_test[dataset].size else new_X
                    Y_test[dataset] = np.concatenate([Y_test[dataset], new_Y]) if Y_test[dataset].size else new_Y

        else: 
            # is_lexical_split = True
            # There cannot be more than one instance coming from the same sentence
            # We must make sure that we split lexically, ie., all instances of a single word goes to the same set

            assignments = {word: ('train' if (random.uniform(0, 1) < 1 - TEST_RATIO) else 'test') for word in vocabulary}
            for i in range(n_sentences):
                new_X = np.concatenate([cls1_instances[dataset][i][:,layer,:].detach().squeeze(1).numpy(), cls2_instances[dataset][i][:,layer,:].detach().squeeze(1).numpy()], axis=0)
                new_Y = np.concatenate([np.zeros(cls1_instances[dataset][i].shape[0], dtype=int), np.ones(cls2_instances[dataset][i].shape[0], dtype=int)])
                
                # assuming there is a single word coming from every sentence!
                # otherwise we wont use this constraint.
                single_cls1_word = cls1_words[dataset][i][0]
                if assignments[single_cls1_word] == 'train':
                    X_train[dataset] = np.concatenate([X_train[dataset], new_X], axis=0) if X_train[dataset].size else new_X
                    Y_train[dataset] = np.concatenate([Y_train[dataset], new_Y]) if Y_train[dataset].size else new_Y
                else:
                    X_test[dataset] = np.concatenate([X_test[dataset], new_X], axis=0) if X_test[dataset].size else new_X
                    Y_test[dataset] = np.concatenate([Y_test[dataset], new_Y]) if Y_test[dataset].size else new_Y


        logging.debug(X_train[dataset].shape)
        logging.debug(Y_train[dataset].shape)
        logging.debug(X_test[dataset].shape)
        logging.debug(Y_test[dataset].shape)

    return X_train, Y_train, X_test, Y_test


def combine_train_test(X_train, Y_train, X_test, Y_test):
    X_combined = np.concatenate([X_train, X_test], axis=0)
    Y_combined = np.concatenate([Y_train, Y_test])

    return X_combined, Y_combined


def restrict_vocab(cls1_instances, cls2_instances, cls1_words, cls2_words):
    logging.info('Restricting the two datasets to the same vocabulary')

    X_train = {}
    X_test = {}

    Y_train = {}
    Y_test = {}

    logging.debug('\n\nClass 1 words in ' + dataset + '\n')
    logging.debug(cls1_words[dataset])
    logging.debug('\n\nClass 2 words in ' + dataset + '\n\n')
    logging.debug(cls2_words[dataset])

    # keep only the intersection of lexical items in both datasets
    shared_vocab = set([item[0] for item in cls2_words[opt.train_on]]).intersection( \
                     set([item[0] for item in cls2_words[opt.test_on]])) # use the passive items, their forms are similar

    logging.debug('\n\nShared vocabulary ')
    logging.debug(shared_vocab)

    new_cls1_instances = {dataset: [] for dataset in opt.dataset}
    new_cls2_instances = {dataset: [] for dataset in opt.dataset}

    new_cls1_words = {dataset: [] for dataset in opt.dataset}
    new_cls2_words = {dataset: [] for dataset in opt.dataset}

    for dataset in opt.dataset:
        for i in range(len(cls1_instances[dataset])):
            passive_word = cls2_words[dataset][i][0]
            if passive_word in shared_vocab:
                new_cls1_instances[dataset].append(cls1_instances[dataset][i])
                new_cls2_instances[dataset].append(cls2_instances[dataset][i])
                new_cls1_words[dataset].append(cls1_words[dataset][i])
                new_cls2_words[dataset].append(cls2_words[dataset][i])

    cls1_instances = new_cls1_instances
    cls2_instances = new_cls2_instances

    cls1_words = new_cls1_words
    cls2_words = new_cls2_words



    return cls1_instances, cls2_instances, cls1_words, cls2_words

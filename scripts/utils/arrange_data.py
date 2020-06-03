import sys
import os
import numpy as np
import random

TEST_RATIO = 0.3

def train_test_split(cls1_instances, cls2_instances, cls1_words, cls2_words, layer, is_lexical_split):
    X_train = np.array([])
    X_test = np.array([])

    Y_train = np.array([])
    Y_test = np.array([])

    n_sentences  = len(cls1_instances)
    vocabulary = set([word for sentence in cls1_words for word in sentence])

    # lexical split will be used only for single-word instances.
    # therefore there is no need to propagate all sentence if lexical split is used

    # Also make sure that both class 1 and class 2 of the same example goes into the same set.


    if not is_lexical_split: 
        # is_lexical_split = False
        # There can be more than one instance coming from the same sentence
        # We must make sure all sentence goes into the same set

        for i in range(n_sentences):
            new_X = np.concatenate([cls1_instances[i][:,layer,:].detach().squeeze(1).numpy(), cls2_instances[i][:,layer,:].detach().squeeze(1).numpy()], axis=0)
            new_Y = np.concatenate([np.zeros(cls1_instances[i].shape[0], dtype=int), np.ones(cls2_instances[i].shape[0], dtype=int)])

            if random.uniform(0, 1) < 1 - TEST_RATIO:
                X_train = np.concatenate([X_train, new_X], axis=0) if X_train.size else new_X
                Y_train = np.concatenate([Y_train, new_Y]) if Y_train.size else new_Y
            else:
                X_test = np.concatenate([X_test, new_X], axis=0) if X_test.size else new_X
                Y_test = np.concatenate([Y_test, new_Y]) if Y_test.size else new_Y

    else: 
        # is_lexical_split = True
        # There cannot be more than one instance coming from the same sentence
        # We must make sure that we split lexically, ie., all instances of a single word goes to the same set

        assignments = {word: ('train' if (random.uniform(0, 1) < 1 - TEST_RATIO) else 'test') for word in vocabulary}
        for i in range(n_sentences):
            # assuming there is a single word coming from every sentence!
            # otherwise we wont use this constraint.
            new_X = np.concatenate([cls1_instances[i][:,layer,:].detach().squeeze(1 ).numpy(), cls2_instances[i][:,layer,:].detach().squeeze(1).numpy()], axis=0)
            new_Y = np.concatenate([np.zeros(cls1_instances[i].shape[0], dtype=int), np.ones(cls2_instances[i].shape[0], dtype=int)])
            
            if assignments[cls1_words[i]] == 'train':
                X_train = np.concatenate([X_train, new_X], axis=0) if X_train.size else new_X
                Y_train = np.concatenate([Y_train, new_Y]) if Y_train.size else new_Y
            else:
                X_test = np.concatenate([X_test, new_X], axis=0) if X_test.size else new_X
                Y_test = np.concatenate([Y_test, new_Y]) if Y_test.size else new_Y


    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    print(Y_train)
    print(Y_test)

    return X_train, Y_train, X_test, Y_test

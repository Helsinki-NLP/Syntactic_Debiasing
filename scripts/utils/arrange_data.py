import sys
import os
import numpy as np
import random

TEST_RATIO = 0.3
labels = {'active':0, 'passive':1}

def train_test_split(cls1_instances, cls2_instances, cls1_words, cls2_words, is_lexical_split):
    X_train = []
    X_test = []

    Y_train = []
    Y_test = []

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
            if random.uniform(0, 1) < 1 - TEST_RATIO:
                X_train += [cls1_instances[i]] + [cls2_instances[i]]
                Y_train += [labels['active']] * len(cls1_instances[i]) + [labels['passive']] * len(cls2_instances[i])
            else:
                X_test += [cls1_instances[i]] + [cls2_instances[i]]
                Y_test += [labels['active']] * len(cls1_instances[i]) + [labels['passive']] * len(cls2_instances[i])

    else: 
        # is_lexical_split = True
        # There cannot be more than one instance coming from the same sentence
        # We must make sure that we split lexically, ie., all instances of a single word goes to the same set

        assignments = {word: ('train' if (random.uniform(0, 1) < 1 - TEST_RATIO) else 'test') for word in vocabulary}
        for i in range(n_sentences):
            # assuming there is a single word coming from every sentence!
            # otherwise we wont use this constraint.
            if assignments[cls1_words[i]] == 'train':
                X_train += [cls1_instances[i]] + [cls2_instances[i]]
                Y_train += [labels['active']] * len(cls1_instances[i]) + [labels['passive']] * len(cls2_instances[i])
            else:
                X_test += [cls1_instances[i]] + [cls2_instances[i]]
                Y_test += [labels['active']] * len(cls1_instances[i]) + [labels['passive']] * len(cls2_instances[i])


    return X_train, Y_train, X_test, Y_test

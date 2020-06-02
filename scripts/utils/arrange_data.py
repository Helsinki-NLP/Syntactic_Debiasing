import sys
import os
import numpy as np
import random

TEST_RATIO = 0.3

def train_set_split(cls1_instances, cls2_instances, cls1_words, cls2_words, is_lexical_split):
    train_set = []
    test_set = []

    n_sentences  = len(cls1_instances)
    vocabulary = set([word for sentence in cls1_words for word in sentence])

    # lexical split will be used only for single-word instances.
    # therefore there is no need to propagate all sentence if lexical split is used

    # Also make sure that both class 1 and class 2 of the same example goes into the same set.

    if not is_lexical_split: 
        # is_lexical_split = False
        # There can be more than one instance coming from the same sentence
        # We must make sure all sentence goes into the same set

        for i in range(len(n_sentences)):
            if random.uniform(0, 1) < 1 - TEST_RATIO:
                train_set += cls1_instances[i] + cls2_instances[i]
                print(f'sentence {i} to train')
            else:
                test_set += cls1_instances[i] + cls2_instances[i]
                print(f'sentence {i} to test')

    else: 
        # is_lexical_split = True
        # There cannot be more than one instance coming from the same sentence
        # We must make sure that we split lexically, ie., all instances of a single word goes to the same set

        assignments = {word: 'train' if (random.uniform(0, 1) < 1 - TEST_RATIO) else word: 'test' for word in vocabulary}
        for i in range(len(n_sentences)):
            # assuming there is a single word coming from every sentence!
            # otherwise we wont use this constraint.
            if assignments[cls1_words[i]] == 'train':
                train_set += cls1_instances[i] + cls2_instances[i]
                print(f'word {cls1_words[i]} to train')
            else:
                test_set += cls1_instances[i] + cls2_instances[i]
                print(f'word {cls1_words[i]} to test')

    return train_set, test_set

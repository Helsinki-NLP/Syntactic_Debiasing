import sys
import os
import numpy as np
import random
import logging
import pickle

TEST_RATIO = 0.3

def train_test_split(model, cls1_instances, cls2_instances, cls1_words, cls2_words, datasets, foci, is_lexical_split, is_random_labels, experiment_number, language=None, do_save=False):
    X_train = {dataset: {} for dataset in datasets}
    X_test  = {dataset: {} for dataset in datasets}
    Y_train = {dataset: {} for dataset in datasets}
    Y_test  = {dataset: {} for dataset in datasets}

    cls1_train_instances = {dataset: {} for dataset in datasets}
    cls2_train_instances = {dataset: {} for dataset in datasets}
    cls1_test_instances  = {dataset: {} for dataset in datasets}
    cls2_test_instances  = {dataset: {} for dataset in datasets}

    ENC_DIM = cls1_instances[datasets[0]][foci[0]][0].shape[2]

    for dataset in datasets:
        for focus in foci:
            X_train[dataset][focus] = np.array([])
            X_test[dataset][focus] = np.array([])

            Y_train[dataset][focus] = np.array([])
            Y_test[dataset][focus] = np.array([])

            cls1_train_instances[dataset][focus] = []
            cls2_train_instances[dataset][focus] = []
            cls1_test_instances[dataset][focus] = []
            cls2_test_instances[dataset][focus] = []

            n_sentences  = len(cls1_instances[dataset][focus])
            vocabulary = set([word for sentence in cls1_words[dataset][focus] for word in sentence])


            # lexical split will be used only for single-word instances.
            # therefore there is no need to propagate all sentence if lexical split is used

            # Also make sure that both class 1 and class 2 of the same example goes into the same set.

            if not is_lexical_split and not is_random_labels: 
                # is_lexical_split = False
                # There can be more than one instance coming from the same sentence
                # We must make sure all sentence goes into the same set

                for i in range(n_sentences):
                    cls1_instance = cls1_instances[dataset][focus][i][:,:,:].detach().numpy()
                    cls2_instance = cls2_instances[dataset][focus][i][:,:,:].detach().numpy()
                    
                    new_X = np.concatenate([cls1_instance, cls2_instance], axis=0)
                    new_Y = np.concatenate([np.zeros(cls1_instance.shape[0], dtype=int), np.ones(cls1_instance.shape[0], dtype=int)])

                    if random.uniform(0, 1) < 1 - TEST_RATIO:
                        X_train[dataset][focus] = np.concatenate([X_train[dataset][focus], new_X], axis=0) if X_train[dataset][focus].size else new_X
                        Y_train[dataset][focus] = np.concatenate([Y_train[dataset][focus], new_Y]) if Y_train[dataset][focus].size else new_Y

                        cls1_train_instances[dataset][focus].append(cls1_instance)
                        cls2_train_instances[dataset][focus].append(cls2_instance)

                    else:
                        X_test[dataset][focus] = np.concatenate([X_test[dataset][focus], new_X], axis=0) if X_test[dataset][focus].size else new_X
                        Y_test[dataset][focus] = np.concatenate([Y_test[dataset][focus], new_Y]) if Y_test[dataset][focus].size else new_Y

                        cls1_test_instances[dataset][focus].append(cls1_instance)
                        cls2_test_instances[dataset][focus].append(cls2_instance)


            elif is_lexical_split and not is_random_labels:
                # is_lexical_split = True
                # There cannot be more than one instance coming from the same sentence
                # We must make sure that we split lexically, ie., all instances of a single word goes to the same set

                assignments = {word: ('train' if (random.uniform(0, 1) < 1 - TEST_RATIO) else 'test') for word in vocabulary}
                
                for i in range(n_sentences):
                    cls1_instance = cls1_instances[dataset][focus][i][:,:,:].detach().numpy()
                    cls2_instance = cls2_instances[dataset][focus][i][:,:,:].detach().numpy()

                    new_X = np.concatenate([cls1_instance, cls2_instance], axis=0)
                    new_Y = np.concatenate([np.zeros(cls1_instance.shape[0], dtype=int), np.ones(cls2_instance.shape[0], dtype=int)])
                    
                    # assuming there is a single word coming from every sentence!
                    # otherwise we wont use this constraint.
                    single_cls1_word = cls1_words[dataset][focus][i][0]

                    if assignments[single_cls1_word] == 'train':
                        X_train[dataset][focus] = np.concatenate([X_train[dataset][focus], new_X], axis=0) if X_train[dataset][focus].size else new_X
                        Y_train[dataset][focus] = np.concatenate([Y_train[dataset][focus], new_Y]) if Y_train[dataset][focus].size else new_Y

                        cls1_train_instances[dataset][focus].append(cls1_instance)
                        cls2_train_instances[dataset][focus].append(cls2_instance)

                    else:
                        X_test[dataset][focus] = np.concatenate([X_test[dataset][focus], new_X], axis=0) if X_test[dataset][focus].size else new_X
                        Y_test[dataset][focus] = np.concatenate([Y_test[dataset][focus], new_Y]) if Y_test[dataset][focus].size else new_Y

                        cls1_test_instances[dataset][focus].append(cls1_instance)
                        cls2_test_instances[dataset][focus].append(cls2_instance)


            elif is_random_labels:
                # assign a random label to every word lexeme.

                # assuming there is a single word coming from every sentence!
                # otherwise this condition is difficult to assert.

                label_assignments = {word: ('correct' if (random.uniform(0, 1) < 1 - TEST_RATIO) else 'reverse') for word in vocabulary}
                
                for i in range(n_sentences):
                    word = cls1_words[dataset][focus][i][0]

                    cls1_instance = cls1_instances[dataset][focus][i][:,:,:].detach().numpy()
                    cls2_instance = cls2_instances[dataset][focus][i][:,:,:].detach().numpy()

                    new_X = np.concatenate([cls1_instance, cls2_instance], axis=0)
                    
                    if label_assignments[word] == 'correct':
                        new_Y = np.concatenate([np.zeros(cls1_instance.shape[0], dtype=int), np.ones(cls2_instance.shape[0], dtype=int)])
                    elif label_assignments[word] == 'reverse':
                        new_Y = np.concatenate([np.ones(cls1_instance.shape[0], dtype=int), np.zeros(cls2_instance.shape[0], dtype=int)])
                    
                    if random.uniform(0, 1) < 1 - TEST_RATIO:
                        X_train[dataset][focus] = np.concatenate([X_train[dataset][focus], new_X], axis=0) if X_train[dataset][focus].size else new_X
                        Y_train[dataset][focus] = np.concatenate([Y_train[dataset][focus], new_Y]) if Y_train[dataset][focus].size else new_Y

                        cls1_train_instances[dataset][focus].append(cls1_instance)
                        cls2_train_instances[dataset][focus].append(cls2_instance)                    

                    else:
                        X_test[dataset][focus] = np.concatenate([X_test[dataset][focus], new_X], axis=0) if X_test[dataset][focus].size else new_X
                        Y_test[dataset][focus] = np.concatenate([Y_test[dataset][focus], new_Y]) if Y_test[dataset][focus].size else new_Y

                        cls1_test_instances[dataset][focus].append(cls1_instance)
                        cls2_test_instances[dataset][focus].append(cls2_instance)                    



            # save the splits:
            if do_save:
                with open(f'../splits/{model}/{language}/X_train_{dataset}_exp_{experiment_number}_{focus}.pkl', 'wb') as fout: pickle.dump(X_train[dataset][focus], fout)
                with open(f'../splits/{model}/{language}/Y_train_{dataset}_exp_{experiment_number}_{focus}.pkl', 'wb') as fout: pickle.dump(Y_train[dataset][focus], fout)
                with open(f'../splits/{model}/{language}/X_test_{dataset}_exp_{experiment_number}_{focus}.pkl', 'wb') as fout: pickle.dump(X_test[dataset][focus], fout)
                with open(f'../splits/{model}/{language}/Y_test_{dataset}_exp_{experiment_number}_{focus}.pkl', 'wb') as fout: pickle.dump(Y_test[dataset][focus], fout)

                with open(f'../splits/{model}/{language}/cls1_train_instances_{dataset}_exp_{experiment_number}_{focus}.pkl', 'wb') as fout: pickle.dump(cls1_train_instances[dataset][focus], fout)
                with open(f'../splits/{model}/{language}/cls2_train_instances_{dataset}_exp_{experiment_number}_{focus}.pkl', 'wb') as fout: pickle.dump(cls2_train_instances[dataset][focus], fout)
                with open(f'../splits/{model}/{language}/cls1_test_instances_{dataset}_exp_{experiment_number}_{focus}.pkl', 'wb') as fout: pickle.dump(cls1_test_instances[dataset][focus], fout)
                with open(f'../splits/{model}/{language}/cls2_test_instances_{dataset}_exp_{experiment_number}_{focus}.pkl', 'wb') as fout: pickle.dump(cls2_test_instances[dataset][focus], fout)

    return X_train, Y_train, X_test, Y_test, cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances




def load_splits(model, datasets, foci, experiment_number, language=None):
    X_train = {dataset: {} for dataset in datasets}
    X_test  = {dataset: {} for dataset in datasets}
    Y_train = {dataset: {} for dataset in datasets}
    Y_test  = {dataset: {} for dataset in datasets}

    cls1_train_instances = {dataset: {} for dataset in datasets}
    cls2_train_instances = {dataset: {} for dataset in datasets}
    cls1_test_instances  = {dataset: {} for dataset in datasets}
    cls2_test_instances  = {dataset: {} for dataset in datasets}

    # load the splits:
    import pickle
    for dataset in datasets:
        for focus in foci:
            logging.debug(f'../splits/{model}/{language}/X_train_{dataset}_exp_{experiment_number}_{focus}.pkl')
            with open(f'../splits/{model}/{language}/X_train_{dataset}_exp_{experiment_number}_{focus}.pkl', 'rb') as fin: X_train[dataset][focus] = pickle.load(fin)
            with open(f'../splits/{model}/{language}/Y_train_{dataset}_exp_{experiment_number}_{focus}.pkl', 'rb') as fin: Y_train[dataset][focus] = pickle.load(fin)
            with open(f'../splits/{model}/{language}/X_test_{dataset}_exp_{experiment_number}_{focus}.pkl', 'rb') as fin:  X_test[dataset][focus] = pickle.load(fin)
            with open(f'../splits/{model}/{language}/Y_test_{dataset}_exp_{experiment_number}_{focus}.pkl', 'rb') as fin:  Y_test[dataset][focus] = pickle.load(fin)

            with open(f'../splits/{model}/{language}/cls1_train_instances_{dataset}_exp_{experiment_number}_{focus}.pkl', 'rb') as fin: cls1_train_instances[dataset][focus] = pickle.load(fin)
            with open(f'../splits/{model}/{language}/cls2_train_instances_{dataset}_exp_{experiment_number}_{focus}.pkl', 'rb') as fin: cls2_train_instances[dataset][focus] = pickle.load(fin)
            with open(f'../splits/{model}/{language}/cls1_test_instances_{dataset}_exp_{experiment_number}_{focus}.pkl', 'rb') as fin:  cls1_test_instances[dataset][focus] = pickle.load(fin)
            with open(f'../splits/{model}/{language}/cls2_test_instances_{dataset}_exp_{experiment_number}_{focus}.pkl', 'rb') as fin:  cls2_test_instances[dataset][focus] = pickle.load(fin)

    return X_train, Y_train, X_test, Y_test, cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances





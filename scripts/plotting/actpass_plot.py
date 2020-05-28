#!/usr/bin/python3

import sys
import logging
import pickle
import numpy as np
import sklearn.linear_model
from sklearn.manifold import MDS, Isomap
from sklearn.decomposition import PCA
import scipy
from scipy import linalg, matrix

import matplotlib
from matplotlib import pyplot as plt


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    active_fname = '../representations/RNN-Priming/active-passive/subject/rnn.active.single.pkl'
    passive_fname = '../representations/RNN-Priming/active-passive/subject/rnn.passive.single.pkl'
    words_fname = '../representations/RNN-Priming/active-passive/subject/words.txt'

    #active_fname = '../representations/SICK/active-passive/verb/simple-past/SICK.active.single.pkl'
    #passive_fname = '../representations/SICK/active-passive/verb/simple-past/SICK.passive.single.pkl'
    #words_fname = '../representations/SICK/active-passive/verb/simple-past/words.txt'

    #active_fname = '../representations/SICK/active-passive/object/SICK.active.single.pkl'
    #passive_fname = '../representations/SICK/active-passive/object/SICK.passive.single.pkl'
    #words_fname = '../representations/SICK/active-passive/object/words.txt'
    
    with open(active_fname, 'rb') as fobj:
        active_data = pickle.load(fobj)
        logger.info(active_data.shape)
    with open(passive_fname, 'rb') as fobj:
        passive_data = pickle.load(fobj)
        logger.info(passive_data.shape)
    with open(words_fname, 'r') as fobj:
        words = [line.strip() for line in fobj]
        logger.info(len(words))
        
    layer = 0
    fig, axes = plt.subplots(4, 3, figsize=(10, 8))
    flataxes = [ax for tmp in axes for ax in tmp]
    setsize = 1000
    #setsize = 50

    
    word_set = set()
    for word in words:
        word_set.add(word)
    word_list = list(word_set)

    if '--unique' in sys.argv:
        unique_ids = [words.index(word) for word in word_list]
        words = [words[uid] for uid in unique_ids]
        active_data = np.stack([active_data[uid,:,:] for uid in unique_ids], axis=0)
        passive_data = np.stack([passive_data[uid,:,:] for uid in unique_ids], axis=0)
        setsize = len(unique_ids)

        print(active_data.shape)

    if '--colored' in sys.argv:    	
        random_colors = [np.random.rand(3,) for i in range(len(word_list))]
        colors = [random_colors[word_list.index(word)] for word in words[:setsize]] + \
                 [random_colors[word_list.index(word)] for word in words[:setsize]]
    else:
        colors = ['red']*setsize + ['blue']*setsize


    for layer in range(0, 12):
        X = np.r_[active_data[:setsize,layer,:], passive_data[:setsize,layer,:]]
        logger.info(X.shape)
        embedding = MDS(n_components=2)
        #embedding = PCA(n_components=2)
        #embedding = Isomap(n_components=2)
        X_transformed = embedding.fit_transform(X)
        logger.info(X_transformed.shape)
        ax = flataxes[layer]
        ax.scatter(X_transformed[:,0], X_transformed[:,1], s=2, c=colors)
    
    plt.savefig('../results/visualization/RNN_subjects_MDS_uncolored.png')

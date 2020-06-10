import sys
import logging
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import debiasing
import matplotlib.colors as colors

N_LAYERS = 12
ENC_DIM = 768

class Vectorize():
    # FIXME: assumes one single word from each sentence at the moment.

    def __init__(self, cls1_instances, cls2_instances, cls1_words, cls2_words, datasets):
        self.cls1_instances = {}
        self.cls2_instances = {}

        for dataset in datasets:
            self.cls1_instances[dataset] = [cls1_instance[:,:,:].detach().numpy() \
                                                for cls1_instance in cls1_instances[dataset]]
            self.cls2_instances[dataset] = [cls2_instance[:,:,:].detach().numpy() \
                                                for cls2_instance in cls2_instances[dataset]]
        
        self.cls1_words = cls1_words
        self.cls2_words = cls2_words

        self.datasets = datasets
        self.vocabulary = set([item[0] for dataset in datasets for item in cls1_words[dataset]] \
                            + [item[0] for dataset in datasets for item in cls2_words[dataset]])

        self.diffvectors = {dataset: np.array([]) for dataset in datasets}
        self.diffclasses = {dataset: {word: np.array([]) for word in self.vocabulary} for dataset in datasets}



    def extract_diffvectors(self, datasets, layer, plotting_on='True'):
        print('extracting')
        for dataset in datasets:
            for sidx, sentence in enumerate(self.cls1_words[dataset]):
                for widx, word in enumerate(sentence):
                    
                    diffvector = np.array(self.cls1_instances[dataset][sidx][widx, layer, :].reshape(-1,ENC_DIM)) \
                                        - np.array(self.cls2_instances[dataset][sidx][widx, layer, :].reshape(-1, ENC_DIM))
                    
                    self.diffvectors[dataset] = np.concatenate([self.diffvectors[dataset], diffvector], axis=0) \
                                                        if self.diffvectors[dataset].size else diffvector
                    self.diffclasses[dataset][word] = np.concatenate([self.diffclasses[dataset][word], diffvector], axis=0) \
                                                                if self.diffclasses[dataset][word].size else diffvector

                    logging.debug(self.diffvectors[dataset].shape)


            if plotting_on:
                #self.plot_diffvectors(datasets)
                self.plot_instances_vs_diffvectors(datasets, layer)



    def plot_diffvectors(self, datasets):
        fig=plt.figure()
        colors = ['r', 'b', 'g', 'k']
        setsize = 276 #len(self.diffvectors[datasets[0]])
        
        if len(datasets) == 1:
            X = np.r_[self.diffvectors[datasets[0]][:setsize, :]]
        elif len(datasets) == 2:
            X = np.r_[self.diffvectors[datasets[0]][:setsize, :], self.diffvectors[datasets[1]][:setsize, :]]
            
        mds = MDS(n_components=2)
        X_transformed = mds.fit_transform(X)

        colors = ['red']*setsize + ['blue']*setsize

        plt.scatter(X_transformed[:,0], X_transformed[:,1], color=colors)
        plt.show()




    def plot_instances_vs_diffvectors(self, datasets, layer):
        for dataset in datasets:
            logging.debug('self.diffvectors[dataset].shape', self.diffvectors[dataset].shape)
            fig=plt.figure()
            colors = ['r', 'b', 'g', 'k']
            setsize = 276 #len(self.diffvectors[datasets[0]])
            
            cls1 = np.concatenate(self.cls1_instances[dataset][:setsize], axis=0).reshape(-1,ENC_DIM)
            cls2 = np.concatenate(self.cls2_instances[dataset][:setsize], axis=0).reshape(-1,ENC_DIM)

            logging.debug('self.diffvectors[dataset].shape', self.diffvectors[dataset].shape)
            mean_diffvector = self.diffvectors[dataset][3,:].reshape(-1,ENC_DIM) #np.mean(self.diffvectors[dataset], axis=0).reshape(-1,ENC_DIM)
            logging.debug('mean_diffvector.shape', mean_diffvector.shape)            

            logging.debug(cls1.shape)
            logging.debug(cls2.shape)
            logging.debug(mean_diffvector.shape)
            X = np.r_[cls1, cls2, mean_diffvector]
                
            transform = MDS(n_components=2)
            X_transformed = transform.fit_transform(X)

            colors = ['red']*setsize + ['blue']*setsize + ['green']

            plt.scatter(X_transformed[:,0], X_transformed[:,1], color=colors)
            plt.title(f'{dataset} Dataset')
            plt.show()



    def plot_word_senses(self, word, dataset, with_debiasing=None):
            
            
            # find random word
            # take the 'mean' word 

            # cleaned corresponding occurrences of word
            # cleaned random word
            # 'mean' of cleaned words

            # multiplying single words with P, or taking cleaned vectors matrix and indexing from there?

            # take corresponding occurrences of word
            logging.debug([i for i in range(len(self.cls1_words[dataset])) if word in self.cls1_words[dataset][i]])

            # find sentence_idx, word_idx pairs for various cases:
            all_occurrences = [(i, self.cls1_words[dataset][i].index(word)) for i in range(len(self.cls1_words[dataset])) if word in self.cls1_words[dataset][i]]
            corresponding_occurrence = random.choice(all_occurrences)
            random_occurrence = random.choice(list(set(all_occurrences).difference(set([corresponding_occurrence]))))

            not_occurrences = [(i, j) for i in range(len(self.cls1_words[dataset])) for j in range(len(self.cls1_words[dataset][i])) if word not in self.cls1_words[dataset][i]]
            random_not_occurrence = random.choice(not_occurrences)

            # Before-Cleaning Vectors
            sidx, widx = corresponding_occurrence
            vec_corr_1 = self.cls1_instances[dataset][sidx][widx,:,:]
            vec_corr_2 = self.cls2_instances[dataset][sidx][widx,:,:]

            sidx, widx = random_occurrence
            vec_random_occ = self.cls1_instances[dataset][sidx][widx,:,:]

            sidx, widx = random_not_occurrence
            vec_random_not_occ = self.cls1_instances[dataset][sidx][widx,:,:]

            vec_mean = np.zeros((1,N_LAYERS,ENC_DIM))
            for sidx in range(len(self.cls1_instances[dataset])):
                    vec_mean += np.mean(self.cls1_instances[dataset][sidx], axis=0) / len(self.cls1_instances[dataset])
            vec_mean = vec_mean.reshape(N_LAYERS,ENC_DIM)
                     
            dist_corr1_corr2          = self.layerwise_eucdist(vec_corr_1, vec_corr_2)
            dist_corr1_random_occ     = self.layerwise_eucdist(vec_corr_1, vec_random_occ)
            dist_corr1_random_not_occ = self.layerwise_eucdist(vec_corr_1, vec_random_not_occ)
            dist_corr1_mean           = self.layerwise_eucdist(vec_corr_1, vec_mean)


            if with_debiasing:
                # After-Cleaning Vectos:
                clean_corr_1          = with_debiasing.clean_data(vec_corr_1)
                clean_corr_2          = with_debiasing.clean_data(vec_corr_2)
                clean_random_occ      = with_debiasing.clean_data(vec_random_occ)
                clean_random_not_occ  = with_debiasing.clean_data(vec_random_not_occ)
                clean_mean            = with_debiasing.clean_data(vec_mean)

                dist_cleancorr1_cleancorr2           = self.layerwise_eucdist(clean_corr_1, clean_corr_2)
                dist_cleancorr1_clean_random_occ     = self.layerwise_eucdist(clean_corr_1, clean_random_occ)
                dist_cleancorr1_clean_random_not_occ = self.layerwise_eucdist(clean_corr_1, clean_random_not_occ)
                dist_cleancorr1_cleanmean            = self.layerwise_eucdist(clean_corr_1, clean_mean)


            # Plot all
            fig=plt.figure()

            plt.plot(range(N_LAYERS), dist_corr1_corr2, 'r', linestyle='solid', label='word_1a - word_1p')
            plt.plot(range(N_LAYERS), dist_corr1_random_occ, 'b', linestyle='solid', label='word_1a - word_2a')
            plt.plot(range(N_LAYERS), dist_corr1_random_not_occ, 'g', linestyle='solid', label='word_1a - nonword_a')
            plt.plot(range(N_LAYERS), dist_corr1_mean, 'k', linestyle='solid', label='word1a - mean')

            if with_debiasing:
                plt.plot(range(N_LAYERS), dist_cleancorr1_cleancorr2, 'r', linestyle='dotted',label='cln_word_1a - cln_word_1p')
                plt.plot(range(N_LAYERS), dist_cleancorr1_clean_random_occ, 'b', linestyle='dotted', label='cln_word_1a - cln_word_2a')
                plt.plot(range(N_LAYERS), dist_cleancorr1_clean_random_not_occ, 'g', linestyle='dotted', label='cln_word_1a - cln_nonword_a')
                plt.plot(range(N_LAYERS), dist_cleancorr1_cleanmean, 'k', linestyle='dotted', label='cln_word1a - cln_mean')

            plt.title(f'{dataset} dataset: "{word}"')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='x-small', ncol=2, fancybox=True, shadow=True)
            plt.show()


    def plot_distances_to_diff(self, ref_word, dataset):

        # find sentence_idx, word_idx pairs for various cases:
        all_occurrences = [(i, self.cls1_words[dataset][i].index(ref_word)) for i in range(len(self.cls1_words[dataset])) if ref_word in self.cls1_words[dataset][i]]
        
        corresponding_occurrence = random.choice(all_occurrences)
        sidx, widx = corresponding_occurrence
        vec_corr_1 = self.cls1_instances[dataset][sidx][widx,:,:]
        vec_corr_2 = self.cls2_instances[dataset][sidx][widx,:,:]
        diff_corr1_corr2 = vec_corr_1 - vec_corr_2

        random_sidx = random.choice(list(set(range(len(self.cls1_instances[dataset]))).difference(set([corresponding_occurrence[0]]))))
        
        fig=plt.figure()
        colors = ['r', 'b', 'g', 'k', 'c', 'm', 'salmon', 'darkgreen', 'maroon', 'teal', 'purple', 'gold', 'slateblue', 'plum']

        for widx, word in enumerate(self.cls1_words[dataset][random_sidx]):
            vec_word = self.cls1_instances[dataset][random_sidx][widx,:,:]
            dist_word = self.layerwise_eucdist(diff_corr1_corr2, vec_word)
            plt.plot(range(N_LAYERS), dist_word, c=colors[widx], linestyle='solid', label=f'{word} [act]')

        for widx, word in enumerate(self.cls2_words[dataset][random_sidx]):
            vec_word = self.cls2_instances[dataset][random_sidx][widx,:,:]
            dist_word = self.layerwise_eucdist(diff_corr1_corr2, vec_word)
            plt.plot(range(N_LAYERS), dist_word, c=colors[widx], linestyle='dotted', label=f'{word} [pass]')

        plt.title(f'{dataset} dataset: "{ref_word}[act] - {ref_word}[pass]"')
        plt.legend(loc='right', bbox_to_anchor=(1.05, 0), fontsize='x-small', ncol=10, fancybox=True, shadow=True)
        plt.show()
            


    def layerwise_eucdist(self, vec1, vec2):
        return [np.linalg.norm(vec1[layer,:] - vec2[layer,:]) for layer in range(N_LAYERS)]
            
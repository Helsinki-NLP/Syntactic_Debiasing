import sys
import logging
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import debiasing
import matplotlib.colors as colors

N_LAYERS = 12
ENC_DIM = 768

class Vectorize():
    # FIXME: assumes one single word from each sentence at the moment.

    def __init__(self):

        self.avg_sim_corr1_corr2 = np.array([])
        self.avg_sim_corr1_random_occ = np.array([])
        self.avg_sim_corr1_random_not_occ = np.array([])
        self.avg_sim_corr1_active = np.array([])
        self.avg_sim_corr1_any = np.array([])

        self.avg_sim_cleancorr1_cleancorr2 = np.array([])
        self.avg_sim_cleancorr1_clean_random_occ = np.array([])
        self.avg_sim_cleancorr1_clean_random_not_occ = np.array([])
        self.avg_sim_cleancorr1_clean_active = np.array([])
        self.avg_sim_cleancorr1_clean_any = np.array([])



    def set_data(self, cls1_instances, cls2_instances, cls1_words, cls2_words, cls1_ids, cls2_ids, datasets):
        self.cls1_instances = {}
        self.cls2_instances = {}

        for dataset in datasets:
            self.cls1_instances[dataset] = [cls1_instance[:,:,:].detach().numpy() \
                                                for cls1_instance in cls1_instances[dataset]]
            self.cls2_instances[dataset] = [cls2_instance[:,:,:].detach().numpy() \
                                                for cls2_instance in cls2_instances[dataset]]
        
        self.cls1_words = cls1_words
        self.cls2_words = cls2_words

        self.cls1_ids = cls1_ids
        self.cls2_ids = cls2_ids

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



    def calc_word_senses(self, word, dataset, distance_fnc, with_debiasing=None):
            
            
            # find random word
            # take the 'mean' word 

            # cleaned corresponding occurrences of word
            # cleaned random word
            # 'mean' of cleaned words

            # multiplying single words with P, or taking cleaned vectors matrix and indexing from there?

            acc_sim_corr1_corr2 = np.zeros((N_LAYERS,1))
            acc_sim_corr1_random_occ = np.zeros((N_LAYERS,1))
            acc_sim_corr1_random_not_occ = np.zeros((N_LAYERS,1))
            acc_sim_corr1_active = np.zeros((N_LAYERS,1))
            acc_sim_corr1_any = np.zeros((N_LAYERS,1))

            acc_sim_cleancorr1_cleancorr2 = np.zeros((N_LAYERS,1))
            acc_sim_cleancorr1_clean_random_occ = np.zeros((N_LAYERS,1))
            acc_sim_cleancorr1_clean_random_not_occ = np.zeros((N_LAYERS,1))
            acc_sim_cleancorr1_clean_active = np.zeros((N_LAYERS,1))
            acc_sim_cleancorr1_clean_any = np.zeros((N_LAYERS,1))

            word_count = 0
            for si, sentence in enumerate(self.cls1_words[dataset]):
                for wi, word in enumerate(sentence):
                    #print(wi+1, self.cls1_ids[dataset][si][0])
                    #if wi+1 == self.cls1_ids[dataset][si][0]:
                        print('word is ', word)
                        word_count += 1

                        # take corresponding occurrences of word
                        logging.debug([i for i in range(len(self.cls1_words[dataset])) if word in self.cls1_words[dataset][i]])

                        # find sentence_idx, word_idx pairs for various cases:
                        all_occurrences = [(i, self.cls1_words[dataset][i].index(word)) for i in range(len(self.cls1_words[dataset])) if word in self.cls1_words[dataset][i]]
                        print(all_occurrences)
                        if len(all_occurrences) == 1:
                            continue

                        corresponding_occurrence = random.choice(all_occurrences)
                        random_occurrence = random.choice(list(set(all_occurrences).difference(set([corresponding_occurrence]))))

                        not_occurrences = [(i, j) for i in range(len(self.cls1_words[dataset])) for j in range(len(self.cls1_words[dataset][i])) if word not in self.cls1_words[dataset][i]]
                        random_not_occurrence = random.choice(not_occurrences)

                        random_active_sidx = random.choice(range(len(self.cls1_words[dataset])))
                        random_active_widx = random.choice(range(len(self.cls1_words[dataset][random_active_sidx])))

                        random_any_class = random.choice([1, 2])
                        if random_any_class == 1:
                            random_any_sidx = random.choice(range(len(self.cls1_words[dataset])))
                            random_any_widx = random.choice(range(len(self.cls1_words[dataset][random_active_sidx])))
                        else:
                            random_any_sidx = random.choice(range(len(self.cls2_words[dataset])))
                            random_any_widx = random.choice(range(len(self.cls2_words[dataset][random_active_sidx])))

                        # Before-Cleaning Vectors
                        sidx, widx = corresponding_occurrence
                        vec_corr_1 = self.cls1_instances[dataset][sidx][widx,:,:]
                        print('vec_corr_1', vec_corr_1)
                        vec_corr_2 = self.cls2_instances[dataset][sidx][widx,:,:]
                        print('vec_corr_2', vec_corr_2)

                        sidx, widx = random_occurrence
                        vec_random_occ = self.cls1_instances[dataset][sidx][widx,:,:]

                        sidx, widx = random_not_occurrence
                        vec_random_not_occ = self.cls1_instances[dataset][sidx][widx,:,:]


                        #vec_mean = np.zeros((1,N_LAYERS,ENC_DIM))
                        #for sidx in range(len(self.cls1_instances[dataset])):
                        #        vec_mean += np.mean(self.cls1_instances[dataset][sidx], axis=0) / len(self.cls1_instances[dataset])
                        #vec_mean = vec_mean.reshape(N_LAYERS,ENC_DIM)
                        sidx, widx = random_active_sidx, random_active_widx
                        vec_random_active = self.cls1_instances[dataset][random_active_sidx][random_active_widx,:,:]

                        sidx, widx = random_any_sidx, random_any_widx
                        if random_any_class == 1:
                            vec_random_any = self.cls1_instances[dataset][random_active_sidx][random_active_widx,:,:]                        
                        else:
                            vec_random_any = self.cls2_instances[dataset][random_active_sidx][random_active_widx,:,:]                        

                        acc_sim_corr1_corr2          += distance_fnc(vec_corr_1, vec_corr_2)
                        acc_sim_corr1_random_occ     += distance_fnc(vec_corr_1, vec_random_occ)
                        acc_sim_corr1_random_not_occ += distance_fnc(vec_corr_1, vec_random_not_occ)
                        acc_sim_corr1_active         += distance_fnc(vec_corr_1, vec_random_active)
                        acc_sim_corr1_any            += distance_fnc(vec_corr_1, vec_random_any)


                        if with_debiasing:
                            # After-Cleaning Vectos:
                            clean_corr_1          = with_debiasing.clean_data(vec_corr_1)
                            print('clean_corr_1', clean_corr_1)
                            clean_corr_2          = with_debiasing.clean_data(vec_corr_2)
                            print('clean_corr_2', clean_corr_2)
                            clean_random_occ      = with_debiasing.clean_data(vec_random_occ)
                            clean_random_not_occ  = with_debiasing.clean_data(vec_random_not_occ)
                            clean_active             = with_debiasing.clean_data(vec_random_active)
                            clean_any             = with_debiasing.clean_data(vec_random_any)


                            acc_sim_cleancorr1_cleancorr2           += distance_fnc(clean_corr_1, clean_corr_2)
                            acc_sim_cleancorr1_clean_random_occ     += distance_fnc(clean_corr_1, clean_random_occ)
                            acc_sim_cleancorr1_clean_random_not_occ += distance_fnc(clean_corr_1, clean_random_not_occ)
                            acc_sim_cleancorr1_clean_active         += distance_fnc(clean_corr_1, clean_active)
                            acc_sim_cleancorr1_clean_any            += distance_fnc(clean_corr_1, clean_any)


            
            if self.avg_sim_corr1_corr2.size == 0:
                self.avg_sim_corr1_corr2                     = acc_sim_corr1_corr2 / word_count
                self.avg_sim_corr1_random_occ                = acc_sim_corr1_random_occ / word_count
                self.avg_sim_corr1_random_not_occ            = acc_sim_corr1_random_not_occ / word_count
                self.avg_sim_corr1_active                    = acc_sim_corr1_active / word_count
                self.avg_sim_corr1_any                       = acc_sim_corr1_any / word_count

                self.avg_sim_cleancorr1_cleancorr2           = acc_sim_cleancorr1_cleancorr2 / word_count
                self.avg_sim_cleancorr1_clean_random_occ     = acc_sim_cleancorr1_clean_random_occ / word_count
                self.avg_sim_cleancorr1_clean_random_not_occ = acc_sim_cleancorr1_clean_random_not_occ / word_count
                self.avg_sim_cleancorr1_clean_active         = acc_sim_cleancorr1_clean_active / word_count
                self.avg_sim_cleancorr1_clean_any            = acc_sim_cleancorr1_clean_any / word_count

            else:
                print(self.avg_sim_corr1_corr2.shape)
                print(acc_sim_corr1_corr2.shape)
                self.avg_sim_corr1_corr2          = np.concatenate((self.avg_sim_corr1_corr2, (acc_sim_corr1_corr2 / word_count)), axis=1) 
                self.avg_sim_corr1_random_occ     = np.concatenate((self.avg_sim_corr1_random_occ, (acc_sim_corr1_random_occ / word_count)), axis=1)
                self.avg_sim_corr1_random_not_occ = np.concatenate((self.avg_sim_corr1_random_not_occ, (acc_sim_corr1_random_not_occ / word_count)), axis=1)
                self.avg_sim_corr1_active         = np.concatenate((self.avg_sim_corr1_active, (acc_sim_corr1_active / word_count)), axis=1)
                self.avg_sim_corr1_any            = np.concatenate((self.avg_sim_corr1_any, (acc_sim_corr1_any / word_count)), axis=1)

                self.avg_sim_cleancorr1_cleancorr2           = np.concatenate((self.avg_sim_cleancorr1_cleancorr2, (acc_sim_cleancorr1_cleancorr2 / word_count)), axis=1)
                self.avg_sim_cleancorr1_clean_random_occ     = np.concatenate((self.avg_sim_cleancorr1_clean_random_occ, (acc_sim_cleancorr1_clean_random_occ / word_count)), axis=1)
                self.avg_sim_cleancorr1_clean_random_not_occ = np.concatenate((self.avg_sim_cleancorr1_clean_random_not_occ, (acc_sim_cleancorr1_clean_random_not_occ / word_count)), axis=1)
                self.avg_sim_cleancorr1_clean_active         = np.concatenate((self.avg_sim_cleancorr1_clean_active, (acc_sim_cleancorr1_clean_active / word_count)), axis=1)
                self.avg_sim_cleancorr1_clean_any            = np.concatenate((self.avg_sim_cleancorr1_clean_any, (acc_sim_cleancorr1_clean_any / word_count)), axis=1)


    def plot_word_senses(self, focus, is_with_debiasing=False):

            # Plot all
            fig=plt.figure()

            print('self.avg_sim_corr1_corr2', self.avg_sim_corr1_corr2)
            print('self.avg_sim_cleancorr1_cleancorr2', self.avg_sim_cleancorr1_cleancorr2)

            plt.plot(range(N_LAYERS), np.mean(self.avg_sim_corr1_corr2, axis=1), 'r', linestyle='dotted', label=f'{focus}_1a - {focus}_1p')
            plt.plot(range(N_LAYERS), np.mean(self.avg_sim_corr1_random_occ, axis=1), 'b', linestyle='dotted', label=f'{focus}_1a - {focus}_2a')
            plt.plot(range(N_LAYERS), np.mean(self.avg_sim_corr1_random_not_occ, axis=1), 'g', linestyle='dotted', label=f'{focus}_1a - other_verb_a')
            plt.plot(range(N_LAYERS), np.mean(self.avg_sim_corr1_active, axis=1), 'c', linestyle='dotted', label=f'{focus}_1a - random_word_a')
            plt.plot(range(N_LAYERS), np.mean(self.avg_sim_corr1_any, axis=1), 'k', linestyle='dotted', label=f'{focus}_1a - random_word')

            if is_with_debiasing:
                plt.plot(range(N_LAYERS), np.mean(self.avg_sim_cleancorr1_cleancorr2, axis=1), 'r', linestyle='solid',label=f'clean_{focus}_1a - clean_{focus}_1p')
                plt.plot(range(N_LAYERS), np.mean(self.avg_sim_cleancorr1_clean_random_occ, axis=1), 'b', linestyle='solid', label=f'clean_{focus}_1a - clean_{focus}_2a')
                plt.plot(range(N_LAYERS), np.mean(self.avg_sim_cleancorr1_clean_random_not_occ, axis=1), 'g', linestyle='solid', label=f'clean_{focus}_1a - clean_other_{focus}_a')
                plt.plot(range(N_LAYERS), np.mean(self.avg_sim_cleancorr1_clean_active, axis=1), 'c', linestyle='solid', label=f'clean_{focus}_1a - clean_random_word_a')
                plt.plot(range(N_LAYERS), np.mean(self.avg_sim_cleancorr1_clean_any, axis=1), 'k', linestyle='solid', label=f'clean_{focus}_1a - clean_random_word')

            #plt.title(f'{dataset} dataset: Average of All Verbs')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='x-small', ncol=2, fancybox=True, shadow=True)
            plt.xticks(range(N_LAYERS), range(1,N_LAYERS+1))

            plt.xlabel('Layers')
            plt.ylabel('Euclidean distance')
            plt.ylim((3, 27))
            plt.yticks([5, 10, 15, 20, 25])            
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
        return np.array([np.linalg.norm(vec1[layer,:] - vec2[layer,:]) for layer in range(N_LAYERS)]).reshape(N_LAYERS,1)

    def layerwise_cosine_sim(self, vec1, vec2):
        return np.array([cosine_similarity(vec1[layer,:].reshape(1,-1), vec2[layer,:].reshape(1,-1)) for layer in range(N_LAYERS)]).reshape(N_LAYERS,1)
            
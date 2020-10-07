import sys
import os
import logging
import torch
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import debiasing
import matplotlib.colors as colors



class Vectorize():
    # FIXME: assumes one single word from each sentence at the moment.

    def __init__(self, model):

        # RED LINE
        self.avg_dist_corr1_corr2 = np.array([])
        self.avg_dist_cleancorr1_cleancorr2 = np.array([])
        
        # BLUE LINE
        self.avg_dist_wordC_anotherC = np.array([])
        self.avg_dist_clean_wordC_clean_anotherC = np.array([])

        # GREEN LINE
        self.avg_dist_within_class = np.array([])
        self.avg_dist_clean_within_class = np.array([])

        # YELLOW LINE
        self.avg_dist_between_class = np.array([])        
        self.avg_dist_clean_between_class = np.array([])

        if model == 'BERT':
            self.n_layers = 12
            self.enc_dim = 768
        elif model == 'MT':
            self.n_layers = 6
            self.enc_dim = 512           



    def set_data(self, cls1_instances, cls2_instances, cls1_words, cls2_words, datasets, foci):
        self.cls1_instances = {}
        self.cls2_instances = {}

        for focus in foci:
            for dataset in datasets:
                self.cls1_instances[dataset] = {focus: [] for focus in foci}
                for cls1_instance in cls1_instances[dataset][focus]:
                    self.cls1_instances[dataset][focus].append(cls1_instance[:,:,:].detach().numpy())

                self.cls2_instances[dataset] = {focus: [] for focus in foci}
                for cls2_instance in cls2_instances[dataset][focus]:
                    self.cls2_instances[dataset][focus].append(cls2_instance[:,:,:].detach().numpy())

        
        self.cls1_words = cls1_words
        self.cls2_words = cls2_words

        self.datasets = datasets
        self.foci = foci

    def extract_diffvectors(self, datasets, layer, plotting_on='True'):
        print('extracting')
        for dataset in datasets:
            for sidx, sentence in enumerate(self.cls1_words[dataset]):
                for widx, word in enumerate(sentence):
                    
                    diffvector = np.array(self.cls1_instances[dataset][sidx][widx, layer, :].reshape(-1,self.enc_dim)) \
                                        - np.array(self.cls2_instances[dataset][sidx][widx, layer, :].reshape(-1, self.enc_dim))
                    
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
            
            cls1 = np.concatenate(self.cls1_instances[dataset][:setsize], axis=0).reshape(-1,self.enc_dim)
            cls2 = np.concatenate(self.cls2_instances[dataset][:setsize], axis=0).reshape(-1,self.enc_dim)

            logging.debug('self.diffvectors[dataset].shape', self.diffvectors[dataset].shape)
            mean_diffvector = self.diffvectors[dataset][3,:].reshape(-1,self.enc_dim) #np.mean(self.diffvectors[dataset], axis=0).reshape(-1,self.enc_dim)
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



    def calc_word_senses(self, dataset, focus, distance_fnc, with_debiasing=None, logfile=None):

            # RED LINE
            acc_dist_corr1_corr2 = np.zeros((self.n_layers,1))
            acc_dist_cleancorr1_cleancorr2 = np.zeros((self.n_layers,1))
            
            # BLUE LINE
            acc_dist_wordC_anotherC = np.zeros((self.n_layers,1))          
            acc_dist_clean_wordC_clean_anotherC = np.zeros((self.n_layers,1))

            # GREEN LINE
            acc_dist_within_class = np.zeros((self.n_layers,1))
            acc_dist_clean_within_class = np.zeros((self.n_layers,1))

            # YELLOW LINE
            acc_dist_between_class = np.zeros((self.n_layers,1))
            acc_dist_clean_between_class = np.zeros((self.n_layers,1))



            # ---- RED LINE ----
            # average difference between the corresponding active and passive instance
            # \mean_{i \in S} ||x_i^A - x_i^P||
            # should go smaller

            word_count = 0
            for sentence in self.cls1_words[dataset][focus]:
                for word in sentence: 
                    word_count += 1    

                    print('word is ', word)

                    # find sentence_idx, word_idx pairs for various cases:
                    all_occurrences = [(i, self.cls1_words[dataset][focus][i].index(word)) for i in range(len(self.cls1_words[dataset][focus])) if word in self.cls1_words[dataset][focus][i]]

                    corresponding_occurrence = random.choice(all_occurrences)
                    sidx, widx = corresponding_occurrence
                    vec_corr_1 = self.cls1_instances[dataset][focus][sidx][widx,:,:]
                    vec_corr_2 = self.cls2_instances[dataset][focus][sidx][widx,:,:]  

                    acc_dist_corr1_corr2 += distance_fnc(vec_corr_1, vec_corr_2)

                    clean_corr_1 = with_debiasing.clean_data(vec_corr_1)
                    clean_corr_2 = with_debiasing.clean_data(vec_corr_2)
                    acc_dist_cleancorr1_cleancorr2 += distance_fnc(clean_corr_1, clean_corr_2)

            if self.avg_dist_corr1_corr2.size == 0:
                self.avg_dist_corr1_corr2           = acc_dist_corr1_corr2 / word_count
                self.avg_dist_cleancorr1_cleancorr2 = acc_dist_cleancorr1_cleancorr2 / word_count
            else:
                self.avg_dist_corr1_corr2           = np.concatenate((self.avg_dist_corr1_corr2, (acc_dist_corr1_corr2 / word_count)), axis=1) 
                self.avg_dist_cleancorr1_cleancorr2 = np.concatenate((self.avg_dist_cleancorr1_cleancorr2, (acc_dist_cleancorr1_cleancorr2 / word_count)), axis=1)
            



            # ---- BLUE LINE ----
            # average difference between an verb and another instance of the same verb within the same class
            # \mean_{i \in S, C \in {A, P}} ||x_i^C - \mean_{j \in S: w_j = w_i, j \neq i} x_j^C||
            # should not go much smaller

            word_count = 0
            for class_id in [1, 2]:
                if class_id == 1:
                    cls_words = self.cls1_words[dataset][focus]
                    cls_instances = self.cls1_instances[dataset][focus]
                elif class_id == 2:
                    cls_words = self.cls2_words[dataset][focus]
                    cls_instances = self.cls2_instances[dataset][focus]

                for sentence in cls_words:
                    for word in sentence:  
                        word_count += 1 

                        all_occurrences = [(i, cls_words[i].index(word)) for i in range(len(cls_words)) if word in cls_words[i]]
                        if len(all_occurrences) == 1:
                            continue

                        corresponding_occurrence = random.choice(all_occurrences)
                        sidx, widx = corresponding_occurrence
                        vec_corr = cls_instances[sidx][widx,:,:]

                        other_occurrence = random.choice(list(set(all_occurrences).difference(set([corresponding_occurrence]))))
                        sidx, widx = other_occurrence
                        vec_other_occ = cls_instances[sidx][widx,:,:]
                        
                        acc_dist_wordC_anotherC += distance_fnc(vec_corr, vec_other_occ)

                        clean_corr      = with_debiasing.clean_data(vec_corr)
                        clean_other_occ = with_debiasing.clean_data(vec_other_occ)

                        acc_dist_clean_wordC_clean_anotherC += distance_fnc(clean_corr, clean_other_occ)


            if self.avg_dist_wordC_anotherC.size == 0:
                self.avg_dist_wordC_anotherC             = acc_dist_wordC_anotherC / word_count
                self.avg_dist_clean_wordC_clean_anotherC = acc_dist_clean_wordC_clean_anotherC / word_count
            else:
                self.avg_dist_wordC_anotherC             = np.concatenate((self.avg_dist_wordC_anotherC, (acc_dist_wordC_anotherC / word_count)), axis=1) 
                self.avg_dist_clean_wordC_clean_anotherC = np.concatenate((self.avg_dist_clean_wordC_clean_anotherC, (acc_dist_clean_wordC_clean_anotherC / word_count)), axis=1)



            # ---- BLACK LINE ----
            # average difference between an verb and a different verb within the same class
            # \mean_{i \in S, C \in {A,P}} ||x_i^C - \mean_{j \in S: w_j \neq w_i} x_j^C||
            # should not go much smaller

            occurrence_count = 0
            for class_id in [1, 2]:
                if class_id == 1:
                    cls_words = self.cls1_words[dataset][focus]
                    cls_instances = self.cls1_instances[dataset][focus]
                elif class_id == 2:
                    cls_words = self.cls2_words[dataset][focus]
                    cls_instances = self.cls2_instances[dataset][focus]


                '''inner loop: w_i \neq w_j condition:
                for sidx, sentence in enumerate(cls_words):
                    print('black', sidx)
                    for widx, word in enumerate(sentence):

                        vec_corr = cls_instances[sidx][widx,:,:]
                        clean_corr = with_debiasing.clean_data(vec_corr)

                        not_occurrences = [(i, j) for i in range(len(cls_words)) for j in range(len(cls_words[i])) if word not in cls_words[i]]
                        #not_occurrence = random.choice(not_occurrences)
                        
                        occurrence_count += len(not_occurrences)

                        for not_occurrence in not_occurrences:
                            sidx_2, widx_2 = not_occurrence

                            vec_not_occ = cls_instances[sidx_2][widx_2,:,:]
                            clean_not_occ = with_debiasing.clean_data(vec_not_occ)

                            acc_dist_within_class += distance_fnc(vec_corr, vec_not_occ)
                            acc_dist_clean_within_class += distance_fnc(clean_corr, clean_not_occ)
                '''


                ''' i = j condition:'''

                centroid_within_class = np.zeros((self.n_layers,self.enc_dim))

                c_wc = 0
                for sidx in range(len(cls_words)):
                    for widx in range(cls_instances[sidx].shape[0]):
                        centroid_within_class += cls_instances[sidx][widx,:,:]
                        c_wc += 1

                centroid_within_class /= c_wc
                clean_centroid_within_class = with_debiasing.clean_data(centroid_within_class)

                for sidx in range(len(cls_words)):
                    for widx in range(cls_instances[sidx].shape[0]):

                        vec_1 = cls_instances[sidx][widx,:,:]
                       
                        acc_dist_within_class += distance_fnc(vec_1, centroid_within_class)

                        clean_vec_1 = with_debiasing.clean_data(vec_1)
                        acc_dist_clean_within_class += distance_fnc(clean_vec_1, clean_centroid_within_class)

                        occurrence_count += 1
                

            if self.avg_dist_within_class.size == 0:
                self.avg_dist_within_class       = acc_dist_within_class / occurrence_count
                self.avg_dist_clean_within_class = acc_dist_clean_within_class / occurrence_count
            else:
                self.avg_dist_within_class       = np.concatenate((self.avg_dist_within_class, (acc_dist_within_class / occurrence_count)), axis=1) 
                self.avg_dist_clean_within_class = np.concatenate((self.avg_dist_clean_within_class, (acc_dist_clean_within_class / occurrence_count)), axis=1)




            # ---- YELLOW LINE ----
            # average difference between an verb and a different verb within different class
            #  \mean_{i \in S, C1 \in {A,P}} ||x_i^C1 - \mean_{j \in S, C2 \in {A, P}: w_j \neq w_i, C2 \neq C1} x_j^C2||
            # should go smaller, but not as much as RED


            occurrence_count = 0
            for class_id in [1, 2]:
                if class_id == 1:
                    cls_words = self.cls1_words[dataset][focus]
                    cls_instances = self.cls1_instances[dataset][focus]
                    not_cls_words = self.cls2_words[dataset][focus]
                    not_cls_instances = self.cls2_instances[dataset][focus]
                elif class_id == 2:
                    cls_words = self.cls2_words[dataset][focus]
                    cls_instances = self.cls2_instances[dataset][focus]
                    not_cls_words = self.cls1_words[dataset][focus]
                    not_cls_instances = self.cls1_instances[dataset][focus]                  

                '''inner loop: w_i \neq w_j condition:
                for sidx, sentence in enumerate(cls_words):
                    print('yellow', sidx)

                    for widx, word in enumerate(sentence):

                        vec_corr = cls_instances[sidx][widx,:,:]
                        clean_corr = with_debiasing.clean_data(vec_corr)

                        not_occurrences = [(i, j) for i in range(len(not_cls_words)) for j in range(len(not_cls_words[i])) if word not in not_cls_words[i]]
                        #not_occurrence = random.choice(not_occurrences)
                        
                        occurrence_count += len(not_occurrences)

                        for not_occurrence in not_occurrences:
                            sidx_2, widx_2 = not_occurrence

                            vec_not_occ = not_cls_instances[sidx_2][widx_2,:,:]
                            clean_not_occ = with_debiasing.clean_data(vec_not_occ)

                            acc_dist_between_class += distance_fnc(vec_corr, vec_not_occ)
                            acc_dist_clean_between_class += distance_fnc(clean_corr, clean_not_occ)
                '''
                ''' i = j condition:'''

                centroid_out_class = np.zeros((self.n_layers,self.enc_dim))

                c_oc = 0
                for sidx in range(len(not_cls_words)):
                    for widx in range(not_cls_instances[sidx].shape[0]):
                        centroid_out_class += not_cls_instances[sidx][widx,:,:]
                        c_oc += 1

                centroid_out_class /= c_oc
                clean_centroid_out_class = with_debiasing.clean_data(centroid_out_class)

                for sidx in range(len(cls_words)):
                    for widx in range(cls_instances[sidx].shape[0]):

                        vec_1 = cls_instances[sidx][widx,:,:]
                       
                        acc_dist_between_class += distance_fnc(vec_1, centroid_out_class)

                        clean_vec_1 = with_debiasing.clean_data(vec_1)
                        acc_dist_clean_between_class += distance_fnc(clean_vec_1, clean_centroid_out_class)

                        occurrence_count += 1
            

            if self.avg_dist_between_class.size == 0:
                self.avg_dist_between_class       = acc_dist_between_class / occurrence_count
                self.avg_dist_clean_between_class = acc_dist_clean_between_class / occurrence_count
            else:
                self.avg_dist_between_class       = np.concatenate((self.avg_dist_between_class, (acc_dist_between_class / occurrence_count)), axis=1) 
                self.avg_dist_clean_between_class = np.concatenate((self.avg_dist_clean_between_class, (acc_dist_clean_between_class / occurrence_count)), axis=1)




            if logfile:
                distances = {}
                distances['pairwise_between_class'] = self.avg_dist_corr1_corr2
                distances['pairwise_between_class_cleaned'] = self.avg_dist_cleancorr1_cleancorr2
        
                distances['same-word within-class'] = self.avg_dist_wordC_anotherC
                distances['same-word within-class_cleaned'] = self.avg_dist_clean_wordC_clean_anotherC

                # GREEN LINE
                distances['global_within_class'] = self.avg_dist_within_class
                distances['global_within_class_cleaned'] = self.avg_dist_clean_within_class

                # YELLOW LINE
                distances['global_between_class'] = self.avg_dist_between_class
                distances['global_between_class_cleaned'] = self.avg_dist_clean_between_class

                logging.debug(f'     saving distances to {logfile}')
                os.system(f'mkdir -p {os.path.dirname(logfile)}')
                with open(logfile, 'wb') as fout:
                    pickle.dump(distances, fout)





    def plot_word_senses(self, focus, dataset, is_with_debiasing=False):

        # Plot all
        if dataset[0:3] == 'SIC' and focus == 'object':
            figsize = (6, 6)
        else:
            figsize = (6, 4)
        fig=plt.figure(figsize=figsize)

        plt.plot(range(self.n_layers), np.mean(self.avg_dist_corr1_corr2, axis=1), 'r', linestyle='dotted', label=f'word1_A - word1_P')
        plt.plot(range(self.n_layers), np.mean(self.avg_dist_wordC_anotherC, axis=1), 'b', linestyle='dotted', label=f'word1_C - word2_C')
        plt.plot(range(self.n_layers), np.mean(self.avg_dist_within_class, axis=1), 'k', linestyle='dotted', label=f'word1_C - otherword_C')
        plt.plot(range(self.n_layers), np.mean(self.avg_dist_between_class, axis=1), 'y', linestyle='dotted', label=f'word_1C - otherword_notC')
        
        plt.plot(range(self.n_layers), np.mean(self.avg_dist_cleancorr1_cleancorr2, axis=1), 'r', linestyle='solid',label=f'clean_word1_A - clean_word1_P')
        plt.plot(range(self.n_layers), np.mean(self.avg_dist_clean_wordC_clean_anotherC, axis=1), 'b', linestyle='solid', label=f'clean_word1_C - clean_word2_C')
        plt.plot(range(self.n_layers), np.mean(self.avg_dist_clean_within_class, axis=1), 'k', linestyle='solid', label=f'clean_word1_C - clean_otherword_C')
        plt.plot(range(self.n_layers), np.mean(self.avg_dist_clean_between_class, axis=1), 'y', linestyle='solid', label=f'clean_word1_C - clean_otherword_notC')
        
        #plt.title(f'{dataset} dataset: Average of All Verbs')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='x-small', ncol=2, fancybox=True, shadow=True)
        plt.xticks(range(self.n_layers), range(1,self.n_layers+1))

        plt.xlabel('Layers')
        plt.ylabel('Euclidean distance')
        #plt.ylim((0, 30))
        #plt.yticks([0, 5, 10, 15, 20, 25, 30])            
        
        if dataset[0:3] == 'RNN': dataset = 'TEMP-AP'

        if focus == 'verb': plt.title(f'{dataset} dataset: VERBS')
        if focus == 'subject': plt.title(f'{dataset} dataset: A-SUBJ / P-AG')
        if focus == 'object': plt.title(f'{dataset} dataset: A-OBJ / P-SUBJ')
        plt.show()




    def plot_word_senses_from_logs(self, basedir, task, model, language, datasets, foci, do_plot_legend=True):
        logging.debug(f'in plot_word_senses_from_logs')
        n_datasets = len(datasets)
        n_foci = len(foci)

        if do_plot_legend:
            fig, axes = plt.subplots(n_datasets, n_foci, figsize=(24, 4.5))
        else:
            fig, axes = plt.subplots(n_datasets, n_foci, figsize=(24, 3))

        fig.tight_layout(pad=2, h_pad=2, w_pad=3)
        #flataxes = [ax for ax in axes for ax in tmp]
        flataxes = axes

        for di, dataset in enumerate(datasets):
            for fi, focus in enumerate(foci):
                loadfile = f'{basedir}/{model}/{language}/{dataset}_{focus}_distances.pkl'
                
                logging.debug(f'   loading distances from {loadfile}')
                with open(loadfile, 'rb') as fin:
                    distances = pickle.load(fin)

                subi = di * n_foci + fi
                ax = flataxes[subi]

                if model == 'BERT':
                    n_layers = 12
                else:
                    n_layers = 6

                ax.plot(range(n_layers), np.mean(distances['pairwise_between_class'], axis=1), 'r', linestyle='dotted', label=f'pairwise inter-class distances (original)', linewidth=3)
                ax.plot(range(n_layers), np.mean(distances['global_between_class'], axis=1), 'y', linestyle='dotted', label=f'global inter-class distances (original)', linewidth=3)
                ax.plot(range(n_layers), np.mean(distances['same-word within-class'], axis=1), 'b', linestyle='dotted', label=f'same-word intra-class distances (original)', linewidth=3)
                ax.plot(range(n_layers), np.mean(distances['global_within_class'], axis=1), 'k', linestyle='dotted', label=f'global intra-class distances (original)', linewidth=3)
                
                ax.plot(range(n_layers), np.mean(distances['pairwise_between_class_cleaned'], axis=1), 'r', linestyle='solid',alpha=0.5, label=f'pairwise inter-class distances (cleaned)', linewidth=3)
                ax.plot(range(n_layers), np.mean(distances['global_between_class_cleaned'], axis=1), 'y', linestyle='solid', label=f'global inter-class distances (cleaned)', linewidth=3)
                ax.plot(range(n_layers), np.mean(distances['same-word within-class_cleaned'], axis=1), 'b', linestyle='solid', label=f'same-word intra-class distances (cleaned)', linewidth=3)                
                ax.plot(range(n_layers), np.mean(distances['global_within_class_cleaned'], axis=1), 'k', linestyle='solid', alpha=0.5, label=f'global intra-class distances (cleaned)', linewidth=3)
                
                ax.set_xticks(range(n_layers))
                ax.set_xticklabels(range(1,n_layers+1))

                if subi >= n_foci * (n_datasets - 1):
                    ax.set_xlabel('Layers', fontsize=16)

                if subi % n_foci == 0:
                    ax.set_ylabel('Euclidean distance', fontsize=16)

                if model == 'BERT':
                    ax.set_ylim((0, 30))
                    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])   
                if model == 'MT':
                    ax.set_ylim((0, 4))
                    ax.set_yticks([0, 1, 2, 3, 4])   

                ax.tick_params(labelsize=16)
                    
                if dataset[0:3] == 'RNN':
                    if task == 'active-passive':
                        ds_alias = 'TEMPL-PAS'
                    elif task == 'positive-negative':
                        ds_alias = 'TEMPL-NEG'
                else:
                    ds_alias = 'SICK'

                if model == 'BERT':
                    model_alias = 'BERT'
                if model == 'MT' and language == 'DE':
                    model_alias = 'MT (EN > DE)'
                if model == 'MT' and language == 'DE-EL':
                    model_alias = 'MT (EN > DE+EL)'                    

                if task == 'active-passive':
                    if focus == 'verb': ax.set_title(f'{model_alias} - VERB', fontsize=16)
                    if focus == 'subject': ax.set_title(f'{model_alias} - A-SUBJ / P-AG', fontsize=16)
                    if focus == 'object': ax.set_title(f'{model_alias} - A-OBJ / P-SUBJ', fontsize=16)
                elif task == 'positive-negative':
                    if focus == 'verb': ax.set_title(f'{model_alias} - VERB', fontsize=16)
                    if focus == 'subject': ax.set_title(f'{model_alias} - SUBJECT', fontsize=16)
                    if focus == 'object': ax.set_title(f'{model_alias} - OBJECT', fontsize=16)

        if do_plot_legend:
            plt.legend(loc='upper left', bbox_to_anchor=(-1.98, -0.32), fontsize='x-large', ncol=2, fancybox=True, shadow=True)
        
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
            plt.plot(range(self.n_layers), dist_word, c=colors[widx], linestyle='solid', label=f'{word} [act]')

        for widx, word in enumerate(self.cls2_words[dataset][random_sidx]):
            vec_word = self.cls2_instances[dataset][random_sidx][widx,:,:]
            dist_word = self.layerwise_eucdist(diff_corr1_corr2, vec_word)
            plt.plot(range(self.n_layers), dist_word, c=colors[widx], linestyle='dotted', label=f'{word} [pass]')

        plt.title(f'{dataset} dataset: "{ref_word}[act] - {ref_word}[pass]"')
        plt.legend(loc='right', bbox_to_anchor=(1.05, 0), fontsize='x-small', ncol=10, fancybox=True, shadow=True)
        plt.show()
            


    def layerwise_eucdist(self, vec1, vec2):
        return np.array([np.linalg.norm(vec1[layer,:] - vec2[layer,:]) for layer in range(self.n_layers)]).reshape(self.n_layers,1)

    def layerwise_cosine_sim(self, vec1, vec2):
        return np.array([cosine_similarity(vec1[layer,:].reshape(1,-1), vec2[layer,:].reshape(1,-1)) for layer in range(self.n_layers)]).reshape(self.n_layers,1)
            
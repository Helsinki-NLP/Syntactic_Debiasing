import sys
import os
import h5py
import pickle
import conllu
import numpy as np
import torch
import logging
from utils.bertify import bertify


MT_ENC_DIM = 512
MT_LAYERS = 6


# ----- h5 file functions -----
def loadh5file(load_path):
    '''load embeddings and convert to list of tensors'''
    logging.debug(f'   loading embeddings from {load_path}')
    h5f = h5py.File(load_path, 'r')
    setlen = len(h5f)
    loaded_reprs = [torch.FloatTensor(h5f.get(str(i))[()]) for i in range(setlen)]
    h5f.close()
    return loaded_reprs


def saveh5file(representations, save_path):
    '''save embeddings in h5 format'''
    logging.debug(f'     saving embeddings to {save_path}')
    os.system(f'mkdir -p {os.path.dirname(save_path)}')
    with h5py.File(save_path, 'w') as fout:
        for idx,rps in enumerate(representations):
            fout.create_dataset(str(idx), rps.shape, dtype='float32', data=rps)


# ----- pickle file functions -----
def loadpickle(load_path):
    '''load words from pickle file'''
    logging.debug(f'   loading words from {load_path}')
    with open(load_path, 'rb') as fin:
        words = pickle.load(fin)
    return words


def savepickle(words, save_path):
    '''save words in pickle format'''
    logging.debug(f'     saving words to {save_path}')
    os.system(f'mkdir -p {os.path.dirname(save_path)}')
    with open(save_path, 'wb') as fout:
        pickle.dump(words, fout)



# ----- conllu parsing functions -----

def connluify(lines):
    for i in range(len(lines)):
        if lines[i] == '\n': continue
        tokens = lines[i].strip().split('\t')
        tokens.insert(2, tokens[1])
        lines[i] = '\t'.join(tokens)+'\n'
    return lines


# ----- auxilaries -----

def bert_extract(dataset, focus, data_path, cls1_name, cls2_name, to_device='cpu'):
    device = torch.device(to_device) 

    # These will be lists of np arrays of shape (seq_len x n_layers x enc_dim), since every sentence can be of arbitrary length now
    cls1_instances = []
    cls2_instances = []

    # These will be list of lists of tokens, one list of tokens per sentence.
    cls1_words = []
    cls2_words = []   

    bert = bertify(device)

    cls1_file = open(data_path + '/' + f'{dataset}.{cls1_name}.pos.parse.conll', 'r', encoding="utf-8")
    firstline = cls1_file.readline()
    cls1_lines = connluify(cls1_file.readlines())
    cls1_items = conllu.parse(''.join(cls1_lines))

    cls2_file = open(data_path + '/' + f'{dataset}.{cls2_name}.pos.parse.conll', 'r', encoding="utf-8")
    firstline = cls2_file.readline()
    cls2_lines = connluify(cls2_file.readlines())
    cls2_items = conllu.parse(''.join(cls2_lines))

    for cls1_tokens, cls2_tokens in zip(cls1_items, cls2_items):
        cls1_sentence = ' '.join([token['form'] for token in cls1_tokens])
        cls2_sentence = ' '.join([token['form'] for token in cls2_tokens])

        #compute embeddings from Bert
        cls1_bert_tokens, cls1_bert_tokenization = bert.tokenize(cls1_sentence)
        cls1_bert_enc = bert.encode(cls1_bert_tokens)

        cls2_bert_tokens, cls2_bert_tokenization = bert.tokenize(cls2_sentence)
        cls2_bert_enc = bert.encode(cls2_bert_tokens)

        cls1_bert_enc = bert.correct_bert_tokenization(cls1_bert_enc, cls1_bert_tokenization)
        cls2_bert_enc = bert.correct_bert_tokenization(cls2_bert_enc, cls2_bert_tokenization)

        #roles:
        #odict_keys(['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel'])
        #print(cls1_tokens[0]['id'])       1
        #print(cls1_tokens[0]['form'])     the
        #print(cls1_tokens[0]['lemma'])    the
        #print(cls1_tokens[0]['upostag'])  _
        #print(cls1_tokens[0]['xpostag'])  DT
        #print(cls1_tokens[0]['feats'])    None
        #print(cls1_tokens[0]['head'])     3
        #print(cls1_tokens[0]['deprel'])   det

        logging.debug(cls1_sentence)
        logging.debug(cls2_sentence)
        #logging.debug(cls2_bert_tokenization)


        # ----- Active-Passive Task -----
        if cls1_name == 'active' and cls2_name == 'passive':
            active_tokens = cls1_tokens
            passive_tokens = cls2_tokens

            active_bert_enc = cls1_bert_enc
            passive_bert_enc = cls2_bert_enc

            if focus != 'all':
                if focus == 'verb':
                    for passive_token in passive_tokens:
                        if passive_token['deprel'].upper() == 'ROOT':
                            WOI_passive_id = passive_token['id']
                            WOI_passive_form = passive_token['form']
                            break
                    for active_token in active_tokens:
                        if active_token['deprel'].upper() == 'ROOT':
                            WOI_active_id = active_token['id']
                            WOI_active_form = active_token['form']
                            break

                if focus == 'subject':
                    WOI_passive_id = -1
                    for passive_token in passive_tokens:
                        if passive_token['form'] == 'by':
                            WOI_passive_id = passive_token['head']
                            WOI_passive_form = passive_token['form']
                            break
                    for passive_token in passive_tokens:
                        if passive_token['id'] == WOI_passive_id:
                            WOI_form = passive_token['form'].lower()
                            WOI_passive_form = passive_token['form'].lower()
                            break
                    if WOI_passive_id == -1:
                        continue
                    for active_token in active_tokens:
                        if active_token['form'].lower() == WOI_form:
                            WOI_active_id = active_token['id']
                            WOI_active_form = active_token['form'].lower()
                            break

                if focus == 'object':
                    WOI_passive_id = -1
                    for passive_token in passive_tokens:
                        if passive_token['deprel'] == 'nsubj:pass' or passive_token['deprel'] == 'nsubjpass':
                            WOI_passive_id = passive_token['id']
                            WOI_passive_form = passive_token['form'].lower()
                            break
                    if WOI_passive_id == -1:
                        continue
                    WOI_active_id = -1
                    for active_token in active_tokens:
                        if active_token['form'].lower() == WOI_passive_form and (active_token['deprel'] == 'dobj' or active_token['deprel'] == 'obj'):
                            WOI_active_id = active_token['id']
                            WOI_active_form = active_token['form'].lower()
                            break
                    if WOI_active_id == -1:
                        continue
                

                instance_1 = np.stack([np.reshape(active_bert_enc[layer][:,WOI_active_id-1,:],(1,bert.ENC_DIM)).detach().cpu().numpy() \
                                                                                for layer in range(bert.N_LAYERS)], \
                                                                                axis=1)
                instance_2 = np.stack([np.reshape(passive_bert_enc[layer][:,WOI_passive_id-1,:],(1,bert.ENC_DIM)).detach().cpu().numpy() \
                                                                                for layer in range(bert.N_LAYERS)], \
                                                                                axis=1)

                words_1 = [WOI_active_form]
                words_2 = [WOI_passive_form]

                logging.debug('Active: ' + WOI_active_form)
                logging.debug('Passive: ' + WOI_passive_form)


            if focus == 'all':
                cls1_wc = active_bert_enc[0].shape[1]
                instance_1 = np.stack([np.reshape(active_bert_enc[layer][:,:,:],(cls1_wc,bert.ENC_DIM)).detach().cpu().numpy() \
                                                                                for layer in range(bert.N_LAYERS)], \
                                                                                axis=1)

                cls2_wc = passive_bert_enc[0].shape[1]
                instance_2 = np.stack([np.reshape(passive_bert_enc[layer][:,:,:],(cls2_wc,bert.ENC_DIM)).detach().cpu().numpy() \
                                                                                for layer in range(bert.N_LAYERS)], \
                                                                                axis=1)
                
                words_1 = [cls1_token['form'] for cls1_token in cls1_tokens]
                words_2 = [cls2_token['form'] for cls2_token in cls2_tokens]



            cls1_instances.append(instance_1)
            cls2_instances.append(instance_2)
            
            cls1_words.append(words_1)
            cls2_words.append(words_2)


    if dataset == 'RNN' and clauses_only:
        pass

    return cls1_instances, cls2_instances, cls1_words, cls2_words






def mt_extract(dataset, focus, mt_reprs_path, data_path, cls1_name, cls2_name, to_device='cpu'):
    device = torch.device(to_device) 

    # These will be lists of np arrays of shape (seq_len x n_layers x enc_dim), since every sentence can be of arbitrary length now
    cls1_instances = []
    cls2_instances = []

    # These will be list of lists of tokens, one list of tokens per sentence.
    cls1_words = []
    cls2_words = []   

    cls1_file = open(data_path + '/' + f'{dataset}.{cls1_name}.pos.parse.conll', 'r', encoding="utf-8")
    firstline = cls1_file.readline()
    cls1_lines = connluify(cls1_file.readlines())
    cls1_items = conllu.parse(''.join(cls1_lines))

    cls2_file = open(data_path + '/' + f'{dataset}.{cls2_name}.pos.parse.conll', 'r', encoding="utf-8")
    firstline = cls2_file.readline()
    cls2_lines = connluify(cls2_file.readlines())
    cls2_items = conllu.parse(''.join(cls2_lines))

    mt_reprs_cls1 = h5py.File(f'{mt_reprs_path}/{dataset}/{dataset}_{cls1_name}.repr.layer_word.hdf5', 'r')
    mt_reprs_cls2 = h5py.File(f'{mt_reprs_path}/{dataset}/{dataset}_{cls2_name}.repr.layer_word.hdf5', 'r')


    for sidx, (cls1_tokens, cls2_tokens) in enumerate(zip(cls1_items, cls2_items)):
        cls1_sentence = ' '.join([token['form'] for token in cls1_tokens])
        cls2_sentence = ' '.join([token['form'] for token in cls2_tokens])

        cls1_representation = mt_reprs_cls1[str(sidx)]
        cls2_representation = mt_reprs_cls2[str(sidx)]

        #roles:
        #odict_keys(['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel'])
        #print(cls1_tokens[0]['id'])       1
        #print(cls1_tokens[0]['form'])     the
        #print(cls1_tokens[0]['lemma'])    the
        #print(cls1_tokens[0]['upostag'])  _
        #print(cls1_tokens[0]['xpostag'])  DT
        #print(cls1_tokens[0]['feats'])    None
        #print(cls1_tokens[0]['head'])     3
        #print(cls1_tokens[0]['deprel'])   det

        logging.debug(cls1_sentence)
        logging.debug(cls2_sentence)
        #logging.debug(cls2_bert_tokenization)


        # ----- Active-Passive Task -----
        if cls1_name == 'active' and cls2_name == 'passive':
            active_tokens = cls1_tokens
            passive_tokens = cls2_tokens

            active_mt_enc = cls1_representation
            passive_mt_enc = cls2_representation

            if focus != 'all':
                if focus == 'verb':
                    for passive_token in passive_tokens:
                        if passive_token['deprel'].upper() == 'ROOT':
                            WOI_passive_id = passive_token['id']
                            WOI_passive_form = passive_token['form']
                            break
                    for active_token in active_tokens:
                        if active_token['deprel'].upper() == 'ROOT':
                            WOI_active_id = active_token['id']
                            WOI_active_form = active_token['form']
                            break

                if focus == 'subject':
                    WOI_passive_id = -1
                    for passive_token in passive_tokens:
                        if passive_token['form'] == 'by':
                            WOI_passive_id = passive_token['head']
                            WOI_passive_form = passive_token['form']
                            break
                    for passive_token in passive_tokens:
                        if passive_token['id'] == WOI_passive_id:
                            WOI_form = passive_token['form'].lower()
                            WOI_passive_form = passive_token['form'].lower()
                            break
                    if WOI_passive_id == -1:
                        continue
                    for active_token in active_tokens:
                        if active_token['form'].lower() == WOI_form:
                            WOI_active_id = active_token['id']
                            WOI_active_form = active_token['form'].lower()
                            break

                if focus == 'object':
                    WOI_passive_id = -1
                    for passive_token in passive_tokens:
                        if passive_token['deprel'] == 'nsubj:pass' or passive_token['deprel'] == 'nsubjpass':
                            WOI_passive_id = passive_token['id']
                            WOI_passive_form = passive_token['form'].lower()
                            break
                    if WOI_passive_id == -1:
                        continue
                    WOI_active_id = -1
                    for active_token in active_tokens:
                        if active_token['form'].lower() == WOI_passive_form and (active_token['deprel'] == 'dobj' or active_token['deprel'] == 'obj'):
                            WOI_active_id = active_token['id']
                            WOI_active_form = active_token['form'].lower()
                            break
                    if WOI_active_id == -1:
                        continue
                

                instance_1 = np.stack([np.reshape(active_mt_enc[layer,WOI_active_id-1,:],(1,MT_ENC_DIM)) \
                                                                                for layer in range(MT_LAYERS)], \
                                                                                axis=1)
                instance_2 = np.stack([np.reshape(passive_mt_enc[layer,WOI_passive_id-1,:],(1,MT_ENC_DIM)) \
                                                                                for layer in range(MT_LAYERS)], \
                                                                                axis=1)

                words_1 = [WOI_active_form]
                words_2 = [WOI_passive_form]

                logging.debug('Active: ' + WOI_active_form)
                logging.debug('Passive: ' + WOI_passive_form)


            if focus == 'all':
                cls1_wc = active_mt_enc.shape[1]
                instance_1 = np.stack([np.reshape(active_mt_enc[layer,:,:],(cls1_wc,MT_ENC_DIM)) \
                                                                                for layer in range(MT_LAYERS)], \
                                                                                axis=1)

                cls2_wc = passive_mt_enc.shape[1]
                instance_2 = np.stack([np.reshape(passive_mt_enc[layer,:,:],(cls2_wc,MT_ENC_DIM)) \
                                                                                for layer in range(MT_LAYERS)], \
                                                                                axis=1)
                
                words_1 = [cls1_token['form'] for cls1_token in cls1_tokens]
                words_2 = [cls2_token['form'] for cls2_token in cls2_tokens]



            cls1_instances.append(instance_1)
            cls2_instances.append(instance_2)
            
            cls1_words.append(words_1)
            cls2_words.append(words_2)


    if dataset == 'RNN' and clauses_only:
        pass

    return cls1_instances, cls2_instances, cls1_words, cls2_words



# ----- API for this module -----

def load_representations(opt):
    cls1_name, cls2_name = opt.task.split('-')

    cls1_instances = {dataset: {} for dataset in opt.dataset}
    cls2_instances = {dataset: {} for dataset in opt.dataset}

    cls1_words = {dataset: {} for dataset in opt.dataset}
    cls2_words = {dataset: {} for dataset in opt.dataset}

    for dataset in opt.dataset:
        for focus in opt.focus:
            logging.info('Loading representations from dataset: ' + dataset + ', focus: ' + focus + ' at ' + opt.load_reprs_path)

            # These will be lists of np arrays of shape (seq_len x n_layers x enc_dim), 
            # since every sentence can be of arbitrary length now
            cls1_instances[dataset][focus] = loadh5file(opt.load_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls1_name}.{focus}.h5')
            cls2_instances[dataset][focus] = loadh5file(opt.load_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls2_name}.{focus}.h5')

            cls1_words[dataset][focus] = loadpickle(opt.load_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls1_name}.{focus}.words.pkl')
            cls2_words[dataset][focus] = loadpickle(opt.load_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls2_name}.{focus}.words.pkl')

    return cls1_instances, cls2_instances, cls1_words, cls2_words


def extract_representations(opt):
    cls1_name, cls2_name = opt.task.split('-')

    cls1_instances = {dataset: {} for dataset in opt.dataset}
    cls2_instances = {dataset: {} for dataset in opt.dataset}

    cls1_words = {dataset: {} for dataset in opt.dataset}
    cls2_words = {dataset: {} for dataset in opt.dataset} 

    for dataset, dataset_path in zip(opt.dataset, opt.dataset_path):
        for focus in opt.focus:
            logging.info('Extracting representations from ' + dataset + ' at ' + dataset_path)
            
            if opt.model == 'bert':
                cls1_instances[dataset][focus], cls2_instances[dataset][focus], cls1_words[dataset][focus], cls2_words[dataset][focus] \
                                = bert_extract(dataset, focus, dataset_path, cls1_name, cls2_name,
                                          to_device=('cuda' if opt.cuda else 'cpu'))
            elif opt.model == 'MT':
                cls1_instances[dataset][focus], cls2_instances[dataset][focus], cls1_words[dataset][focus], cls2_words[dataset][focus] \
                                = mt_extract(dataset, focus, opt.mt_reprs_path, dataset_path, cls1_name, cls2_name,
                                          to_device=('cuda' if opt.cuda else 'cpu'))                             

            logging.info('Saving representations to dataset: ' + dataset + ', focus: ' + focus + ' at ' + opt.save_reprs_path)            
            saveh5file(cls1_instances[dataset][focus], opt.save_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls1_name}.{focus}.h5')
            saveh5file(cls2_instances[dataset][focus], opt.save_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls2_name}.{focus}.h5')

            savepickle(cls1_words[dataset][focus], opt.save_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls1_name}.{focus}.words.pkl')
            savepickle(cls2_words[dataset][focus], opt.save_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls2_name}.{focus}.words.pkl')


    return cls1_instances, cls2_instances, cls1_words, cls2_words


    
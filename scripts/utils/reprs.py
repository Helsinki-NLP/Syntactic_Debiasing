import sys
import h5py
import conllu
import numpy as np
from utils.bertify import bertify
from utils.logger import logger


# ----- h5 file functions -----
def loadh5file(load_path):
    '''load embeddings and convert to list of tensors'''
    logger.info('   loading from {0}'.format(load_path))
    h5f = h5py.File(load_path, 'r')
    setlen = len(h5f)
    loaded_reprs = [torch.FloatTensor(h5f.get(str(i))[()]) for i in range(setlen)]
    h5f.close()
    return loaded_reprs


def saveh5file(representations, save_path):
    '''save embeddings in h5 format'''
    logger.info(f'     saving embeddings to {save_path}')
    os.system(f'mkdir -p {os.path.dirname(save_path)}')
    with h5py.File(outfile, 'w') as fout:
        for idx,rps in enumerate(representations):
            fout.create_dataset(str(idx), rps.shape, dtype='float32', data=rps)


# ----- conllu parsing functions -----

def connluify(lines):
    for i in range(len(lines)):
        if lines[i] == '\n': continue
        tokens = lines[i].strip().split('\t')
        tokens.insert(2, tokens[1])
        lines[i] = '\t'.join(tokens)+'\n'
    return lines


def extract(dataset, data_path, cls1_name, cls2_name, focus, clauses_only, device):
    # This will be a list of np arrays of shape (seq_len x n_layers x enc_dim), since every sentence can be of arbitrary length now
    cls1_instances = []
    cls2_instances = []

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

        print(cls1_sentence)
        print(cls1_bert_tokenization)

        # ----- Active-Passive Task -----
        if cls1_name == 'active' and cls2_name == 'passive':
            active_tokens = cls1_tokens
            passive_tokens = cls2_tokens

            active_bert_enc = cls1_bert_enc
            passive_bert_enc = cls2_bert_enc

            if focus != 'all':
                if focus == 'verb':
                    for passive_token in passive_tokens:
                        if passive_token['deprel'] == 'ROOT':
                            WOI_active_id = passive_token['id']
                            break
                    for active_token in active_tokens:
                        if active_token['deprel'] == 'ROOT':
                            WOI_passive_id = active_token['id']
                            WOI_form = active_token['form']
                            break

                if focus == 'subject':
                    for passive_token in passive_tokens:
                        if passive_token['form'] == 'by':
                            WOI_passive_id = passive_token['head']
                            break
                    for passive_token in passive_tokens:
                        if passive_token['id'] == WOI_passive_id:
                            WOI_form = passive_token['form']
                            break
                    for active_token in active_tokens:
                        if active_token['form'] == WOI_form:
                            WOI_active_id = active_token['id']
                            break

                if focus == 'object':
                    for passive_token in passive_tokens:
                        if passive_token['deprel'] == 'nsubjpass':
                            WOI_passive_id = passive_token['id']
                            WOI_form = passive_token['form']
                            break
                    for active_token in active_tokens:
                        if active_token['form'] == WOI_form and active_token['deprel'] == 'dobj':
                            WOI_active_id = active_token['id']
                            break

                instance_1 = np.stack([np.reshape(active_bert_enc[layer][:,WOI_active_id-1,:],(1,bert.ENC_DIM)).detach().cpu().numpy() \
                                                                                for layer in range(bert.N_LAYERS)], \
                                                                                axis=1)
                instance_2 = np.stack([np.reshape(passive_bert_enc[layer][:,WOI_passive_id-1,:],(1,bert.ENC_DIM)).detach().cpu().numpy() \
                                                                                for layer in range(bert.N_LAYERS)], \
                                                                                axis=1)

                logger.info('Active: ' + active_token['form'])
                logger.info('Passive: ' + passive_token['form'])


            if focus == 'all':
                pass


            cls1_instances.append(instance_1)
            cls2_instances.append(instance_2)


    if dataset == 'RNN' and clauses_only:
        pass

    return cls1_instances, cls2_instances

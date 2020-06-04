"""
Run the syntactic debiasing
Code structure inspired from: Juan Raul Vazquez Carillo
"""

import sys
import random
import argparse
from utils.logger import logger
from utils import reprs
from utils import arrange_data
import debiasing


def main(opt):
    
    #----- Loading data -----

    cls1_name, cls2_name = opt.task.split('-')

    cls1_instances = {}
    cls2_instances = {}

    cls1_words = {}
    cls2_words = {}

    if opt.load_reprs_path:
        for dataset in opt.dataset:
            logger.info('Loading representations from ' + dataset + ' at ' + opt.load_reprs_path)

            # These will be lists of np arrays of shape (seq_len x n_layers x enc_dim), 
            # since every sentence can be of arbitrary length now
            cls1_instances[dataset] = reprs.loadh5file(opt.load_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls1_name}.{opt.focus}.h5')
            cls2_instances[dataset] = reprs.loadh5file(opt.load_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls2_name}.{opt.focus}.h5')

            cls1_words[dataset] = reprs.loadpickle(opt.load_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls1_name}.{opt.focus}.words.pkl')
            cls2_words[dataset] = reprs.loadpickle(opt.load_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls2_name}.{opt.focus}.words.pkl')
    else:
        for dataset, dataset_path in zip(opt.dataset, opt.dataset_path):
            logger.info('Extracting representations from ' + dataset + ' at ' + dataset_path)
            cls1_instances[dataset], cls2_instances[dataset], cls1_words[dataset], cls2_words[dataset] \
                                = reprs.extract(dataset, dataset_path, cls1_name, cls2_name,
                                                opt.focus, opt.clauses_only, to_device=('cuda' if opt.cuda else 'cpu'))

            logger.info('Saving representations to ' + dataset + ' at ' + opt.save_reprs_path)            
            reprs.saveh5file(cls1_instances[dataset], opt.save_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls1_name}.{opt.focus}.h5')
            reprs.saveh5file(cls2_instances[dataset], opt.save_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls2_name}.{opt.focus}.h5')

            reprs.savepickle(cls1_words[dataset], opt.save_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls1_name}.{opt.focus}.words.pkl')
            reprs.savepickle(cls2_words[dataset], opt.save_reprs_path + '/' + f'{dataset}/{opt.task}/{dataset}.{cls2_name}.{opt.focus}.words.pkl')

        if opt.extract_only:
            logger.info('Finishing...')
            sys.exit(0)


    #----- Train-Test Splits -----

    logger.info('Separating training and test sets')
    X_train = {}
    X_test = {}

    Y_train = {}
    Y_test = {}

    #FIXME: This assumes single lexical item from each example sentence for now, eg. a verb.
    if opt.cross_dataset_lexical_split and opt.train_on and opt.test_on:
        # keep only the intersection of lexical items in both datasets
        shared_vocab = set([item[0] for item in cls2_words[opt.train_on]]).intersection( \
                         set([item[0] for item in cls2_words[opt.test_on]])) # use the passive items, their forms are similar

        new_cls1_instances = {dataset: [] for dataset in opt.dataset}
        new_cls2_instances = {dataset: [] for dataset in opt.dataset}

        new_cls1_words = {dataset: [] for dataset in opt.dataset}
        new_cls2_words = {dataset: [] for dataset in opt.dataset}

        for dataset in opt.dataset:
            for i in range(len(cls1_instances[dataset])):
                passive_word = cls2_words[dataset][i][0]
                if passive_word in shared_vocab:
                    new_cls1_instances[dataset].append(cls1_instances[dataset][i])
                    new_cls2_instances[dataset].append(cls2_instances[dataset][i])
                    new_cls1_words[dataset].append(cls1_words[dataset][i])
                    new_cls2_words[dataset].append(cls2_words[dataset][i])

        cls1_instances = new_cls1_instances
        cls2_instances = new_cls2_instances

        cls1_words = new_cls1_words
        cls2_words = new_cls2_words


    for dataset in opt.dataset:
        X_train[dataset], Y_train[dataset], X_test[dataset], Y_test[dataset] = arrange_data.train_test_split(cls1_instances[dataset], cls2_instances[dataset],
                                                                             cls1_words[dataset], cls2_words[dataset],
                                                                             opt.layer, 
                                                                             opt.lexical_split)

        print('\n\nClass 1 words in ' + dataset + '\n')
        print(cls1_words[dataset])
        print('\n\nClass 2 words in ' + dataset + '\n\n')
        print(cls2_words[dataset])


    #----- Training -----
    
    db = debiasing.Debiasing(classifier='LinearSVC', n_iterations=30)

    # if specified, train/test with the train_on/test_on dataset
    # else, use the default dataset for train and test
    if not (opt.train_on and opt.test_on):
        P = db.train(X_train[dataset], Y_train[dataset], X_test[dataset], Y_test[dataset])

    elif opt.transfer_classifier:
        # if set, we try the test dataset on the same classifier that is trained on the train dataset
        P = db.train(X_train[opt.train_on], Y_train[opt.train_on], X_test[opt.test_on], Y_test[opt.test_on])
    
    else:
        P = db.train(X_train[opt.train_on], Y_train[opt.train_on], X_test[opt.train_on], Y_test[opt.train_on])

        # "clean" the test dataset:
        X_train_cleaned = {opt.test_on: db.clean_data(X_train[opt.test_on], P)}
        X_test_cleaned = {opt.test_on: db.clean_data(X_test[opt.test_on], P)}

        # try to re-debias the "cleaned" dataset:
        P_second = db.train(X_train_cleaned[opt.test_on], Y_train[opt.test_on], X_test_cleaned[opt.test_on], Y_test[opt.test_on])



    #----- Logging results -----

    output_dir = f'{opt.outdir}/{opt.dataset}/{opt.task}'


    #if opt.plot_results:
    #    Plt.makeplot()
    #else:
    #    logger('finishing ... to make a plot call: ')
    #    logger(' you can also use option --plot_results')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, default=['SICK'], nargs='+',
                        help='dataset to use [SICK (default) | SICK_tensecorr (manually tense-corrected) | RNN]')

    parser.add_argument('--cuda', action='store_true',
                        help='whether to use a cuda device')

    parser.add_argument('--seed', type=int,
                        help='random seed to fix (optional)')
    
    parser.add_argument('--extract_only', action='store_true',
                        help='run only until the extraction and saving of representations')

    parser.add_argument('--dataset_path', required=False, type=str, nargs='+',
                        default=['/scratch/project_2002233/debiasing/data/SICK/Filtered'],
                        help='path to the raw dataset location. Defaults to puhti:SICK location')

    parser.add_argument('--debug_mode', action='store_true',
                        help='launch ipdb debugger if script crashes.')
    
    parser.add_argument('--load_reprs_path', required=False, type=str,
                        help='previously extracted representations\' path')

    parser.add_argument('--save_reprs_path', required=False, type=str,
                        help='save the extracted representations to')

    parser.add_argument('--outdir', type=str, required=False, default='../outputs/',
                        help='path to dir where output logs will be saved')

    parser.add_argument('--plot_results', action='store_true',
                        help='if active, will plot visualizations of the data')

    parser.add_argument('--task', required=True, default='active-passive',
                        help='which syntactic information to debias, default active-passive')    

    parser.add_argument('--focus', required=True, default='verb',
                        help='which part of the sentence to focus on [verb [default] | subject | object | all')   

    parser.add_argument('--layer', required=False, type=int,
                        help='which layer of representations to debias on\' path')   

    parser.add_argument('--clauses_only', action='store_true',
                        help='use only the subject clause from the RNN dataset')

    parser.add_argument('--lexical_split', action='store_true',
                        help='enforces that same lexical entities go to the same train/test set split')

    parser.add_argument('--cross_dataset_lexical_split', action='store_true',
                        help='enforces that same lexical entities go to the same train/test set split')

    parser.add_argument('--train_on', required=False,
                        help='train on the given dataset [SICK | RNN]')

    parser.add_argument('--test_on', required=False,
                        help='test on the given dataset [SICK | RNN]')   

    parser.add_argument('--transfer_classifier', action='store_true',
                        help='if set, we try the test dataset on the same classifier that is trained on the train dataset')   

    opt = parser.parse_args()

    if opt.seed:
        random.seed(opt.seed)
    
    if opt.debug_mode:
        with ipdb.launch_ipdb_on_exception():
            main(opt)
    else:
        main(opt)

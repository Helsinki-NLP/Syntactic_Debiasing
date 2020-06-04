"""
Run the syntactic debiasing
Code structure inspired from: Juan Raul Vazquez Carillo
"""

import sys
import random
import argparse
import logging
import ipdb
from utils import reprs
from utils import arrange_data
import debiasing


def main(opt):
    
    #----- Loading data -----

    if opt.load_reprs_path:
        logging.info('Loading representations')
        cls1_instances, cls2_instances, cls1_words, cls2_words = reprs.load_representations(opt)        
    else:
        logging.info('Extracting representations')
        cls1_instances, cls2_instances, cls1_words, cls2_words = reprs.extract_representations(opt)

        if opt.extract_only:
            logging.info('Finishing...')
            sys.exit(0)


    #----- Train-Test Splits -----

    #FIXME: This assumes single lexical item from each example sentence for now, eg. a verb.
    if opt.use_shared_vocab and opt.train_on and opt.test_on:
        logging.info('Restricting to shared vocabulary between the datasets')
        cls1_instances, cls2_instances, cls1_words, cls2_words = arrange_data.restrict_vocab(cls1_instances, cls2_instances, cls1_words, cls2_words)

    logging.info('Separating training and test sets')
    X_train, Y_train, X_test, Y_test = arrange_data.train_test_split(cls1_instances, cls2_instances,
                                                                         cls1_words, cls2_words,
                                                                         opt.dataset,
                                                                         opt.layer, 
                                                                         opt.lexical_split)
    #for dataset in opt.dataset:
    #    X_train[dataset], Y_train[dataset], X_test[dataset], Y_test[dataset] = arrange_data.train_test_split(cls1_instances[dataset], cls2_instances[dataset],
    #                                                                         cls1_words[dataset], cls2_words[dataset],
    #                                                                         opt.layer, 
    #                                                                         opt.lexical_split)

    #----- Training -----
    
    db = debiasing.Debiasing(classifier='LinearSVC', n_iterations=30)

    # if specified, train/test with the train_on/test_on dataset
    # else, use the default dataset for train and test
    if not (opt.train_on and opt.test_on):
        dataset = opt.dataset[0]
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

    parser.add_argument('--debug', action='store_true',
                        help='debug-level logging, launch ipdb debugger if script crashes.')
    
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

    parser.add_argument('--use_shared_vocab', action='store_true',
                        help='restricts the different datasets to use only the instances with shared vocabulary')

    parser.add_argument('--train_on', required=False,
                        help='train on the given dataset [SICK | RNN]')

    parser.add_argument('--test_on', required=False,
                        help='test on the given dataset [SICK | RNN]')   

    parser.add_argument('--transfer_classifier', action='store_true',
                        help='if set, we try the test dataset on the same classifier that is trained on the train dataset')   

    opt = parser.parse_args()

    if opt.seed:
        random.seed(opt.seed)
    
    if opt.debug:
        logging.basicConfig(level=logging.DEBUG)
        with ipdb.launch_ipdb_on_exception():
            main(opt)
    else:
        main(opt)

"""
Run the syntactic debiasing
Code structure inspired from: Juan Raul Vazquez Carillo
"""

import sys
import random
import torch #FIXME
from utils import opts
from utils.logger import logger
from utils import reprs
from utils import arrange_data
import debiasing


def main(opt):
    device = torch.device('cuda' if opt.cuda else 'cpu') 

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
                                                opt.focus, opt.clauses_only, device)

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

    if opt.cross_dataset_lexical_split:
        # keep only the intersection of lexical items in both datasets
        pass

    for dataset in opt.dataset:
        X_train[dataset], Y_train[dataset], X_test[dataset], Y_test[dataset] = arrange_data.train_test_split(cls1_instances[dataset], cls2_instances[dataset],
                                                                             cls1_words[dataset], cls2_words[dataset],
                                                                             opt.layer, 
                                                                             opt.lexical_split)


    #----- Training -----
    
    # if specified, train/test with the train_on/test_on dataset
    # else, use the default dataset for train and test
    if opt.train_on:
        train_dataset = opt.train_on
    else:
        train_dataset = dataset

    if opt.test_on:
        test_dataset = opt.test_on
    else:
        test_dataset = dataset

    db = debiasing.Debiasing(classifier='LinearSVC', n_iterations=40)
    P = db.train(X_train[train_dataset], Y_train[train_dataset], X_test[test_dataset], Y_test[test_dataset])


    #----- Logging results -----

    output_dir = f'{opt.outdir}/{opt.dataset}/{opt.task}'


    #if opt.plot_results:
    #    Plt.makeplot()
    #else:
    #    logger('finishing ... to make a plot call: ')
    #    logger(' you can also use option --plot_results')


if __name__ == '__main__':
    parser = opts.get_parser()
    opt = parser.parse_args()
    if opt.seed:
        random.seed(opt.seed)
    if opt.debug_mode:
        with ipdb.launch_ipdb_on_exception():
            main(opt)
    else:
        main(opt)

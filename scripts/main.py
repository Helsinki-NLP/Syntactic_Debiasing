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
import vectorize
from utils import plotting

def main(opt):
    
    #----- Loading data -----

    if opt.load_reprs_path:
        logging.info('Loading representations')
        cls1_instances, cls2_instances, cls1_words, cls2_words, cls1_ids, cls2_ids = reprs.load_representations(opt)        
    else:
        logging.info('Extracting representations')
        cls1_instances, cls2_instances, cls1_words, cls2_words, cls1_ids, cls2_ids = reprs.extract_representations(opt)

        if opt.extract_only:
            logging.info('Finishing...')
            sys.exit(0)


    #----- Train-Test Splits -----
    #if opt.use_shared_vocab and opt.train_on and opt.test_on:
    #    logging.info('Restricting to shared vocabulary between the datasets')
    #    cls1_instances, cls2_instances, cls1_words, cls2_words, '''cls1_ids, cls2_ids''' = arrange_data.restrict_vocab(cls1_instances, cls2_instances, cls1_words, cls2_words, opt)

    if opt.save_splits:
        logging.info('Separating training and test sets')
        #FIXME: This assumes single lexical item from each example sentence for now, eg. a verb.

        for experiment_number in range(opt.different_split_exp_count):
            X_train, Y_train, X_test, Y_test, \
                cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances \
                    = arrange_data.train_test_split(cls1_instances, cls2_instances,
                                                 cls1_words, cls2_words,
                                                 opt.dataset,
                                                 opt.layer-1, 
                                                 opt.lexical_split,
                                                 opt.random_labels,
                                                 opt.focus,
                                                 experiment_number)
        sys.exit(0)


    #for N_ITERATIONS in [2, 5, 10]:
    #    for REG_COEFF in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    for N_ITERATIONS in [3]:
        for REG_COEFF in [0.00001]:  #0.00000001
            logfile = open('../results/classification_acc/' + opt.train_on + f'/reg_coeff_{REG_COEFF}_it_{N_ITERATIONS}_{opt.focus}_layer-{opt.layer}.txt', 'w')

            for experiment_number in range(opt.different_split_exp_count):
                
                X_train, Y_train, X_test, Y_test, \
                    cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances \
                        = arrange_data.load_splits(opt.dataset, opt.focus, opt.layer-1, experiment_number)

                #for dataset in opt.dataset:
                #    X_train[dataset], Y_train[dataset], X_test[dataset], Y_test[dataset] = arrange_data.train_test_split(cls1_instances[dataset], cls2_instances[dataset],
                #                                                                         cls1_words[dataset], cls2_words[dataset],
                #                                                                         opt.layer, 
                #                                                                         opt.lexical_split)

                #----- Debiasing -----
                if not (opt.train_on and opt.test_on):
                    opt.train_on = opt.dataset[0]
                    opt.test_on = opt.dataset[0]    

                if opt.debias == 'Goldberg':
                    for j in range(opt.same_split_exp_count):
                        db = debiasing.Goldberg_Debiasing(classifier='LogisticRegression', n_iterations=N_ITERATIONS, reg_coeff=REG_COEFF)
                        db.debias(X_train, Y_train, X_test, Y_test, opt.train_on, opt.test_on, opt.transfer_projmatrix, opt.transfer_classifier, opt.plot, logfile)

                elif opt.debias == 'GBDD':
                    db = debiasing.GBDD_Debiasing(cls1_train_instances[opt.train_on], cls2_train_instances[opt.train_on], 
                                                  cls1_test_instances[opt.test_on], cls2_test_instances[opt.test_on])
                    db.calc_bias_dir_vec()
                    db.plot_before_after(cls1_test_instances[opt.test_on], 
                                         cls2_test_instances[opt.test_on])

                elif opt.debias == 'BAM':
                    db = debiasing.BAM_Debiasing()
                    db.debias(X_train, Y_train, X_test, Y_test)        
                
                #----- MDS plotting ------
                if opt.plot:
                    plotting.plot_mds(cls1_instances[opt.test_on], cls2_instances[opt.test_on])
                    if opt.debias:
                        plotting.plot_mds(cls1_instances[opt.test_on], cls2_instances[opt.test_on], with_debiasing=db)

                #----- Vector Explorations -----

                if opt.vector_fun:
                    if experiment_number == 0:
                        vc = vectorize.Vectorize()
                    
                    #vc.extract_diffvectors(opt.dataset, opt.layer, plotting_on=opt.plot_results)
                    
                    vc.set_data(cls1_instances, cls2_instances, cls1_words, cls2_words, cls1_ids, cls2_ids, opt.dataset)
                    vc.calc_word_senses('playing', opt.dataset[0], distance_fnc=vc.layerwise_eucdist, with_debiasing=db)

                    if experiment_number == opt.different_split_exp_count - 1:
                        vc.plot_word_senses(opt.focus, is_with_debiasing=True)

                    #vc.plot_distances_to_diff('pouring', 'SICK')
                    
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

    parser.add_argument('--dataset_path', required=False, type=str, nargs='+',
                        default=['/scratch/project_2002233/debiasing/data/SICK/Filtered'],
                        help='path to the raw dataset location. Defaults to puhti:SICK location')  

    parser.add_argument('--load_reprs_path', required=False, type=str,
                        help='previously extracted representations\' path')

    parser.add_argument('--save_reprs_path', required=False, type=str,
                        help='save the extracted representations to')

    parser.add_argument('--outdir', type=str, required=False, default='../outputs/',
                        help='path to dir where output logs will be saved')     

    parser.add_argument('--task', required=True, default='active-passive',
                        help='which syntactic information to debias, default active-passive')    

    parser.add_argument('--focus', required=True, default='verb',
                        help='which part of the sentence to focus on [verb [default] | subject | object | all')                                                
    
    parser.add_argument('--extract_only', action='store_true',
                        help='run only until the extraction and saving of representations')

    parser.add_argument('--debias', required=False, type=str,
                        help='debiasing method if it is to be performed [Goldberg (default) | GBDD | BAM]')

    parser.add_argument('--vector_fun', action='store_true',
                        help='whether to have fun with vectors')

    parser.add_argument('--plot_results', action='store_true',
                        help='if active, will plot visualizations of the data')

    parser.add_argument('--layer', required=False, type=int, default=6,
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

    parser.add_argument('--transfer_projmatrix', action='store_true',
                        help='if set, we learn the projection matrix on the train dataset, then use it to clean the test dataset')   

    parser.add_argument('--transfer_classifier', action='store_true',
                        help='if set, we try the test dataset on the same classifier that is trained on the train dataset')  

    parser.add_argument('--random_labels', action='store_true',
                        help='classify with random labels (baseline)')                                                

    parser.add_argument('--cuda', action='store_true',
                        help='whether to use a cuda device')

    parser.add_argument('--seed', type=int,
                        help='random seed to fix (optional)')  

    parser.add_argument('--debug', action='store_true',
                        help='debug-level logging, launch ipdb debugger if script crashes.')  

    parser.add_argument('--plot', action='store_true',
                        help='shall plot the MDS before and after cleaning.')                              

    parser.add_argument('--different_split_exp_count', type=int, default=1,
                        help='how many times to calculate a different train-test split and repeat the experiments.') 

    parser.add_argument('--same_split_exp_count', type=int, default=1,
                        help='how many times to repeat experiments with each train-test split.')   

    parser.add_argument('--save_splits', action='store_true',
                        help='create and save the splits to load later.')                           

    opt = parser.parse_args()

    if opt.seed:
        random.seed(opt.seed)
    

    # --- sanity checks for interactions between the parameters --- 
    if opt.transfer_projmatrix and opt.transfer_classifier:
        logging.error('--transfer_projmatrix and --transfer_classifier cannot be set at the same time')
        sys.exit(1)

    if opt.focus == 'all' and opt.debias == 'GBDD':
        logging.error('GBDD debiasing is usable with only single word focus for now')
        sys.exit(1)

    if opt.focus == 'all' and opt.lexical_split:
        logging.error('Lexical split is possible for only single word focus')
        sys.exit(1)

    if opt.focus == 'all' and opt.random_labels:
        logging.error('Random label baseline is possible for only single word focus for now')
        sys.exit(1)        
    # ---

    if opt.debug:
        logging.basicConfig(level=logging.DEBUG)
        with ipdb.launch_ipdb_on_exception():
            main(opt)
    else:
        main(opt)

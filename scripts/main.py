"""
Run the syntactic debiasing
"""

import sys
import os
import glob
import random
import argparse
import logging
import ipdb

import debiasing
import vectorize
from utils import reprs
from utils import arrange_data
from utils import dirs
from plotting import visualize

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


    #----- Create and Save Train-Test Splits -----
    if opt.save_splits:
        logging.info('Dividing data into training and test sets and saving these into splits/ dir')

        _, _, _, _, _, _, _, _ = arrange_data.train_test_split(cls1_instances, 
                                                               cls2_instances,
                                                               cls1_words, 
                                                               cls2_words,
                                                               opt,
                                                               do_save=True)
        sys.exit(0)


    #----- Arrange where to log the results -----
    logfile_base = dirs.prep_logfile(opt)


    #----- MAIN LOOP -----
    #      repeat exp_count times

    for experiment_number in range(opt.exp_count):

        logging.info(f'\n\n\nExperiment #{experiment_number}\n')


        #----- 1. Acquiring the train-test splits -----

        # If we are loding from existing pre-ready split files:
        if opt.use_ready_splits:
            X_train, Y_train, X_test, Y_test, \
            cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances \
                    = arrange_data.load_splits(experiment_number, opt)

        # If we want to create a split on the fly:
        else:
            X_train, Y_train, X_test, Y_test, \
            cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances \
                    = arrange_data.train_test_split(cls1_instances, 
                                                    cls2_instances,
                                                    cls1_words, 
                                                    cls2_words,
                                                    opt,
                                                    do_save=False)


        #----- 2. Debiasing -----

        # If iNLP debiasing is selected: 
        if opt.debias == 'iNLP':
            db = debiasing.iNLP_Debiasing(classifier='LogisticRegression',
                                          n_iterations=opt.n_iterations,
                                          reg_coeff=opt.reg_coeff,
                                          model=opt.model)

            db.debias(X_train, Y_train, X_test, Y_test, opt, logfile_base)


        # IF Bias-Direction Debiasing is selected:
        elif opt.debias == 'BDD':
            # FIXME: correct for layer-wise cleaning
            db = debiasing.BDD_Debiasing(cls1_train_instances[opt.train_dataset][opt.train_focus],
                                          cls2_train_instances[opt.train_dataset][opt.train_focus],
                                          cls1_test_instances[opt.test_dataset][opt.test_focus],
                                          cls2_test_instances[opt.test_dataset][opt.test_focus])
            #db.calc_bias_dir_vec()
            #db.plot_before_after()
            db.rotate(n_iterations=opt.n_iterations)



        #----- 3. MDS / tSNE plotting ------
        if opt.visualize: #'mds' or 'tsne'

            # Visualize original data (_before_ transformation)
            visualize.project(cls1_instances[opt.test_dataset][opt.test_focus],
                              cls2_instances[opt.test_dataset][opt.test_focus],
                              opt)
            
            # If we have done debiasing, visualize also _after_ transformation
            if opt.debias:
                visualize.project(cls1_instances[opt.test_dataset][opt.test_focus],
                                  cls2_instances[opt.test_dataset][opt.test_focus],
                                  opt, 
                                  with_debiasing=db)



        #----- 4. Vector Explorations -----

        if opt.vector_fun:
            if experiment_number == 0:
                vc = vectorize.Vectorize(opt.model)

            vc.set_data(cls1_instances, cls2_instances, cls1_words, cls2_words, opt)
            vectors_base_logdir = f'{opt.results_dir}/distances/{opt.task}/'

            if opt.plot_vectors:    
                foci_to_plot = ['verb', 'subject', 'object']
                vc.plot_word_senses_from_logs(vectors_base_logdir, opt, foci_to_plot)


            # Before the final experiment:
            elif experiment_number < opt.exp_count - 1:
                vc.calc_word_senses(opt.test_dataset, opt.test_focus,
                                    distance_fnc=vc.layerwise_eucdist, with_debiasing=db)

            # The final experiment:
            else:
                # At the end of the final experiment, save all the accumulated results
                vc.calc_word_senses(opt.test_dataset, opt.test_focus,
                                    distance_fnc=vc.layerwise_eucdist, with_debiasing=db,
                                    log_dir=vectors_base_logdir)
                # Then plot
                vc.plot_word_senses(opt)


# --- end of the main function





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasets', type=str, required=False, default=['SICK'], nargs='+',
                        help='dataset to use [SICK (default) | SICK_tensecorr (manually tense-corrected) | RNN]')

    parser.add_argument('--dataset_path', required=False, type=str, nargs='+',
                        help='path to the raw dataset location')

    parser.add_argument('--model', required=False, type=str,
                        default='BERT',
                        help='encoder model to use [BERT (default) | MT]')

    parser.add_argument('--language', required=False, type=str,
                        help='which language of translation is used, used for MT model only [ DE | DE-EL | CS-DE-EL ]')

    parser.add_argument('--load_reprs_path', required=False, type=str,
                        help='previously extracted representations\' path')

    parser.add_argument('--save_reprs_path', required=False, type=str,
                        help='save the extracted representations to')

    parser.add_argument('--mt_reprs_path', required=False, type=str,
                        help='if MT encoder is being used, path to the previously extracted representations from the MT model')

    parser.add_argument('--outdir', type=str, required=False, default='../outputs/',
                        help='path to dir where output logs will be saved')

    parser.add_argument('--task', required=True, default='active-passive',
                        help='which syntactic information to debias, default active-passive')

    parser.add_argument('--foci', required=False, type=str, default='verb', nargs='+',
                        help='which part of the sentence to focus on [verb [default] | subject | object | all]')

    parser.add_argument('--extract_only', action='store_true',
                        help='run only until the extraction and saving of representations')

    parser.add_argument('--debias', required=False, type=str,
                        help='debiasing method if it is to be performed [iNLP | GBDD | BAM]')

    parser.add_argument('--vector_fun', action='store_true',
                        help='whether to have fun with vectors')

    parser.add_argument('--cca', action='store_true',
                        help='do SVCCA/PWCCA analysis')

    parser.add_argument('--plot_results', action='store_true',
                        help='if active, will plot visualizations of the data')

    parser.add_argument('--lexical_split', action='store_true',
                        help='enforces that same lexical entities go to the same train/test set split')

    parser.add_argument('--lexical_split_across_datasets', action='store_true',
                        help='enforces that across multiple datasets, same lexical entities go to the same train/test set split')

    parser.add_argument('--use_shared_vocab', action='store_true',
                        help='restricts the different datasets to use only the instances with shared vocabulary')

    parser.add_argument('--train_dataset', required=False, type=str,
                        help='which dataset to train on (defaults to dataset) [SICK | RNN]')

    parser.add_argument('--test_dataset', required=False, type=str,
                        help='which dataset to test on (defaults to dataset) [SICK | RNN]')

    parser.add_argument('--train_focus', required=False, type=str,
                        help='which part of sentence to train on (defaults to focus) [verb | subject | object | all]')

    parser.add_argument('--test_focus', required=False, type=str,
                        help='which part of sentence to test on (defaults to focus) [verb | subject | object | all]')

    parser.add_argument('--train_on_layer', required=False, type=int,
                        help='(optional) which layer to train on')

    parser.add_argument('--test_on_layer', required=False, type=int,
                        help='(optional) which layer to test on')

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

    parser.add_argument('--visualize', type=str, required=False,
                        help='visualize layers before and after cleaning? [mds | tsne]')     

    parser.add_argument('--plot_vectors', action='store_true',
                        help='shall plot the vector distances before and after cleaning.')                                                

    parser.add_argument('--exp_count', type=int, default=1,
                        help='how many experiments to run.')

    parser.add_argument('--use_ready_splits', action='store_true',
                        help='to use pre-prepared splits or not.')

    parser.add_argument('--save_splits', action='store_true',
                        help='create and save the splits to load later.')

    parser.add_argument('--reg_coeff', type=float, default=0.0001,
                        help='inverse regularization coefficient for LogReg in iNLP debiasing (default=0.00001)')                                               

    parser.add_argument('--n_iterations', type=int, default=2,
                        help='number of iterations for iNLP debiasing algorithm (default=2)') 

    parser.add_argument('--results_dir', type=str, default='../results',
                        help='folder to log results (default=\'../results\')')     

    opt = parser.parse_args()

    if opt.seed:
        random.seed(opt.seed)


    # --- sanity checks for interactions between the parameters ---
    if opt.train_dataset and opt.test_dataset:
        opt.datasets = list(set([opt.train_dataset, opt.test_dataset]))

    if opt.train_focus and opt.test_focus:
        opt.foci = list(set([opt.train_focus, opt.test_focus]))

    if not opt.train_dataset:
        opt.train_dataset = opt.datasets[0]

    if not opt.test_dataset:
        opt.test_dataset = opt.datasets[0]

    if not opt.train_focus:
        opt.train_focus = opt.foci[0]

    if not opt.test_focus:
        opt.test_focus = opt.foci[0]


    if opt.transfer_projmatrix and opt.transfer_classifier:
        logging.error('--transfer_projmatrix and --transfer_classifier cannot be set at the same time')
        sys.exit(1)

    if opt.foci == 'all' and opt.debias == 'GBDD':
        logging.error('GBDD debiasing is usable with only single word focus for now')
        sys.exit(1)

    if opt.foci == 'all' and opt.lexical_split:
        logging.error('Lexical split is possible for only single word focus')
        sys.exit(1)

    if opt.foci == 'all' and opt.random_labels:
        logging.error('Random label baseline is possible for only single word focus for now')
        sys.exit(1)
    # ---

    if opt.debug:
        logging.basicConfig(level=logging.DEBUG)
        with ipdb.launch_ipdb_on_exception():
            main(opt)
    else:
        main(opt)

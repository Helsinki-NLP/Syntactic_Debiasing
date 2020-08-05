"""
Run the syntactic debiasing
Code structure inspired from: Juan Raul Vazquez Carillo
"""

import sys
import os
import glob
import random
import argparse
import logging
import ipdb
from utils import reprs
from utils import arrange_data
import debiasing
import vectorize
from plotting import visualize
from cca import CCA

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
    if opt.save_splits:
        logging.info('Separating training and test sets')
        #FIXME: This assumes single lexical item from each example sentence for now, eg. a verb.

        for experiment_number in range(opt.exp_count):
            X_train, Y_train, X_test, Y_test, \
                cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances \
                    = arrange_data.train_test_split(opt.model,
                                                 cls1_instances, cls2_instances,
                                                 cls1_words, cls2_words,
                                                 opt.dataset, 
                                                 opt.focus,                                                 
                                                 opt.lexical_split,
                                                 opt.random_labels,
                                                 experiment_number,
                                                 opt.language,
                                                 do_save=True)  
        sys.exit(0)


    if opt.transfer_projmatrix: 
        logfile_base = f'{opt.results_dir}/transfer_learning/{opt.model}/{opt.language}/{opt.train_on_dataset}-{opt.train_on_focus}_2_{opt.test_on_dataset}-{opt.test_on_focus}/transfer_results_reg_coeff_{opt.reg_coeff}_it_{opt.n_iterations}'
    else:
        logfile_base = f'{opt.results_dir}/classification_acc/{opt.model}/{opt.language}/{opt.dataset[0]}-{opt.focus[0]}/class_acc_reg_coeff_{opt.reg_coeff}_it_{opt.n_iterations}'
    os.system(f'mkdir -p {logfile_base}')

    for f in glob.glob(logfile_base + '/*'):
        os.remove(f)

    for experiment_number in range(opt.exp_count):
        if opt.use_ready_splits:
            X_train, Y_train, X_test, Y_test, \
                cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances \
                    = arrange_data.load_splits(opt.model, opt.dataset, opt.focus, experiment_number, opt.language)

        else:
            X_train, Y_train, X_test, Y_test, \
                cls1_train_instances, cls2_train_instances, cls1_test_instances, cls2_test_instances \
                    = arrange_data.train_test_split(opt.model, 
                                                     cls1_instances, cls2_instances,
                                                     cls1_words, cls2_words,
                                                     opt.dataset, 
                                                     opt.focus,                                         
                                                     opt.lexical_split,
                                                     opt.random_labels,
                                                     experiment_number,
                                                     opt.language,
                                                     do_save=False)


        #----- Debiasing -----
        if opt.debias == 'iNLP':
            db = debiasing.iNLP_Debiasing(classifier='LogisticRegression', n_iterations=opt.n_iterations+1, reg_coeff=opt.reg_coeff, model=opt.model)
            db.debias(X_train, Y_train, X_test, Y_test, 
                      opt.train_on_dataset, 
                      opt.test_on_dataset,
                      opt.train_on_focus, 
                      opt.test_on_focus,
                      opt.transfer_projmatrix, 
                      opt.transfer_classifier, 
                      opt.visualize, 
                      logfile_base)


        # FIXME: correct for layer-wise cleaning.
        elif opt.debias == 'BDD':
            db = debiasing.BDD_Debiasing(cls1_train_instances[opt.train_on_dataset][opt.train_on_focus], cls2_train_instances[opt.train_on_dataset][opt.train_on_focus], 
                                          cls1_test_instances[opt.test_on_dataset][opt.test_on_focus], cls2_test_instances[opt.test_on_dataset][opt.test_on_focus])
            db.calc_bias_dir_vec()
            db.plot_before_after()


        elif opt.debias == 'GBDD':
            db = debiasing.GBDD_Debiasing(cls1_train_instances[opt.train_on_dataset][opt.train_on_focus], cls2_train_instances[opt.train_on_dataset][opt.train_on_focus], 
                                          cls1_test_instances[opt.test_on_dataset][opt.test_on_focus], cls2_test_instances[opt.test_on_dataset][opt.test_on_focus])
            #db.calc_bias_dir_vec()
            #db.plot_before_after()
            db.rotate(n_iterations=opt.n_iterations)

        # to-implement
        #elif opt.debias == 'BAM':
        #    db = debiasing.BAM_Debiasing()
        #    db.debias(X_train, Y_train, X_test, Y_test)        
        

        #----- MDS / tSNE plotting ------
        if opt.visualize: #'mds' or 'tsne'
            visualize.project(cls1_instances[opt.test_on_dataset][opt.test_on_focus], 
                              cls2_instances[opt.test_on_dataset][opt.test_on_focus], 
                              opt.model,
                              opt.language,
                              opt.visualize)
            if opt.debias:
                visualize.project(cls1_instances[opt.test_on_dataset][opt.test_on_focus], 
                                  cls2_instances[opt.test_on_dataset][opt.test_on_focus], 
                                  opt.model,
                                  opt.language,
                                  opt.visualize, with_debiasing=db)


        #----- Vector Explorations -----


        if opt.vector_fun:
            if experiment_number == 0:
                vc = vectorize.Vectorize(opt.model)
             

            vc.set_data(cls1_instances, cls2_instances, cls1_words, cls2_words, opt.dataset, opt.focus)
            
            if experiment_number < opt.exp_count - 1:
                vc.calc_word_senses(opt.test_on_dataset, opt.test_on_focus, 
                                    distance_fnc=vc.layerwise_eucdist, with_debiasing=db)
            else:
                vc.calc_word_senses(opt.test_on_dataset, opt.test_on_focus, 
                                    distance_fnc=vc.layerwise_eucdist, with_debiasing=db,
                                    logfile=f'{opt.results_dir}/distances/{opt.model}/{opt.language}/{opt.dataset[0]}_{opt.focus[0]}_distances.pkl')
                vc.plot_word_senses(opt.test_on_dataset, opt.test_on_focus, is_with_debiasing=True)
                
                if opt.plot_vectors:
                    if opt.model == 'MT' and opt.language == 'DE-EL':
                        do_plot_legend = True
                    else:
                        do_plot_legend = False
                    vc.plot_word_senses_from_logs(f'{opt.results_dir}/distances/{opt.model}/{opt.language}', 
                                        opt.model, opt.language, opt.dataset, ['verb', 'subject', 'object'], do_plot_legend)

        #------ SVCCA / PWCCA -------

        if opt.cca:
            cca = CCA(cls1_instances[opt.train_on_dataset][opt.train_on_focus], 
                      cls2_instances[opt.train_on_dataset][opt.train_on_focus],
                      opt.train_on_layer-1,
                      cls1_instances[opt.test_on_dataset][opt.test_on_focus], 
                      cls2_instances[opt.test_on_dataset][opt.test_on_focus],
                      opt.test_on_layer-1)

            cca.apply_pwcca()
            cca.apply_svcca()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=False, default=['SICK'], nargs='+',
                        help='dataset to use [SICK (default) | SICK_tensecorr (manually tense-corrected) | RNN]')

    parser.add_argument('--dataset_path', required=False, type=str, nargs='+',
                        default=['/scratch/project_2002233/debiasing/data/SICK/Filtered'],
                        help='path to the raw dataset location. Defaults to puhti:SICK location')  

    parser.add_argument('--model', required=False, type=str,
                        default='BERT',
                        help='encoder model to use [BERT (default) | MT]')  

    parser.add_argument('--language', required=False, type=str,
                        help='which language of translation is used, for MT model only [ DE | DE-EL | CS-DE-EL ]')  

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

    parser.add_argument('--focus', required=False, type=str, default='verb', nargs='+',
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

    parser.add_argument('--train_on_dataset', required=False, type=str,
                        help='which dataset to train on (defaults to dataset) [SICK | RNN]')

    parser.add_argument('--test_on_dataset', required=False, type=str,
                        help='which dataset to test on (defaults to dataset) [SICK | RNN]') 

    parser.add_argument('--train_on_focus', required=False, type=str,
                        help='which part of sentence to train on (defaults to focus) [verb | subject | object | all]')

    parser.add_argument('--test_on_focus', required=False, type=str,
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
    if opt.train_on_dataset and opt.test_on_dataset:
        opt.dataset = list(set([opt.train_on_dataset, opt.test_on_dataset]))

    if opt.train_on_focus and opt.test_on_focus:
        opt.focus = list(set([opt.train_on_focus, opt.test_on_focus]))

    if not opt.train_on_dataset:
        opt.train_on_dataset = opt.dataset[0]

    if not opt.test_on_dataset: 
        opt.test_on_dataset = opt.dataset[0]   

    if not opt.train_on_focus:
        opt.train_on_focus = opt.focus[0]

    if not opt.test_on_focus: 
        opt.test_on_focus = opt.focus[0]


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

'''
Run the syntactic debiasing
Code structure inspired from: Juan Raul Vazquez Carillo
'''

import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, default=['SICK'], nargs='+',
                        help='Dataset to use [SICK (default) | RNN]')

    parser.add_argument('--cuda', action='store_true',
                        help='Whether to use a cuda device.')
    
    parser.add_argument('--dataset_path', required=False, type=str, nargs='+',
                        default=['/scratch/project_2002233/debiasing/data/SICK/Filtered'],
                        help='Path to the raw dataset location. Defaults to puhti:SICK location')

    parser.add_argument('--debug_mode', action='store_true',
                        help='Launch ipdb debugger if script crashes.')
    
    parser.add_argument('--load_reprs_path', required=False, type=str, nargs='+',
                        help='Previously extracted representations\' locations')

    parser.add_argument('--save_reprs_path', required=False, type=str, nargs='+',
                        help='Save the extracted representations to')

    parser.add_argument('--outdir', type=str, required=False, default='../outputs/',
                        help='path to dir where output logs will be saved')

    parser.add_argument('--plot_results', action='store_true',
                        help='if active, will plot visualizations of the data')

    parser.add_argument('--task', required=True, default='active-passive',
                        help='which syntactic information to debias, default active-passive')    

    parser.add_argument('--focus', required=True, default='verb',
                        help='which part of the sentence to focus on [verb [default] | subject | object | all')   

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

    return parser
import os
import glob

def prep_logfile(opt):
    if opt.transfer_projmatrix:
        logfile_base = f'{opt.results_dir}/transfer_learning/'+   \
                       f'{opt.task}/{opt.model}/{opt.language}/'+ \
                       f'{opt.train_on_dataset}-{opt.train_on_focus}_2_{opt.test_on_dataset}-{opt.test_on_focus}/'+ \
                       f'transfer_results_reg_coeff_{opt.reg_coeff}_it_{opt.n_iterations}'
    else:
        logfile_base = f'{opt.results_dir}/classification_acc/'+  \
                       f'{opt.task}/{opt.model}/{opt.language}/'+ \
                       f'{opt.datasets[0]}-{opt.foci[0]}/'+       \
                       f'class_acc_reg_coeff_{opt.reg_coeff}_it_{opt.n_iterations}'

    os.system(f'mkdir -p {logfile_base}')

    for f in glob.glob(logfile_base + '/*'):
        os.remove(f)

    return logfile_base
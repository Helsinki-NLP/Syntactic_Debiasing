# Syntactic_Debiasing


### Data folders on Puhti:
* SICK data: Under /scratch/project_2002233/debiasing/data/SICK/Filtered/
    * Active sentences: /scratch/project_2002233/debiasing/data/SICK/Filtered/SICK.active.pos.parse.conll
    * Passive sentences: /scratch/project_2002233/debiasing/data/SICK/Filtered/SICK.passive.pos.parse.conll

* RNN-Priming data: Under /scratch/project_2002233/debiasing/data/RNN-Priming/test
    * Active sentences: Under /scratch/project_2002233/debiasing/data/RNN-Priming/test/\*/list\*\_orc.pos.parse.conll
    * Passive sentences: Under /scratch/project_2002233/debiasing/data/RNN-Priming/test/\*/list\*\_prc.pos.parse.conll

### Running the code on Puhti:
_You can use the checked-out repository at /scratch/project_2002233/debiasing/github/_

    $ cd /scratch/project_2002233/debiasing/github/scripts
    $ module load pytorch/1.2.0

#### To extract representations:
    $ python main.py --extract_only \ 
                    --dataset SICK \
                    --dataset_path /scratch/project_2002233/debiasing/data/SICK/Filtered/ \
                    --save_reprs_path /scratch/project_2002233/debiasing/representations/ \
                    --cuda


#### To run with previously extracted representations:


#### Enforce options:
    --lexical_split: 
    --cross_dataset_lexical_split:
    --clauses_only:
    --train_on:
    --test_on:

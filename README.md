# Syntactic_Debiasing


### Data folders on cluster:
* SICK data: Under /path/to/project/data/SICK/Filtered/
    * Active sentences: /path/to/project/data/SICK/Filtered/SICK.active.pos.parse.conll
    * Passive sentences: /path/to/project/data/SICK/Filtered/SICK.passive.pos.parse.conll

* Manually tense-corrected SICK data: Under /path/to/project/data/SICK_tensecorr
    * Active sentences: /path/to/project/data/SICK_tensecorr/SICK_tensecorr.active.pos.parse.conll
    * Passive sentences: /path/to/project/data/SICK_tensecorr/SICK_tensecorr.passive.pos.parse.conll

* RNN-Priming data: Under /path/to/project/data/RNN-Priming/combined
    * Active sentences: Under /path/to/project/data/RNN-Priming/combined/RNN.active.pos.parse.conll
    * Passive sentences: Under /path/to/project/data/RNN-Priming/combined/RNN.passive.pos.parse.conll


### Running the code on cluster:
_You can use the checked-out repository at /path/to/project/github/_

    $ cd /path/to/project/github/scripts
    $ module load pytorch/1.2.0

#### To extract representations:

Eg. SICK dataset:
    $ python main.py --extract_only \ 
                    --dataset SICK \
                    --dataset_path /path/to/project/data/SICK/Filtered/ \
                    --save_reprs_path /path/to/project/representations/ \
                    --task active-passive \
                    --focus verb

Eg. SICK dataset + RNN dataset:
    $ python main.py --extract_only \ 
                    --dataset SICK RNN \
                    --dataset_path /path/to/project/data/SICK/Filtered/ /path/to/project/data/RNN-Priming/combined/ \
                    --save_reprs_path /path/to/project/representations/ \
                    --task active-passive \
                    --focus verb

#### To run with previously extracted representations:


#### Optional arguments:
    --cuda: 
    --lexical_split: 
    --cross_dataset_lexical_split:
    --clauses_only:
    --train_on:
    --test_on:

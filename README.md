# Syntactic_Debiasing


### Data folders on cluster:
* SICK data: Under /path/to/project/data/SICK/Filtered/
    * Active sentences: /path/to/project/data/SICK/Filtered/SICK.active.pos.parse.conll
    * Passive sentences: /path/to/project/data/SICK/Filtered/SICK.passive.pos.parse.conll

* RNN-Priming data: Under /path/to/project/data/RNN-Priming/test
    * Active sentences: Under /path/to/project/data/RNN-Priming/test/\*/list\*\_orc.pos.parse.conll
    * Passive sentences: Under /path/to/project/data/RNN-Priming/test/\*/list\*\_prc.pos.parse.conll

### Running the code on cluster:
_You can use the checked-out repository at /path/to/project/github/_

    $ cd /path/to/project/github/scripts
    $ module load pytorch/1.2.0

#### To extract representations:
    $ python main.py --extract_only \ 
                    --dataset SICK \
                    --dataset_path /path/to/project/data/SICK/Filtered/ \
                    --save_reprs_path /path/to/project/representations/ \
                    --cuda


#### To run with previously extracted representations:


#### Enforce options:
    --lexical_split: 
    --cross_dataset_lexical_split:
    --clauses_only:
    --train_on:
    --test_on:

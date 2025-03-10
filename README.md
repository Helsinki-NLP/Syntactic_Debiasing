# Syntactic Debiasing of Contextualized Language Representations from Imprint of Passivization and Negation

Controlling the Imprint of Passivization and Negation in Contextualized Representations [[bib](#citation)]

Hande Celikkanat, Sami Virpioja, Jorg Tiedemann, and Marianna Apidianaki 


## How-To

### Running the code:
```
## Extracting representations:
python main.py --dataset RNN-priming-short-5000 --dataset_path ../data/RNN-Priming-short/ --save_reprs_path ../representations/ --task active-passive --focus subject --extract_only --debug

## Saving splits:
python main.py --dataset RNN-priming-short-5000 --load_reprs_path ../representations/ --task active-passive --focus subject --layer 12 --save_splits --different_split_exp_count 20

## Obtaining word-sense graphs:
python main.py --dataset SICK --load_reprs_path ../representations/ --task active-passive --focus verb --debias Goldberg --lexical_split --test_on SICK --train_on SICK --vector_fun --exp_count 20 --layer 12

## Plotting MDS:
python main.py --dataset SICK --load_reprs_path ../representations/ --task active-passive --focus verb --debias Goldberg --lexical_split --test_on SICK --train_on SICK --exp_count 1 --layer 12 --plot_mds

## Plotting distances:
python main.py --dataset RNN-priming-short-1000 --load_reprs_path ../representations/ --task active-passive --focus object --debias Goldberg --lexical_split --test_on RNN-priming-short-1000 --train_on RNN-priming-short-1000 --exp_count 1 --vector_fun --plot_vectors --debug
```

### Organization of data folders:
* SICK data: Under /path/to/project/data/SICK/Filtered/
    * Active sentences: /path/to/project/data/SICK/Filtered/SICK.active.pos.parse.conll
    * Passive sentences: /path/to/project/data/SICK/Filtered/SICK.passive.pos.parse.conll

* Manually tense-corrected SICK data: Under /path/to/project/data/SICK_tensecorr
    * Active sentences: /path/to/project/data/SICK_tensecorr/SICK_tensecorr.active.pos.parse.conll
    * Passive sentences: /path/to/project/data/SICK_tensecorr/SICK_tensecorr.passive.pos.parse.conll

* RNN-Priming data: Under /path/to/project/data/RNN-Priming/combined
    * Active sentences: Under /path/to/project/data/RNN-Priming/combined/RNN.active.pos.parse.conll
    * Passive sentences: Under /path/to/project/data/RNN-Priming/combined/RNN.passive.pos.parse.conll



## Citation
```
@inproceedings{ynie2020chaosnli,
	Author = {Hande \c{C}elikkanat, Sami Virpioja, J\"{o}rg Tiedemann, and Marianna Apidianaki},
	Booktitle = {Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP},
	Title = {Controlling the Imprint of Passivization and Negation in Contextualized Representations},
	Year = {2020}
}
```

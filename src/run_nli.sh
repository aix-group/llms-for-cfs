#!/bin/sh
nohup python eval_augmentation.py --data_origin counterfactually-augmented-data --task NLI --training_split original --gpu 0 > nli_orig.out &&
nohup python eval_augmentation.py --data_origin counterfactually-augmented-data --task NLI --training_split revised_hypothesis --gpu 0 > nli_rh.out &&
nohup python eval_augmentation.py --data_origin counterfactually-augmented-data --task NLI --training_split revised_premise --gpu 0 > nli_rp.out &&
nohup python eval_augmentation.py --data_origin counterfactually-augmented-data --task NLI --training_split revised_combined --gpu 0 > nli_rcombined.out &&
nohup python eval_augmentation.py --data_origin counterfactually-augmented-data --task NLI --training_split all_combined --gpu 0 > nli_all.out



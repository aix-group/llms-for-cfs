#!/bin/sh
nohup python eval_no_training.py --data_origin counterfactually-augmented-data --task sentiment --training_split orig --gpu 0 --model bert-base-uncased &&
nohup python eval_no_training.py --data_origin counterfactually-augmented-data --task sentiment --training_split new --gpu 0 --model bert-base-uncased &&
nohup python eval_no_training.py --data_origin counterfactually-augmented-data --task sentiment --training_split combined --gpu 0 --model bert-base-uncased &&
nohup python eval_no_training.py --data_origin llms/llama2-20231209/ --task sentiment --training_split new --gpu 0 --model bert-base-uncased &&
nohup python eval_no_training.py --data_origin llms/llama2-20231209/ --task sentiment --training_split combined --gpu 0 --model bert-base-uncased &&
nohup python eval_no_training.py --data_origin llms/mistral-20240118/ --task sentiment --training_split new --gpu 0 --model bert-base-uncased &&
nohup python eval_no_training.py --data_origin llms/mistral-20240118/ --task sentiment --training_split combined --gpu 0 --model bert-base-uncased &&
nohup python eval_no_training.py --data_origin llms/gpt4-20240313/ --task sentiment --training_split new --gpu 0 --model bert-base-uncased &&
nohup python eval_no_training.py --data_origin llms/gpt4-20240313/ --task sentiment --training_split combined --gpu 0 --model bert-base-uncased 
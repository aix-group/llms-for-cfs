#!/bin/sh
nohup python eval_no_training.py --data_origin counterfactually-augmented-data --task NLI --training_split original --gpu 0 --model bert-base-uncased  > ./logs/nli_orig_notrain.out &&
nohup python eval_no_training.py --data_origin counterfactually-augmented-data --task NLI --training_split revised_hypothesis --gpu 0 --model bert-base-uncased > ./logs/nli_rh_notrain.out &&
nohup python eval_no_training.py --data_origin counterfactually-augmented-data --task NLI --training_split revised_premise --gpu 0 --model bert-base-uncased > ./logs/nli_rp_notrain.out &&
nohup python eval_no_training.py --data_origin counterfactually-augmented-data --task NLI --training_split revised_combined --gpu 0 --model bert-base-uncased > ./logs/nli_rcombined_notrain.out &&
nohup python eval_no_training.py --data_origin counterfactually-augmented-data --task NLI --training_split all_combined --gpu 0 --model bert-base-uncased > ./logs/nli_all_notrain.out &&
nohup python eval_no_training.py --data_origin llms/llama2-20231209/ --task NLI --training_split revised_hypothesis --gpu 0 --model bert-base-uncased > ./logs/llama2_nli_rh_notrain.out &&
nohup python eval_no_training.py --data_origin llms/llama2-20231209/ --task NLI --training_split revised_premise --gpu 0 --model bert-base-uncased > ./logs/llama2_nli_rp_notrain.out &&
nohup python eval_no_training.py --data_origin llms/llama2-20231209/ --task NLI --training_split revised_combined --gpu 0 --model bert-base-uncased > ./logs/llama2_nli_rcombined_notrain.out &&
nohup python eval_no_training.py --data_origin llms/llama2-20231209/ --task NLI --training_split all_combined --gpu 0 --model bert-base-uncased > ./logs/llama2_nli_all_notrain.out &&
nohup python eval_no_training.py --data_origin llms/mistral-20240118/ --task NLI --training_split revised_hypothesis --gpu 0 --model bert-base-uncased > ./logs/mistral_nli_rh_notrain.out &&
nohup python eval_no_training.py --data_origin llms/mistral-20240118/ --task NLI --training_split revised_premise --gpu 0 --model bert-base-uncased > ./logs/mistral_nli_rp_notrain.out &&
nohup python eval_no_training.py --data_origin llms/mistral-20240118/ --task NLI --training_split revised_combined --gpu 0 --model bert-base-uncased > ./logs/mistral_nli_rcombined_notrain.out &&
nohup python eval_no_training.py --data_origin llms/mistral-20240118/ --task NLI --training_split all_combined --gpu 0 --model bert-base-uncased > ./logs/mistral_nli_all_notrain.out &&
nohup python eval_no_training.py --data_origin llms/gpt4-20240313/ --task NLI --training_split revised_hypothesis --gpu 0 --model bert-base-uncased > ./logs/gpt4_nli_rh_notrain.out &&
nohup python eval_no_training.py --data_origin llms/gpt4-20240313/ --task NLI --training_split revised_premise --gpu 0 --model bert-base-uncased > ./logs/gpt4_nli_rp_notrain.out &&
nohup python eval_no_training.py --data_origin llms/gpt4-20240313/ --task NLI --training_split revised_combined --gpu 0 --model bert-base-uncased > ./logs/gpt4_nli_rcombined_notrain.out &&
nohup python eval_no_training.py --data_origin llms/gpt4-20240313/ --task NLI --training_split all_combined --gpu 0 --model bert-base-uncased > ./logs/gpt4_nli_all_notrain.out

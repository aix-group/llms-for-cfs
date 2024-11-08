from transformers import AutoTokenizer
import transformers
import torch
import json
import pandas as pd
import time
import numpy as np
import argparse
import os

from sentence_transformers import SentenceTransformer, util
import difflib
from utils.utils_gen import *
from datetime import datetime
current_date = datetime.now()
date_string = current_date.strftime("%Y%m%d%H%M%S")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def create_simple_prompt(instance, target_sentence = "premise"):
    """ instance: original_sentence1,original_sentence2,contrast text,original label,contrast label,closest_instance """
    example_map = {
        "neutral":{
            "premise": "Seven people are riding bikes on a sandy track.",
            "hypothesis": "The people are racing."
        },
        "entailment": {
            "premise": "Seven people are racing bikes on a sandy track.",
            "hypothesis": "People are riding bikes on a track."
        },
        "contradiction":{
            "premise": "Seven people are repairing bikes on a sandy track.",
            "hypothesis": "People are walking on a sandy track."
        }
    }
    sentence_1 = instance['original_sentence1']
    sentence_2 = instance['original_sentence2']
    orig_label = instance['original label']
    target_label = example['contrast label']
    if target_sentence == "premise":
        temp = f"""Premise: {example_map[orig_label]['premise']}\nHypothesis: {example_map["neutral"]['hypothesis']}"""
    else:
        temp = f"""Premise: {example_map["neutral"]['premise']}\nHypothesis: {example_map[orig_label]['hypothesis']}"""
    template = f"""Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one. Do not make any unnecessary changes. For example:
Original relation: {orig_label}
{temp}
Target relation: {target_label}
(Edited {target_sentence}): {example_map[target_label][target_sentence]}


Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one. Do not make any unnecessary changes. Do not add anything else.
Original relation: {orig_label}
Premise: {sentence_1}
Hypothesis: {sentence_2}
Target relation: {target_label}
(Edited {target_sentence}):"""
    return template
def create_prompt(example, closest_instance, contrast_instance, target_sentence):

    """
    closest_instance: sentence1. sentence2
    """
    text_changes = compare_text(closest_instance[target_sentence], contrast_instance[target_sentence])
    text_change_split = text_changes.split("\n")
    temp_step_1 = [t.split(" to ")[0] for t in text_change_split]
    step_1 = ",\n".join(temp_step_1)
    step_2 = text_changes
    template = f"""Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the target relation from the original one. Do not make any unnecessary changes. For example:
Original relation: {closest_instance['gold_label']}
Two original sentences: {closest_instance['concat_sentence']}
Target relation: {contrast_instance['gold_label']}
Target sentence: {target_sentence}
Step 1: Identify phrases, words in the {target_sentence} leading to the {closest_instance['gold_label']} relation: 
{step_1}
Step 2: Change these phrases, words to get {contrast_instance['gold_label']} relation with minimal changes: 
{step_2}
Step 3: replace the phrases, words from step 1 in the original text by the phrases, words, sentences in step 2:
(Edited {target_sentence}): {contrast_instance[target_sentence]}
#####End Example####
Request: Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {contrast_instance['gold_label']} relation from the original one. Do not make any unnecessary changes. Do not add anything else.
Enclose the generated text within <new> tags.
Original relation: {example['gold_label']}
Two original sentences: {example['concat_sentence']}
Target relation: {contrast_instance['gold_label']}
Target sentence: {target_sentence}"""

    return template
def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Benchmark arguments")

    # Add positional arguments
    parser.add_argument("-split", required=True, help="Split set that needs to generate the counterfactual ", choices=['dev', 'train', 'test'])
    parser.add_argument("-model", required=True, help="Model to generate counterfactual ")
    parser.add_argument("-target_sentence", required=True, help="target sentence to change", choices=['sentence1', 'sentence2'])

    # Add optional arguments
    # parser.add_argument("-batch_size", type=int, default=100, help="Batch size for evaluation.")
    # parser.add_argument("-return_csv", action="store_true", help="Whether to save the metrics to the original CSV file.")

    # Parse the command line arguments
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()

    split = args.split
    # llm_model = "meta-llama/Llama-2-7b-chat-hf"
    llm_model = args.model
    # llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name, llm_pipeline, tokenizer = load_model(llm_model)
    sbert_model_name = "all-mpnet-base-v2"
    dev_pairs = pd.read_csv("../../counterfactually-augmented-data/NLI/original/dev.tsv", delimiter="\t")
    dev_hypothesis = pd.read_csv("../../counterfactually-augmented-data/NLI/revised_hypothesis/dev.tsv", delimiter="\t")
    dev_premise = pd.read_csv("../../counterfactually-augmented-data/NLI/revised_premise/dev.tsv", delimiter="\t")
    test_pairs = pd.read_csv("../../counterfactually-augmented-data/NLI/original/test.tsv", delimiter="\t")
    train_pairs = pd.read_csv("../../counterfactually-augmented-data/NLI/original/train.tsv", delimiter="\t")
    if split == "dev":
        df_merge = dev_pairs.copy()
        # dev_pairs = train_pairs.copy()
    if split == "train":
        df_merge = train_pairs.copy()
        # df_merge = df_merge.iloc[1232:1233]
    if split == "test":
        df_merge = test_pairs.copy()
    # df_merge = dev_pairs
    # dev_pairs = train_pairs
    df_merge['concat_sentence'] = df_merge.apply(lambda x: x['sentence1'] + " " + x['sentence2'], axis=1)

    #load model
    sbert_model = SentenceTransformer(sbert_model_name)
   

    # passage_embedding = model.encode(list_of_text)

    #split sentiment pairs
    dev_pairs['concat_sentence'] = dev_pairs.apply(lambda x: x['sentence1'] + " " + x['sentence2'] if x['sentence1'][-1] == "." else x['sentence1'] + ". " + x['sentence2'], axis=1)
    dev_neutral_pairs = dev_pairs[dev_pairs['gold_label'] == "neutral"]
    dev_entailment_pairs = dev_pairs[dev_pairs['gold_label'] == "entailment"]
    dev_contradiction_pairs = dev_pairs[dev_pairs['gold_label'] == "contradiction"]

    #generate sentiment embeddings for each sentiments:
    neutral_embeddings = sbert_model.encode(dev_neutral_pairs['concat_sentence'].to_list())
    entailment_embeddings = sbert_model.encode(dev_entailment_pairs['concat_sentence'].to_list())
    contradiction_embeddings = sbert_model.encode(dev_contradiction_pairs['concat_sentence'].to_list())
    label_to_embeddings = {"neutral":neutral_embeddings,
                            "entailment":entailment_embeddings,
                            "contradiction":contradiction_embeddings}
    label_to_sentences= {"neutral":dev_neutral_pairs,
                            "entailment":dev_entailment_pairs,
                            "contradiction":dev_contradiction_pairs}
    list_contrast_labels = []
    list_contrast_texts = []
    list_original_sent1 = []
    list_original_sent2 = []
    list_original_labels = []
    list_original_texts = []
    list_closest_instances = []
    list_prompts = []
    # df_test = test_pairs
    #pick an instance
    # Specify the chunk size
    chunk_size = 1
    target_sentence = args.target_sentence
    target_sentence_to_df = {"sentence1":dev_premise, #sentence1	sentence2	gold_label
                    "sentence2":dev_hypothesis}
    # Calculate the number of chunks
    num_chunks = len(df_merge) // chunk_size + 1
    start_time = time.time()
    for i in range(num_chunks):
        list_prompts = []
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk = df_merge.iloc[start_idx:end_idx]
        for _, example in chunk.iterrows():

            dev_embeddings = label_to_embeddings[example['gold_label']]
            dev_sentences = label_to_sentences[example['gold_label']]
            #return index of the closest instance
            index = int(find_closest_instance(sbert_model.encode(example['concat_sentence']), dev_embeddings))
            #extract the closest instance
            closest_instance = dev_sentences.iloc[int(index)]
            #get dev set that contains either premise or hypothesis changed
            dev_contrast = target_sentence_to_df[target_sentence]
            index = int(closest_instance.name)
            for i in [2*index, 2*index+1]:
                list_original_sent1.append(example['sentence1'])
                list_original_sent2.append(example['sentence2'])
                list_original_labels.append(example['gold_label'])
            # closest_instance = reference_pairs.iloc[int(index)]
            
                list_closest_instances.append(closest_instance['concat_sentence'])
                contrast_instance = dev_contrast.iloc[i]
                # original_text = closest_instance[target_sentence]
                # text_changes = compare_text(original_text, contrast_text)
                prompt = create_prompt(example,closest_instance,contrast_instance,target_sentence)
                # create_prompt(example,closest_instance,text_changes,target_sentence)
                list_prompts.append(prompt)
                list_contrast_labels.append(contrast_instance['gold_label'])
    
        if "gpt" in llm_model:
            fp = open(f'raw_snli_{model_name}_0125_{split}_{target_sentence}_{date_string}.txt', 'a')
            for prompt in list_prompts:
                gpt_prompt =[
                        {
                            "role": "user",
                            "content": prompt
                        }
                        ]
                results = delayed_completion(model=llm_model,
                                    messages=gpt_prompt,
                                    max_tokens=256)
                raw_text = results.choices[0].message.content
                
                fp.write("[start]%s[end]\n" % raw_text)
                text_split = raw_text.split(f": ")[-1]
                contrast_text = text_split.split("\n")[0]
                list_contrast_texts.append(contrast_text)
        else:
            sequences = llm_pipeline(
                list_prompts,
                do_sample=True,
                top_k=50,
                num_return_sequences=1,
                max_new_tokens=256,
            )
            
            with open(f'raw_snli_{model_name}_{split}_{target_sentence}_{date_string}.txt', 'a') as fp:
                for seq in sequences:
                # print(seq[0]['generated_text']) 
                    text = seq[0]['generated_text']
                    fp.write("[start]%s[end]\n" % text)
            for seq in sequences:
                text = seq[0]['generated_text']
                text = text.split("#####End Example####")[1]
                if len(text.split(f"(Edited {target_sentence}): ")) > 1:
                    text_split = text.split(f"(Edited {target_sentence}): ")[1]
                else:
                    text_split = text.split(f"(Edited {target_sentence}): ")[0]
                contrast_text = text_split.split("\n")[0]
                list_contrast_texts.append(contrast_text)
    end_time = time.time()
    pd.DataFrame({"original_sentence1": list_original_sent1,
                "original_sentence2":list_original_sent2,
                "contrast text": list_contrast_texts,
                "original label": list_original_labels,
                "contrast label": list_contrast_labels,
                "closest_instance": list_closest_instances}).to_csv(f"snli_{target_sentence}_{model_name}_{split}_{date_string}.csv")
    duration = end_time - start_time
    print(duration)
    # with open(r'duration_.txt', 'w') as fp:
    
    #     fp.write("Duration: %s" % str(duration))
    

    # assert len(df_merge) == len(list_contrast_texts) == len(list_original_texts) == len(list_original_labels) == len(list_contrast_labels)
    # assert list_original_labels[0] != list_contrast_labels[0]
    # assert list_original_texts[0] != list_contrast_texts[0]
import difflib
import argparse
import pandas as pd
from transformers import AutoTokenizer
import transformers
import torch
from datetime import datetime
from openai import OpenAI
import re
current_date = datetime.now()
date_string = current_date.strftime("%Y%m%d%H%M%S")
mapping = {"sentence1":"Premise",
               "sentence2":"Hypothesis"}
def compare_text(s1, s2):
    words_s1 = s1.split()
    words_s2 = s2.split()
    final_string = ""
    d = difflib.SequenceMatcher(None, words_s1, words_s2)
    for tag, i1, i2, j1, j2 in d.get_opcodes():
        if tag == 'delete':
            final_string+=f'"{ " ".join(words_s1[i1:i2]) }" to ""'
            final_string+="\n"
        elif tag == 'equal':
            continue
        elif tag == 'insert':
            final_string+=f'"" to "{ " ".join(words_s2[j1:j2]) }"'
            final_string+="\n"
        elif tag == 'replace':
            final_string+=f'"{ " ".join(words_s1[i1:i2]) }" to "{ " ".join(words_s2[j1:j2]) }"'
            final_string+="\n"
    return final_string
def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Benchmark arguments")

    # Add positional arguments
    parser.add_argument("-split", required=True, help="Split set that needs to generate the counterfactual ", choices=['dev', 'train', 'test'])
    parser.add_argument("-model", required=True, help="Model to generate counterfactual ")
    # parser.add_argument("-target_sentence", required=True, help="target sentence to change", choices=['sentence1', 'sentence2'])
    # parser.add_argument("-cot", required=True, help="apply COT")
    # Parse the command line arguments
    args = parser.parse_args()
    return args
def create_simple_prompt(instance, target_sentence = "sentence1"):
    """ instance: original_sentence1,original_sentence2,contrast text,original label,contrast label,closest_instance """
    example_map = {
        "neutral":{
            "sentence1": "Seven people are riding bikes on a sandy track.",
            "sentence2": "The people are racing."
        },
        "entailment": {
            "sentence1": "Seven people are racing bikes on a sandy track.",
            "sentence2": "People are riding bikes on a track."
        },
        "contradiction":{
            "sentence1": "Seven people are repairing bikes on a sandy track.",
            "sentence2": "People are walking on a sandy track."
        }
    }
    sentence_1 = instance['original_sentence1']
    sentence_2 = instance['original_sentence2']
    orig_label = instance['original label']
    target_label = instance['contrast label']
    
    if target_sentence == "sentence1":
        temp = f"""Premise: {example_map[orig_label]['sentence1']}\nHypothesis: {example_map["neutral"]['sentence2']}"""
    else:
        temp = f"""Premise: {example_map["neutral"]['sentence1']}\nHypothesis: {example_map[orig_label]['sentence2']}"""

    text_changes = compare_text(example_map[orig_label][target_sentence], example_map[target_label][target_sentence])
    text_change_split = text_changes.split("\n")
    temp_step_1 = [t.split(" to ")[0] for t in text_change_split]
    step_1 = ",\n".join(temp_step_1)
    step_2 = text_changes
    
    template = f"""Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {mapping[target_sentence]} with minimal edits to achieve the {target_label} relation from the original one. Do not make any unnecessary changes. For example:
Original relation: {orig_label}
{temp}
Target relation: {target_label}
Step 1: Identify phrases, words in the {mapping[target_sentence]} leading to the {target_label} relation: 
{step_1}
Step 2: Change these phrases, words to get {target_label} relation with minimal changes: 
{step_2}
Step 3: replace the phrases, words from step 1 in the original text by the phrases, words, sentences in step 2:
Edited {mapping[target_sentence]}: {example_map[target_label][target_sentence]}


Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {mapping[target_sentence]} with minimal edits to achieve the {target_label} relation from the original one. Do not make any unnecessary changes. Do not add anything else.
Original relation: {orig_label}
Premise: {sentence_1}
Hypothesis: {sentence_2}
Target relation: {target_label}
"""
    

    template_simple  = f"""Given two sentences (a premise and a hypothesis) and their original relationship, which determines whether the hypothesis entails the premise (label: entailment), contradicts the premise (label: contradiction), or is neutral to the premise (label: neutral). Change the {mapping[target_sentence]} with minimal edits to achieve the target label {target_label} from the original label. Do not make any unnecessary changes.

Example:
Original relation: {orig_label}
{temp}
Target relation: {target_label}
Edited {mapping[target_sentence]}: {example_map[target_label][target_sentence]}

Test:
Original relation: {orig_label}
Premise: {sentence_1}
Hypothesis: {sentence_2}
Target relation: {target_label}
Edited {mapping[target_sentence]} (Return a sentence only): """
    template_simple = {
        "role": "user",
        "content": template
    }
    return [template_simple]
def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Benchmark arguments")

    # Add positional arguments
    parser.add_argument("-split", required=True, help="Split set that needs to generate the counterfactual ", choices=['dev', 'train', 'test'])
    parser.add_argument("-model", required=True, help="Model to generate counterfactual ")
    parser.add_argument("-target_sentence", required=True, help="target sentence to change", choices=['sentence1', 'sentence2'])
    # parser.add_argument("-path", required=True, help="path to csv file")
    # parser.add_argument("-cot", required=True, help="apply COT")
    # Parse the command line arguments
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    
    
    args = get_args()
    # path = args.path
    split = args.split
    llm_model = args.model
    target_sentence = args.target_sentence
    path = f"snli_{target_sentence}_gpt-4_{split}.csv"
    df = pd.read_csv(path)
    if "gpt" not in llm_model:
        model_name = llm_model.split("/")[1]
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        # tokenizer.pad_token_id=tokenizer.eos_token_id
        tokenizer.eos_token_id = tokenizer.encode('.')[0]
        llm_pipeline = transformers.pipeline(
            "text-generation",
            model=llm_model,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer = tokenizer,
        )
    else:
        client = OpenAI()
        if "gpt-4" in llm_model:
            model_name = "gpt-4"
        else:
            model_name = "gpt-35"

    batch_size = 100
    # num_chunks = len(df) // chunk_size
    fp = open(f'raw_snli_{model_name}_{split}_{target_sentence}_{date_string}.txt', 'a')
    list_contrast_texts = []
    pattern_search = rf"(?:Edited {mapping[target_sentence]}): (.*?)(?:\n|$)"
    for i in range(0, df.shape[0], batch_size):
        batch = df.iloc[i:i+batch_size]
        list_prompts = []
        list_prompts = [create_simple_prompt(instance,target_sentence) for _, instance in batch.iterrows()]
        if "gpt" in llm_model:
            for prompt in list_prompts:
                gpt_prompt =[
                        {
                            "role": "user",
                            "content": prompt
                        }
                        ]
                results = client.chat.completions.create(model=llm_model,
                                    messages=gpt_prompt,
                                    max_tokens=128)
                raw_text = results.choices[0].message.content
                text_split = raw_text.split(f": ")[-1]
                contrast_text = text_split.split("\n")[0]
                fp.write("[start]%s[end]\n" % raw_text)
                list_contrast_texts.append(contrast_text)
            
        else:
            sequences = llm_pipeline(
                list_prompts,
                do_sample=True,
                top_k=50,
                num_return_sequences=1,
                max_new_tokens=256,
            )
            for seq in sequences:
            # print(seq[0]['generated_text']) 
                text = seq[0]['generated_text'][1]['content']
                fp.write("[prompt]%s[prompt]\n" % seq[0]['generated_text'][0]['content'])
                fp.write("[answer]%s[answer]\n" % text)
                # contrast_text = text.split(f": ")[-1]
                # if "\n" in contrast_text:
                #     contrast_text = contrast_text.split("\n")[1]
                target_match = re.search(pattern_search, text, re.DOTALL)
                if target_match:
                    contrast_text = target_match.group(1).strip()
                else:
                    contrast_text = text
                    print(text)
                list_contrast_texts.append(contrast_text)
    df['contrast text'] = list_contrast_texts
    df.to_csv(f"snli_{model_name}_{split}_{target_sentence}_{date_string}.csv", index = False)
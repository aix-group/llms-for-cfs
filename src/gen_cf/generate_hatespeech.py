import pandas as pd
from utils.utils_gen import *
import argparse
import re
from sentence_transformers import SentenceTransformer
from datetime import datetime
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
current_date = datetime.now()
date_string = current_date.strftime("%Y%m%d%H%M%S")
def create_prompt(example, closest_instance, text_changes):
    contrast_map = {"hate": "nothate", "nothate": "hate"}
    text_change_split = text_changes.split("\n")
    temp_step_1 = [t.split(" to ")[0] for t in text_change_split]
    step_1 = ",\n".join(temp_step_1)
    step_2 = text_changes
    template = f"""Request: Given a sentence with its original label, indicate whether the sentence contains hate speech (label: hate) or not (label: nothate) in the form of "label: Text". Modify the text with minimal edits to change the original label to the target label. Do not make any unneccesary changes.
        Enclose the final generated text within <new> tags:
        (Original text)
        {closest_instance['label']}: {closest_instance['text']}
        Target: {closest_instance['cf_label']}
        Step 1: Identify phrases, words leading to the {closest_instance['label']} label: 
        {step_1}
        Step 2: Change these phrases, words, sentences to {closest_instance['cf_label']} sentiment with minimal changes: 
        {step_2}
        Step 3: replace the phrases, words, sentences from step 1 in the original text by the phrases, words, sentences in step 2:
        {closest_instance['cf_label']}: <new>{closest_instance['cf_text']}</new>
        #End example#
        Similarly, given a sentence with its original label, indicate whether the sentence contains hate speech (label: hate) or not (label: nothate) in the form of "label: Text". Modify the text with minimal edits to change the original label to the target label. Do not make any unneccesary changes. Enclose the generated text within <new> tags.
        (Original text)
        {example['label']}: {example['text']}
        Target: {closest_instance['cf_label']}
    """
    template_simple = [{
        "role": "user",
        "content": template
    }]
    return template_simple
def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Benchmark arguments")

    # Add positional arguments
    parser.add_argument("-split", required=True, help="Split set that needs to generate the counterfactual ", choices=['dev', 'train', 'test'])
    parser.add_argument("-model", required=True, help="Model to generate counterfactual ")
    # parser.add_argument("-task", required=True, help="Name of the task. Currently, only IMDB and SNLI are supported.", choices=['imdb', 'snli'])
    parser.add_argument("-batch_size", type=int, default=100, help="Batch size for evaluation.")
    # Add optional arguments
    # parser.add_argument("-batch_size", type=int, default=100, help="Batch size for evaluation.")
    # parser.add_argument("-return_csv", action="store_true", help="Whether to save the metrics to the original CSV file.")

    # Parse the command line arguments
    args = parser.parse_args()
    return args
def extract_answer(answer):
    start_edited_pattern = r'(?:\<new\>)(.*?)(?:\<\/new\>)'
    edited_match = re.search(start_edited_pattern, answer, re.DOTALL)
    if edited_match:
        edited_text = edited_match.group(1).strip()
        if len(edited_text.split(": ")) >1:
            edited_text = edited_text.split(": ")[1]
    #     target_match = re.search(pattern_search, edited_text, re.DOTALL)
    #     if target_match:
    #         contrast_text = target_match.group(1).strip()
    #     else:
    #         contrast_text = edited_text
    else:
        edited_text = None
    #     target_match = re.search(pattern_search, answer)
    #     if target_match:
    #         contrast_text = target_match.group(1).strip()
    #     else:
    #         print(answer)
    #         contrast_text = None
    return edited_text
if __name__ == '__main__':
    args = get_args()

    split = args.split
    # llm_model = "meta-llama/Llama-2-7b-chat-hf"
    # llm_model = "tiiuae/falcon-7b-instruct"
    # llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
    # llm_model = "meta-llama/Llama-2-70b-chat-hf"
    llm_model = args.model
    model_name, llm_pipeline, tokenizer = load_model(llm_model)
    # llm_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # llm_model = "EleutherAI/pythia-6.9b"
    sbert_model_name = "all-mpnet-base-v2"
    sbert_model = SentenceTransformer(sbert_model_name)
    dev_pairs = pd.read_csv("processed_data/hatespeech/dev.csv")
    test_pairs = pd.read_csv("processed_data/hatespeech/test.csv")
    train_pairs = pd.read_csv("processed_data/hatespeech/train.csv")

    if split == "dev":
        df = dev_pairs
        df_dev = train_pairs
    
    if split == "train":
        # df_merge = train_pairs.copy()
        df = train_pairs
        df_dev = dev_pairs
    if split == "test":
        df = test_pairs
        df_dev = dev_pairs
    hate_pairs = df_dev[df_dev['label'] == "hate"]
    nothate_pairs = df_dev[df_dev['label'] == "nothate"]

    
    hate_embeddings = sbert_model.encode(hate_pairs['text'].to_list())
    nothate_embeddings = sbert_model.encode(nothate_pairs['text'].to_list())
    list_contrast_labels = []
    list_contrast_texts = []
    list_original_labels = []
    list_original_texts = []
    list_closest_instances = []
    list_prompts = []
    batch_size = args.batch_size
    for i in range(0, df.shape[0], batch_size):
        list_prompts = []
        batch = df.iloc[i:i+batch_size]
        for index, example in batch.iterrows():
            reference_pairs = None
            if example['label'] == "hate":
                index = find_closest_instance(sbert_model.encode(example['text']), hate_embeddings)
                list_contrast_labels.append("nothate")
                reference_pairs = hate_pairs
            else:
                index = find_closest_instance(sbert_model.encode(example['text']), nothate_embeddings)
                list_contrast_labels.append("hate")
                reference_pairs = nothate_pairs
            list_original_texts.append(example['text'])
            list_original_labels.append(example['label'])
            closest_instance = reference_pairs.iloc[int(index)]
            
            original_text = closest_instance['text']
            list_closest_instances.append(original_text)
            contrast_text = closest_instance['cf_text']
            text_changes = compare_text(original_text, contrast_text)
            prompt = create_prompt(example,closest_instance,text_changes)
            if tokenizer is not None:
                prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
            list_prompts.append(prompt)
        if "gpt" in llm_model:
            #query the gpt api to get the counterfactual (input: prompt, output: counterfactual)
            fp = open(f'raw_speech_{model_name}_0125_{split}.txt_{date_string}', 'a')
            for prompt in list_prompts:
                gpt_prompt =[
                        {
                            "role": "user",
                            "content": prompt
                        }
                        ]
                results = delayed_completion(model=llm_model,
                                    messages=prompt,
                                    max_tokens=256)
                raw_text = results.choices[0].message.content
                
                fp.write("[start]%s[end]\n" % raw_text)
                # if len(raw_text.split("hate:")) > 1:    
                #     desired_text = raw_text.split("hate:")[1].strip()
                # else:
                #     print(raw_text)
                #     desired_text = None
                contrast_text = extract_answer(raw_text)
                list_contrast_texts.append(contrast_text)
        else: 
            sequences = llm_pipeline(
                list_prompts,
                do_sample=True,
                top_k=50,
                num_return_sequences=1,
                max_new_tokens=256
            )
            
            with open(f'raw_speech_{model_name}_{split}_{date_string}.txt', 'a') as fp:
                for seq in sequences:
                # print(seq[0]['generated_text']) 
                    text = seq[0]['generated_text'].split("\n\n\n")[0]
                    fp.write("[start]%s[end]\n" % text)
            for seq in sequences:
                text = seq[0]['generated_text']
                # temp_instance = text.split("(Original text)")[2]
                # if (len(temp_instance.split("in step 2:")) < 2):
                #     list_contrast_texts.append(None)
                #     continue
                # text = temp_instance.split("in step 2:")[1]
                # text = text.split("Request:")[0]
                # text = text.split("\n\n")[0]

                text = text.split("[/INST]")[1]
                # contrast_label = contrast_map[batch.iloc[index]["Sentiment"]]
                # contrast_label = text_split[0]
                # if len(text_split) >= 2:
                #     contrast_text = ":".join(text_split[1:])
                # else:
                #     contrast_text = text_split[0]
                # text_= text.split(f"{contrast_label}:")[1]
                # text_= text.split(f"\n")[0]
                contrast_text = extract_answer(text)
                list_contrast_texts.append(contrast_text)
    pd.DataFrame({"original_text": list_original_texts,
                "contrast text": list_contrast_texts,
                "original label": list_original_labels,
                "contrast label": list_contrast_labels,
                "closest_instance": list_closest_instances}).to_csv(f"speech_{model_name}_{split}_{date_string}.csv")
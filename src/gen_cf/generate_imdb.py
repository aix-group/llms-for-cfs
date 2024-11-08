from transformers import AutoTokenizer
import transformers
import torch
import argparse
import pandas as pd
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils.utils_gen import *
import re
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datetime import datetime
current_date = datetime.now()
date_string = current_date.strftime("%Y%m%d%H%M%S")
def create_prompt(example, closest_instance, text_changes):
    
    text_change_split = text_changes.split("\n")
    temp_step_1 = [t.split(" to ")[0] for t in text_change_split]
    step_1 = ",\n".join(temp_step_1)
    step_2 = text_changes
    template = f"""Request: Given a piece of text with the original sentiment in the form of "Sentiment: Text". Change the text with minimal edits to get the target sentiment from the original sentiment. Do not make any unneccesary changes.
        Enclose the generated text within <new> tags.
        (Original text)
        {closest_instance['Original Sentiment']}: {closest_instance['Original Text']}
        Target: {closest_instance['Contrast Sentiment']}
        Step 1: Identify phrases, words leading to the {closest_instance['Original Sentiment']} sentiment: 
        {step_1}
        Step 2: Change these phrases, words, sentences to {closest_instance['Contrast Sentiment']} sentiment with minimal changes: 
        {step_2}
        Step 3: replace the phrases, words, sentences from step 1 in the original text by the phrases, words, sentences in step 2:
        {closest_instance['Contrast Sentiment']}: <new>{closest_instance['Contrast Text']}</new>
        ## End Example ##
        Similarly, given a piece of text with the original sentiment in the form of "Sentiment: Text". Change the text with minimal edits to get the target sentiment from the original sentiment. Do not make any unneccesary changes.
        Enclose the generated text within <new> tags.
        (Original text)
        {example['Sentiment']}: {example['Text']}
        Target: {closest_instance['Contrast Sentiment']}
    """
    template_simple = [{
        "role": "user",
        "content": template
    }]

    return template_simple
def extract_answer(answer):
    start_edited_pattern = r'\<new\>(.*?)(?:\<\/new\>)'
    edited_match = re.search(start_edited_pattern, answer, re.DOTALL)
    if edited_match:
        edited_text = edited_match.group(1).strip()
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
def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Benchmark arguments")

    # Add positional arguments
    parser.add_argument("-split", required=True, help="Split set that needs to generate the counterfactual ", choices=['dev', 'train', 'test'])
    parser.add_argument("-model", required=True, help="Model to generate counterfactual ")
    # parser.add_argument("-task", required=True, help="Name of the task. Currently, only IMDB and SNLI are supported.", choices=['imdb', 'snli'])

    # Add optional arguments
    parser.add_argument("-batch_size", type=int, default=100, help="Batch size for evaluation.")
    # parser.add_argument("-return_csv", action="store_true", help="Whether to save the metrics to the original CSV file.")

    # Parse the command line arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    contrast_map = {"Positive": "Negative", "Negative": "Positive"}
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
    dev_pairs = pd.read_csv("processed_data/IMDB/dev_pairs.csv")
    test_pairs = pd.read_csv("../../counterfactually-augmented-data/sentiment/combined/paired/test_paired.tsv", delimiter="\t")
    train_pairs = pd.read_csv("../../counterfactually-augmented-data/sentiment/combined/paired/train_paired.tsv", delimiter="\t")
    # df_merge = pd.concat([train_pairs, test_pairs])
    #create the dev set from the train set
    train_dev = train_pairs.iloc[0::2]
    contrast_text_list = train_pairs.iloc[1::2]['Text'].to_list()
    contrast_sent_list = train_pairs.iloc[1::2]['Sentiment'].to_list()
    train_dev['Contrast Text'] = contrast_text_list
    train_dev['Contrast Sentiment'] = contrast_sent_list
    train_dev = train_dev.rename(columns={"Text": "Original Text", "Sentiment": "Original Sentiment"})
    # df_merge = dev_pairs.rename(columns={"Original Text": "Text", "Original Sentiment": "Sentiment"})
    if split == "dev":
        df_merge = dev_pairs.rename(columns={"Original Text": "Text", "Original Sentiment": "Sentiment"})
        dev_pairs = train_dev.copy()
    if split == "train":
        # df_merge = train_pairs.copy()
        df_merge = train_pairs.iloc[::2]
    if split == "test":
        df_merge = test_pairs.iloc[::2]
    #load model
    sbert_model = SentenceTransformer(sbert_model_name)
    
    # tokenizer.pad_token_id=tokenizer.eos_token_id

    # llm_pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=llm_model,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    #     tokenizer = tokenizer
    # )

    # passage_embedding = model.encode(list_of_text)

    #split sentiment pairs
    #enable to filter length of example
    # dev_pairs["length_of_orig"] = dev_pairs['Original Text'].apply(lambda x: len(x.split()))
    # dev_pairs = dev_pairs[dev_pairs['length_of_orig'] < 128]
    positive_pairs = dev_pairs[dev_pairs['Original Sentiment'] == "Positive"]
    negative_pairs = dev_pairs[dev_pairs['Original Sentiment'] == "Negative"]

    #generate sentiment embeddings for each sentiments:
    positive_embeddings = sbert_model.encode(positive_pairs['Original Text'].to_list())
    negative_embeddings = sbert_model.encode(negative_pairs['Original Text'].to_list())
    list_contrast_labels = []
    list_contrast_texts = []
    list_original_labels = []
    list_original_texts = []
    list_closest_instances = []
    
    # df_test = test_pairs
    #pick an instance
    # Specify the chunk size
    batch_size = args.batch_size
    start_time = time.time()
    # Calculate the number of chunks
    # num_chunks = len(df_merge) // chunk_size + 1
    # df_merge = df_merge.iloc[:5]
    for i in tqdm(range(0, df_merge.shape[0], batch_size)):
        batch = df_merge.iloc[i:i+batch_size]
        list_prompts = []
        for index, example in batch.iterrows():
            reference_pairs = None
            if example['Sentiment'] == "Positive":
                index = find_closest_instance(sbert_model.encode(example['Text']), positive_embeddings)
                list_contrast_labels.append("Negative")
                reference_pairs = positive_pairs
            else:
                index = find_closest_instance(sbert_model.encode(example['Text']), negative_embeddings)
                list_contrast_labels.append("Positive")
                reference_pairs = negative_pairs
            list_original_texts.append(example['Text'])
            list_original_labels.append(example['Sentiment'])
            closest_instance = reference_pairs.iloc[int(index)]
            
            original_text = closest_instance['Original Text']
            list_closest_instances.append(original_text)
            contrast_text = closest_instance['Contrast Text']
            text_changes = compare_text(original_text, contrast_text)
            prompt = create_prompt(example,closest_instance,text_changes)
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
            list_prompts.append(prompt)
        
        if "gpt" in llm_model:
            #query the gpt api to get the counterfactual (input: prompt, output: counterfactual)
            fp = open(f'raw_imdb_{model_name}_0125_{split}.txt_{date_string}', 'a')
            for prompt in list_prompts:
                gpt_prompt =[
                        {
                            "role": "user",
                            "content": prompt
                        }
                        ]
                results = delayed_completion(model=llm_model,
                                    messages=gpt_prompt,
                                    max_tokens=1000)
                raw_text = results.choices[0].message.content
                
                fp.write("[start]%s[end]\n" % raw_text)
                if len(raw_text.split("tive:")) > 1:    
                    desired_text = raw_text.split("tive:")[1].strip()
                else:
                    print(raw_text)
                    desired_text = None
                list_contrast_texts.append(desired_text)
        else: 
            sequences = llm_pipeline(
                list_prompts,
                do_sample=True,
                top_k=50,
                num_return_sequences=1,
                max_new_tokens=1024
            )
            
            with open(f'raw_imdb_{model_name}_{split}.txt_{date_string}', 'a') as fp:
                for seq in sequences:
                # print(seq[0]['generated_text']) 
                    text = seq[0]['generated_text'].split("\n\n\n")[0]
                    fp.write("[start]%s[end]\n" % text)
            for index,seq in enumerate(sequences):
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
                "closest_instance": list_closest_instances}).to_csv(f"imdb_{model_name}_{split}_{date_string}.csv")
    end_time = time.time()
    duration = end_time - start_time
    print(duration)
    with open(f'duration_imdb_{model_name}_{split}_{date_string}.txt', 'w') as fp:
    
        fp.write("Duration: %s" % str(duration))
    

    # assert len(df_merge) == len(list_contrast_texts) == len(list_original_texts) == len(list_original_labels) == len(list_contrast_labels)
    # assert list_original_labels[0] != list_contrast_labels[0]
    # assert list_original_texts[0] != list_contrast_texts[0]
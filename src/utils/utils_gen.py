import difflib
from sentence_transformers import util
from transformers import AutoTokenizer
import transformers
import torch
import numpy as np
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
def find_closest_instance(example, embeddings):
    """
    select the example (pair of original text and counterfactual text) in the training and dev set that has the closest semantic meaning to the current example
    input: pair [original_label, original_text, cf_label, cf_text]
    output: closest pair
    requirement: need to have the same label
    """
    cos_sim_scores = util.cos_sim(example, embeddings)
    index = np.argmax(cos_sim_scores)
    return index
def delayed_completion(**kwargs):
        from openai import OpenAI
        client = OpenAI()
        # rate_limit_per_minute = 120
        # delay = 60.0 / rate_limit_per_minute
        # time.sleep(delay)
        return client.chat.completions.create(**kwargs)
def load_model(llm_model):
    if "gpt" not in llm_model:
        model_name = llm_model.split("/")[1]
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        tokenizer.pad_token_id=tokenizer.eos_token_id
        llm_pipeline = transformers.pipeline(
            "text-generation",
            model=llm_model,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer = tokenizer
        )
    else:
        if "gpt-4" in llm_model:
            model_name = "gpt-4"
        else:
            model_name = "gpt-35"
        llm_pipeline = None
        tokenizer = None
    return model_name, llm_pipeline, tokenizer
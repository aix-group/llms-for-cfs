import torch
import numpy as np
import random
from pathlib import Path


import nltk 
from spacy.lang.en import English


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_test_sets(task):
    result = list(Path("../llms/").rglob("*.tsv"))
    result += list(Path("../counterfactually-augmented-data/").rglob("*.tsv"))
    result += list(Path("../contrast-sets/IMDb/").rglob("*.tsv"))

    result = [str(x) for x in result]

    test_sets = []
    for s in result:
        if 'eighty_percent' in s:
            continue
        if (s.endswith('test.tsv') and task.lower() in s.lower()) or (s.endswith('test_contrast.tsv') and task.lower() == 'sentiment'):
            name_p1 = s.split('/')[-4].split('-')[0]
            name_p2 = s.split('/')[-2]

            test_sets.append((name_p1 + '_' + name_p2, s))
    return test_sets
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



def score_minimality(orig_sent: str, edited_sent: str, normalized: bool = True) -> float:
        """
          Calculate Levenshtein distance(token-level) indicating the minimality of changes between two sentences.
          This method takes in an original sentence and an edited sentence, both as strings.
          It calculates the Levenshtein edit distance between the tokenized versions of these sentences,
          representing the minimum number of single-token edits needed to transform one into the other.
          Parameters:
          - orig_sent (str): The original sentence before editing.
          - edited_sent (str): The edited version of the sentence.
          - normalized (bool, optional): If True, returns a normalized score relative to the length of
            the original sentence. If False, returns the raw edit distance value.
          Returns:
          - float: The calculated minimality score. If ‘normalized’ is True, the score represents the
            proportion of changes relative to the original sentence length.u
            Source:
          """
        nlp = English()
        tokenizer = nlp.tokenizer
        tokenized_original = [t.text for t in tokenizer(orig_sent)]
        tokenized_edited = [t.text for t in tokenizer(edited_sent)]
        levenshtein_dist = nltk.edit_distance(tokenized_original, tokenized_edited)
        if normalized:
            return levenshtein_dist / len(tokenized_original)
        else:
            return levenshtein_dist

def compute_dist(s1, s2):
    #assert((df[SENT_COLUMN] != df[CF_SENT_COLUMN]).all())
    assert len(s1) == len(s2)
    dist = []

    for x, y in zip(s1,s2):
            dist.append(score_minimality(x, y))
    return dist
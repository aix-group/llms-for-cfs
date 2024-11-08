import os
import re
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score as acc

from datasets import load_dataset, load_from_disk
import torch 
import json 
import argparse
import pandas as pd

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# local imports
sys.path.append('./src/')
from utils.utils import set_seed, get_test_sets



def run(data_origin, task_name, training_split, gpu_index, smoke_test, model_name, seed):
    # Config

    MODEL_NAME = model_name

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    set_seed(seed)

    train_path = '../{}/{}/{}/train.tsv'.format(data_origin, task_name, training_split)

    dev_path = '../{}/{}/{}/dev.tsv'.format(data_origin, task_name, training_split)


    task_name = task_name.lower().strip()
    test_sets = get_test_sets(task_name)
    print(test_sets)

    ## Loading dataset
    data_files = {
        "train": train_path, 
        "dev": dev_path, 
        #"test": orig_test_path, 
        #"test_new": new_test_path, 
        #"test_combined": combined_test_path
    }

    for name, path in test_sets:
        data_files[name] = path

    #data = load_from_disk("csv", data_files=data_files, sep = '\t', ignore_verifications=True)
    data = load_dataset("csv", data_files=data_files, sep = '\t', ignore_verifications=True)        


    LABEL_ENCODER = LabelEncoder()

    if task_name == 'sentiment':
        LABEL_ENCODER.fit(data['dev']['Sentiment'])
    elif task_name == 'nli':
            LABEL_ENCODER.fit(data['dev']['gold_label'])
    elif task_name == 'speech':
        LABEL_ENCODER.fit(data['train']['Label'])
    else:
        raise ValueError
    LABEL_ENCODER.classes_

    ## Preprocessing
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        do_lower_case=False
    )

    def preprocess(row):
        d = {             
        }


        if task_name == 'sentiment':
            d['label'] = LABEL_ENCODER.transform([row['Sentiment']])[0]
            d['input'] = row['Text']

        elif task_name == 'nli':
            d['label'] = LABEL_ENCODER.transform([row['gold_label']])[0]
            d['input1'] = row['sentence1']
            d['input2'] = row['sentence2']

        elif task_name == 'speech':
            d['label'] = LABEL_ENCODER.transform([row['Label']])[0]
            d['input'] = row['Text']
        else:
            raise ValueError
        
        return d

    def tokenize(examples):
        if task_name == 'sentiment' or task_name == 'speech':
            return tokenizer(
                examples['input'],
                truncation=True,
                max_length=512
            )
        elif task_name == 'nli':
                return tokenizer(
                examples['input1'],
                examples['input2'],
                truncation=True,
                max_length=512
            )
        else:
            raise ValueError

    data = data.map(preprocess)
    data = data.map(tokenize, batched=True)


    print('Training data looks like this: \n', tokenizer.decode(data['dev'][0]['input_ids']))

    ## Training
    def get_model():
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL_ENCODER.classes_)
        )
        model.resize_token_embeddings(len(tokenizer))

        ## fix
        #label_to_id = {}
        #for c in LABEL_ENCODER.classes_:
        #    label_to_id[c] = int(LABEL_ENCODER.transform([c])[0])
        #id_to_label = {v: k for k, v in label_to_id.items()}

        #model.config.label2id = label_to_id
        #model.config.id2label = id_to_label
        #print('###')
        #print(model.config)
        #print('###')


        return model.to(device)
    
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        accuracy = acc(labels, preds)
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'acc': accuracy
        }
    
    training_args = TrainingArguments(
        output_dir='../llms-ppl-preds/',
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs = 0,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=512,
        warmup_steps=0,
        weight_decay=0.1,  
        logging_dir = './logs',
        metric_for_best_model='eval_acc',
    )

    trainer = Trainer(model_init= get_model, 
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    trainer.model.to(device)



    # Evaluation on all splits
    def predict(trainer, data, split_name: str, out_path, le):

        preds = trainer.predict(data[split_name])
        logits = torch.tensor(preds.predictions)
        y_pred = np.argmax(preds.predictions, axis=-1)
        y_pred_proba = torch.nn.functional.softmax(logits, dim=1)
        y_true = preds.label_ids

        y_true = le.inverse_transform(y_true)
        y_pred = le.inverse_transform(y_pred)
        # changes label encoding if it does not correspond to model.config
        if task_name == 'speech' and (le.classes_ != list(trainer.model.config.id2label.values())).all():
            print('Changed predictions...')
            y_pred_new = []
            for p in y_pred:
                if p == le.classes_[0]:
                    y_pred_new.append(le.classes_[1])
                elif p == le.classes_[1]:
                    y_pred_new.append(le.classes_[0])
            y_pred = y_pred_new
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)

        dict_results = {
            'accuracy' : report['accuracy'],
            'precision' : report['macro avg']['precision'],
            'recall' : report['macro avg']['recall'],
            'macro-f1' : report['macro avg']['f1-score'],
            'preds' : y_pred
        }


        #with open(out_path / f"y_true_{split_name}.txt", "w") as fout:
        #    for i in y_true:
        #        fout.write(str(i) + "\n")

        #with open(out_path / f"y_pred_{split_name}.txt", "w") as fout:
        #    for i in y_pred:
        #        fout.write(str(i) + "\n")
                
        #with open(out_path / f"y_pred_proba_{split_name}.jsonl", "w") as fout:
        #    for ps in y_pred_proba.tolist():
        #        json.dump(ps, fout)
        #        fout.write('\n')
        return dict_results
        
    # = predict(trainer, data, 'test', OUT_PATH, LABEL_ENCODER)
    #results_new = predict(trainer, data, 'test_new', OUT_PATH, LABEL_ENCODER)
    #results_combined = predict(trainer, data, 'test_combined', OUT_PATH, LABEL_ENCODER)


    for name, path in test_sets:
        res = predict(trainer, data, name, path, LABEL_ENCODER)
        preds = res['preds']
        df = pd.read_csv(path, sep = '\t')
        df['predicted_label'] = preds
        new_path = '/'.join(path.split('/')[:-1]) + '/'
   
        if '/llms/' in new_path:
            new_path = new_path.replace('/llms/', '/llms-ppl-preds/')
        elif '/counterfactually-augmented-data/' in new_path:
            new_path = new_path.replace('/counterfactually-augmented-data/',  '/llms-ppl-preds/counterfactually-augmented-data/')
        elif '/contrast-sets/' in new_path:
            new_path = new_path.replace('/contrast-sets/',  '/llms-ppl-preds/contrast-sets/')
        else:
            raise ValueError

        
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        df.to_csv(new_path +  'test.tsv', sep='\t', index = False)

if __name__ == "__main__":
    # args:
    # language model/human  
    # training set (orig., new, combined)
    # task: sentiment / NLI
    # smoking_test : true/false
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--data_origin", required=True, type=str, help="origin of dataset language model or human generated")
    parser.add_argument("--task", required=True, type=str, help="sentiment/NLI")
    parser.add_argument("--training_split", required=True, type=str, help="original/new/combined")
    parser.add_argument("--gpu", required=True, type=str, help="index")

    # Not required
    parser.add_argument("--smoke_test", required=False, type=bool, help="true/false", default=False)
    parser.add_argument("--model", required=False, type = str, default='distilbert-base-uncased', help="e.g., 'distilbert-base-uncased'")
    parser.add_argument("--seed", required=False, type = int, default=42)


    args=parser.parse_args()
    run(args.data_origin, args.task, args.training_split, args.gpu, args.smoke_test, args.model, args.seed)
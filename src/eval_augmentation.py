import os
import re
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import accuracy_score as acc

from datasets import load_dataset, disable_caching
import torch 
import json 
import argparse
from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import ray
from ray import tune
from ray.tune import CLIReporter

from ray.tune.schedulers import PopulationBasedTraining

# local imports
sys.path.append('./src/')
from utils.utils import set_seed, get_test_sets



def run(data_origin, task_name, training_split, gpu_index, smoke_test, model_name, seed, save_best):
    disable_caching()
    ray.init(ignore_reinit_error=True, num_cpus=4)
    
    # Config
    OUT_PATH = './results/{}/{}/{}/{}/'.format(data_origin, task_name, training_split, seed)
    MODELS_PATH = './best_models/{}/{}/{}/'.format(data_origin, task_name, training_split)

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

    if 'counterfactually' in data_origin:
        ## Loading dataset
        data_files = {
            "train": train_path, 
            "dev": dev_path, 
            #"test": orig_test_path, 
            #"test_new": new_test_path, 
            #"test_combined": combined_test_path
        }
    else:
        # if 'combined' and an LLM is chosen, use only the 'new' split 
        assert ('combined' in training_split.lower())

        if task_name == 'sentiment' or task_name == 'speech':

            trainin_paths = ['../{}/{}/{}/train.tsv'.format(data_origin, task_name, 'new'), 
                            '../{}/{}/{}/train.tsv'.format('llms/counterfactually-augmented-data', task_name, 'orig')]

            dev_paths = ['../{}/{}/{}/dev.tsv'.format(data_origin, task_name, 'new'), 
                        '../{}/{}/{}/dev.tsv'.format('llms/counterfactually-augmented-data', task_name, 'orig')]
        elif task_name.lower() == 'nli':

            trainin_paths = ['../{}/{}/{}/train.tsv'.format(data_origin, task_name.upper(), 'revised_combined'), 
                            '../{}/{}/{}/train.tsv'.format('llms/counterfactually-augmented-data', task_name.upper(), 'original')]

            dev_paths = ['../{}/{}/{}/dev.tsv'.format(data_origin, task_name.upper(), 'revised_combined'), 
                        '../{}/{}/{}/dev.tsv'.format('llms/counterfactually-augmented-data', task_name.upper(), 'original')]
        else:
            raise ValueError
        
        ## Loading dataset
        data_files = {
            "train": trainin_paths, 
            "dev": dev_paths, 
            #"test": orig_test_path, 
            #"test_new": new_test_path, 
            #"test_combined": combined_test_path
        }

    for name, path in test_sets:
        print(name)
        print(path)
        data_files[name] = path

    data = load_dataset("csv", data_files=data_files, sep = '\t', cache_dir=None)
    
    LABEL_ENCODER = LabelEncoder()

    if task_name == 'sentiment':
        LABEL_ENCODER.fit(data['train']['Sentiment'])
    elif task_name == 'nli':
            LABEL_ENCODER.fit(data['train']['gold_label'])
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


    print('Training data looks like this: \n', tokenizer.decode(data['train'][0]['input_ids']))

    ## Training
    def get_model():
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL_ENCODER.classes_)
        )
        model.resize_token_embeddings(len(tokenizer))
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
        output_dir=OUT_PATH,
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs = 2,
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

    tune_config = {
        "per_device_train_batch_size": tune.choice([16, 32, 64]),
        "per_device_eval_batch_size": 64,
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
        "max_steps": 1 if smoke_test else -1,  
        "seed" : seed
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric= "eval_acc", 
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
            #"per_device_train_batch_size": [16, 32, 64],
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=["eval_acc", "eval_f1" "eval_loss", "epoch", "training_iteration"],
    )

    best_run = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config, #  A function that defines the hyperparameter search space.
        backend="ray",
        n_trials=10, 
        resources_per_trial={"cpu": 1, "gpu": 1},
        scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 1} if smoke_test else None,
        progress_reporter=reporter,
        local_dir="./ray_results/{}_{}_{}/".format(data_origin, task_name, training_split), 
        name="tune_transformer_pbt",
        log_to_file=True,   
        direction="maximize",
    )

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)
    trainer.train()

    # Evaluation on all splits
    def predict(trainer, data, split_name: str, out_path, le):
        out_path = Path(out_path)
        preds = trainer.predict(data[split_name])
        logits = torch.tensor(preds.predictions)
        y_pred = np.argmax(preds.predictions, axis=-1)
        y_pred_proba = torch.nn.functional.softmax(logits, dim=1)
        y_true = preds.label_ids

        y_true = le.inverse_transform(y_true)
        y_pred = le.inverse_transform(y_pred)

        report = classification_report(y_true, y_pred, digits=4, output_dict=True)

        dict_results = {
            'accuracy' : report['accuracy'],
            'precision' : report['macro avg']['precision'],
            'recall' : report['macro avg']['recall'],
            'macro-f1' : report['macro avg']['f1-score']
        }


        with open(out_path / f"y_true_{split_name}.txt", "w") as fout:
            for i in y_true:
                fout.write(str(i) + "\n")

        with open(out_path / f"y_pred_{split_name}.txt", "w") as fout:
            for i in y_pred:
                fout.write(str(i) + "\n")
                
        with open(out_path / f"y_pred_proba_{split_name}.jsonl", "w") as fout:
            for ps in y_pred_proba.tolist():
                json.dump(ps, fout)
                fout.write('\n')
        return dict_results
        
    # = predict(trainer, data, 'test', OUT_PATH, LABEL_ENCODER)
    #results_new = predict(trainer, data, 'test_new', OUT_PATH, LABEL_ENCODER)
    #results_combined = predict(trainer, data, 'test_combined', OUT_PATH, LABEL_ENCODER)
    if smoke_test:
        return

    results = []
    for name, _ in test_sets:
        res = predict(trainer, data, name, OUT_PATH, LABEL_ENCODER)
        results.append((name, res))

    with open('../main_results.csv', 'a') as f:
        for name, res in results: #[('orig.', results_orig), ('cfs.', results_new), ('combined', results_combined)]:
            # data origin, task name, training split, #training instances, test split,  #test instances, model name, accuracy, macro-f1, precision, recall, test, seed
            f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(data_origin, task_name,  training_split, str(len(data['train'])), name, str(len(data[name])), MODEL_NAME, res['accuracy'], res['macro-f1'], res['precision'], res['recall'], smoke_test, str(seed)))    

    ## Save Model 
    if save_best and not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    if save_best:
        trainer.save_model(MODELS_PATH)

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
    parser.add_argument("--model", required=False, type = str, default='bert-base-uncased', help="e.g., 'bert-base-uncased'")
    parser.add_argument("--seed", required=False, type = int, default=42)
    parser.add_argument("--save_best", required=False, type=bool, help="true/false", default=False)


    args=parser.parse_args()
    run(args.data_origin, args.task, args.training_split, args.gpu, args.smoke_test, args.model, args.seed, args.save_best)
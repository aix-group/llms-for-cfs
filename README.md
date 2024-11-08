# LLMs for Generating and Evaluating Counterfactuals: A Comprehensive Study

## Code
Make sure to use --recurse-submodules flag to clone the submodules as well.
```sh
git clone --recurse-submodules 
```


## Environment
We use the [Mamba](https://github.com/mamba-org/mamba) package manager.


```sh
mamba env create -f cfg.yml
conda activate cfg
```


## Where is what?
* The generated CFs can be found under  [./llms/](./llms/). CFs from GPT3.5/4 will be published upon approval from OpenAI (Credits for the experiments were obtained through OpenAI API Researcher Access Program). 
* GPT4 evaluation scores can be found under [./llm-eval-gpt4/](./llm-eval-gpt4/)
* To run the data augmentation results use [./src/eval_augmentation.py](./src/eval_augmentation.py) (e.g., `python eval_augmentation.py --data_origin llms/gpt3.5-20240313 --task sentiment --training_split combined --gpu 0 --model bert-base-uncased --seed 0`)
* To get the predictions using a finetuned classifier use [./src/add_preds.py](./src/eval_augmentation.py) (e.g., `python add_preds.py --data_origin llms/llama2-20231209/ --task sentiment --training_split combined --gpu 0 --model textattack/bert-base-uncased-imdb`)
* To add perplexity use [./notebooks/add_ppl.ipynb](./notebooks/add_ppl.ipynb)
* To add distance use the scripts under [./notebooks/](./notebooks/) that start with `dist`
* The CFs generation process can be found under [./src/gen_cf](./src/gen_cf)
 
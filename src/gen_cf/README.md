### Setup:
- Install necessary packages by running:
  ```bash
  pip install -r requirements.txt
  ```
### Run counterfactual generation:
Navigate to the current folder `src/gen_cf`
 - **IMDB:**
 ```bash
python generate_imdb.py -split $SPLIT -model $MODEL -batch_size $SIZE
```

 Where:
  - `SPLIT`: can be either "dev" , "test, "train".
  - `MODEL`: Huggingface or GPT Model
  - `SIZE`: Batch Size

  For Example:


  ```bash
  python generate_imdb.py -split dev -model meta-llama/Llama-2-7b-chat-hf -batch_size 100
  ```

  
 - **SNLI:**
```bash
python generate_snli.py -split $SPLIT -model $MODEL -target_sentence $TARGET
```
  Where:
  - `SPLIT`: can be either "dev" , "test, "train".
  - `MODEL`: Huggingface or GPT Model
  - `TARGET`: either "sentence1" or "sentence2"

  For Example:
 ```bash
python generate_snli.py -split dev -model meta-llama/Llama-2-7b-chat-hf -target_sentence sentence1
```

 - **Hatespeech:**
 ```bash
python generate_hatespeech.py -split $SPLIT -model $MODEL -batch_size $SIZE
```

 Where:
  - `SPLIT`: can be either "dev" , "test, "train".
  - `MODEL`: Huggingface or GPT Model
  - `SIZE`: Batch Size

  For Example:


  ```bash
  python generate_hatespeech.py -split dev -model meta-llama/Llama-2-7b-chat-hf -batch_size 100
  ```


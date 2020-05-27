# Data-Rejuvenation
Implementation of Data Rejuvenation pipeline.

## Code Base
This implementation is based on [fairseq](https://github.com/pytorch/fairseq/tree/v0.9.0/fairseq) v0.9.0, with customized modification of scripts.

To start, you need to clone this repo and install `fairseq` firstly. Use the following pip command in `fairseq/`:
```
pip install --editable .
```

Additional Functionalities:
- Transformer based LSTM;
- Force decoding: `force_decode.py`;
- Identification: `identify_split.py`;

## Pipline

### Identification
1. Create four folders in `fairseq/`.
```
mkdir dataset
mkdir data-bin
mkdir checkpoints
mkdir results
```

2. Train an identification NMT model and obtain the token-wise prediction probability.
  - Train the NMT model: run `sh sh_train.sh`.
    - Check the output in `fairseq/checkpoints/wmt14_en_de_base_untied_fp16`:
      ```
      checkpoint_best.pt
      ```
  - Force-decode: run `sh sh_forcedecode.sh`. 
    - Check the output in `fairseq/results/wmt14_en_de_base_untied_fp16/sample_status/`:
      ```
      status_train_[BestStep].txt
      ```
  
3. Compute the sentence probability and split inactive examples and active examples.
  - Identify and split: run `python identify_split.py`.
    - Check the output in `fairseq/dataset/wmt14_en_de_base_untied_identified`;
      ```
      inactive.en
      inactive.de
      active.en
      active.de
      ```
    
### Rejuvenation
1. Train a rejuvenation NMT model and generate over the inactive samples.
  - Train the NMT model as normal but on the active examples: run `sh sh_train.sh`.
    - Check the output in `fairseq/checkpoints/wmt14_en_de_base_untied_active`:
    ```
    checkpoint_best.pt
    ```
  - Generate over the inactive examples without the use of `--remove-bpe`;
    - Check the output in `fairseq/results/wmt14_en_de_base_untied_active/inactive`:
      ```
      source.txt
      target.txt
      decoding.txt
      ```
    

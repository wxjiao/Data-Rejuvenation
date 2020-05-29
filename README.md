# Data-Rejuvenation
Implementation of Data Rejuvenation pipeline.

## Code Base
This implementation is based on [fairseq](https://github.com/pytorch/fairseq/tree/v0.9.0/fairseq)(v0.9.0), with customized modification of scripts.

To start, you need to clone this repo and install `fairseq` firstly. Use the following pip command in `fairseq/`:
```
pip install --editable .
```

Additional Functionalities:
- Transformer-based LSTM;
- Force decoding: `force_decode.py`;
- Identification: `identify_split.py`;

## Pipline
Take the Transformer-Base model and WMT14 En-De dataset as an example.

### Identification
1. Create four folders in `fairseq/`.
   ```
   mkdir dataset
   mkdir data-bin
   mkdir checkpoints
   mkdir results
   ```

2. Train an identification NMT model and obtain the token-wise prediction probability.
   - Train the NMT model on full training data of WMT14 En-De:
     ```
     sh sh_train.sh
     ```
   - Check the best model:
     ```
     fairseq/checkpoints/wmt14_en_de_base/checkpoint_best.pt
     ```
   - Force-decode the full training data: 
     ```
     sh sh_forcedecode.sh
     ```
   - Check the token-wise probability:
     ```
     fairseq/results/wmt14_en_de_base/sample_status/status_train_[BestStep].txt
     ```
  
3. Compute the sentence-level probability and split inactive examples and active examples.
   - Identify and split:
     ```
     python identify_split.py
     ```
   - Check the inactive examples:
     ```
     fairseq/dataset/wmt14_en_de_base_identified/inactive.en
     fairseq/dataset/wmt14_en_de_base_identified/inactive.de
     fairseq/dataset/wmt14_en_de_base_identified/active.en
     fairseq/dataset/wmt14_en_de_base_identified/active.de
     ```
    
### Rejuvenation
1. Train a rejuvenation NMT model and generate over the inactive samples.
   - Train the NMT model as normal but on the active examples: 
     ```
     sh sh_train.sh
     ```
   - Check the best model:
     ```
     fairseq/checkpoints/wmt14_en_de_base_active/checkpoint_best.pt
     ```
   - Generate over the inactive examples (w/o `--remove-bpe`):
     ```
     sh sh_generate_extra.sh
     ```
   - Check the rejuvenated examples:
     ```
     fairseq/results/wmt14_en_de_base_active/inactive/source.txt
     fairseq/results/wmt14_en_de_base_active/inactive/target.txt
     fairseq/results/wmt14_en_de_base_active/inactive/decoding.txt
     ```
    

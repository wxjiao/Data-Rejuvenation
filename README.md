# Data-Rejuvenation
Implementation of our paper "Data Rejuvenation: Exploiting Inactive Training Examples for Neural Machine Translation" to appear in EMNLP2020.

## Brief Introduction
Large-scale training datasets lie at the core of the recent success of neural machine translation (NMT) models. However, the complex patterns and potential noises in the large-scale data make training NMT models difficult. In this work, we explore to identify the inactive training examples which contribute less to the model performance, and show that the existence of inactive examples depends on the data distribution. We further introduce **data rejuvenation** to improve the training of NMT models on large-scale datasets by exploiting inactive examples. The proposed framework consists of three phases. 
First, we train an *identification model* on the original training data, and use it to distinguish inactive examples and active examples by their sentence-level output probabilities.
Then, we train a *rejuvenation model* on the active examples, which is used to re-label the inactive examples with forward-translation. Finally, the rejuvenated examples and the active examples are combined to train the final NMT model. Experimental results on WMT14 English-German and English-French datasets show that the proposed *data rejuvenation* consistently and significantly improves performance for several strong NMT models. Extensive analyses reveal that our approach stabilizes and accelerates the training process of NMT models, resulting in final models with better generalization capability.

![The framework of Data Rejuvenation](/image/DataReju.png) <!-- .element width="60%" -->

## Code Base
This implementation is based on [fairseq(v0.9.0)](https://github.com/pytorch/fairseq/tree/v0.9.0/fairseq), with customized modification of scripts.

To start, you need to clone this repo and install `fairseq` firstly. Use the following pip command in `fairseq/`:
```
pip install --editable .
```

Additional Functionalities:
- Transformer-based LSTM;
- Force decoding: `force_decode.py`;
- Identification: `identify_split.py`;

## Pipeline
Take the Transformer-Base model and WMT14 En-De dataset as an example.

### Identification
1. Create four folders in `fairseq/`.
   ```
   mkdir dataset
   mkdir data-bin
   mkdir checkpoints
   mkdir results
   ```
   These four folders are used as below:
   - `fairseq/dataset/`: Save raw dataset with BPE.
     ```
     wmt14_en_de_base/train.en
     wmt14_en_de_base/train.de
     wmt14_en_de_base/valid.en
     wmt14_en_de_base/valid.de
     wmt14_en_de_base/test.en
     wmt14_en_de_base/test.de
     ```
   - `fairseq/data-bin/`: Save the binarized data after pre-processing.
   - `fairseq/checkpoints/`: Save the checkpoints of models during training.
   - `fairseq/results/`: Save the output results, including training log, inference output, token-wise probability, etc.

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
  
3. Compute the sentence-level probability and split **_inactive_** examples and **_active_** examples.
   - Identify and split:
     ```
     python identify_split.py
     ```
   - Check the **_inactive_** examples:
     ```
     fairseq/dataset/wmt14_en_de_base_identified/inactive.en
     fairseq/dataset/wmt14_en_de_base_identified/inactive.de
     fairseq/dataset/wmt14_en_de_base_identified/active.en
     fairseq/dataset/wmt14_en_de_base_identified/active.de
     ```
    
### Rejuvenation
1. Train a rejuvenation NMT model and generate over the **_inactive_** samples.
   - Train the NMT model as normal but on the **_active_** examples: 
     ```
     sh sh_train.sh
     ```
   - Check the best model:
     ```
     fairseq/checkpoints/wmt14_en_de_base_active/checkpoint_best.pt
     ```
   - Generate over the **_inactive_** examples (w/o `--remove-bpe`):
     ```
     sh sh_generate_extra.sh
     ```
   - Check the **_rejuvenated_** examples:
     ```
     fairseq/results/wmt14_en_de_base_active/inactive/source.txt
     fairseq/results/wmt14_en_de_base_active/inactive/target.txt
     fairseq/results/wmt14_en_de_base_active/inactive/decoding.txt
     ```
    
\**Note**: A strong identification NMT models can take over the job of the rejuvenation NMT model, thus reducing the effort for training a new model. For example, the large-batch configured Transformer-Big and Dynamic-Conv models.
    
### Final NMT Model
1. Train a final NMT model from scratch.
   - Train the NMT model on the combination of **_active_** examples and **_rejuvenated_** examples: 
     ```
     sh sh_train.sh
     ```
   - Check the best model:
     ```
     fairseq/checkpoints/wmt14_en_de_base_rejuvenated/checkpoint_best.pt
     ```
     

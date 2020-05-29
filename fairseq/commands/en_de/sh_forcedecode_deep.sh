#!/usr/bin/env bash
DATA=wmt14_en_de_deep_untied_fp16
cd ../../
DISK=/apdcephfs/share_916081/joelwxjiao/checkpoints
CHECKPOINT_DIR=$DISK/$DATA
EVAL_OUTPUT_PATH=./results/$DATA/evaluation/
CHECKPOINT=checkpoint_best.pt
CHECKFILE=$CHECKPOINT_DIR/$CHECKPOINT


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python force_decode.py data-bin/$DATA \
  --fp16 \
  -s en -t de \
  --encoder-layers 20 \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --lr 0.002 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 \
  --attention-dropout 0.1 --dropout 0.1 \
  --max-tokens 32768 \
  --update-freq 1 \
  --arch transformer \
  --optimizer adam --adam-betas '(0.9, 0.997)' \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 16000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir ./results/$DATA/logs \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --no-progress-bar \
  --log-format simple \
  --log-interval 100 \
  --save-interval-updates 1000 \
  --max-update 50000 \
  --max-epoch 100 \
  --keep-interval-updates 1 \
  --keep-last-epochs 1 \
  --beam 1 \
  --remove-bpe \
  --results-path ./results/$DATA \
  --restore-file $CHECKFILE \
  --valid-subset 'train' \
  --skip-invalid-size-inputs-valid-test \
  --no-load-trainer-data \
  --no-bleu-eval \
  --quiet \
  --all-gather-list-size 4800000 \
  --num-ref $DATA=1 \
  --valid-decoding-path $EVAL_OUTPUT_PATH \
  --multi-bleu-path ./scripts/ \
 # |& tee ./results/$DATA/logs/train.log
 #--share-decoder-input-output-embed \ 
 # --save-interval 1 \
 # --keep-interval-updates 5 \
 # --keep-last-epochs 5 \

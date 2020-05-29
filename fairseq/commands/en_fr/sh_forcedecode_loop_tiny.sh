#!/usr/bin/env bash
DATA=wmt14_en_de_tiny_untied_fp16
cd ../../
DISK=/apdcephfs/share_916081/joelwxjiao
CHECKPOINT_DIR=$DISK/checkpoints/$DATA
EVAL_OUTPUT_PATH=$DISK/results/$DATA/evaluation/


echo "Access the example probs of $DATA"
for ckp in {20000..100000..2000}
do
  CHECKFILE=$CHECKPOINT_DIR/*_$ckp.pt
  echo "---------- start CHECKPOINT $ckp -----------"

  CUDA_VISIBLE_DEVICES=0,1,2,3 python force_decode.py data-bin/$DATA \
  --fp16 \
  -s en -t de \
  --lr 0.0007 --min-lr 1e-09 \
  --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 \
  --max-tokens 49152 \
  --update-freq 1 \
  --arch transformer_tiny \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 \
  --warmup-updates 4000 \
  --save-dir $CHECKPOINT_DIR \
  --tensorboard-logdir $DISK/results/$DATA/logs \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --no-progress-bar \
  --log-format simple \
  --log-interval 100 \
  --save-interval-updates 2000 \
  --max-update 100000 \
  --max-epoch 100 \
  --beam 1 \
  --remove-bpe \
  --results-path $DISK/results/$DATA \
  --restore-file $CHECKFILE \
  --valid-subset 'train' \
  --skip-invalid-size-inputs-valid-test \
  --no-load-trainer-data \
  --no-bleu-eval \
  --quiet \
  --all-gather-list-size 3600000 \
  --num-ref $DATA=1 \
  --valid-decoding-path $EVAL_OUTPUT_PATH \
  --multi-bleu-path ./scripts/
  echo "---------- Done CHECKPOINT $ckp ----------"
done
 # |& tee ./results/$DATA/logs/train.log
 #--share-decoder-input-output-embed \ 
 # --save-interval 1 \
 # --keep-interval-updates 5 \
 # --keep-last-epochs 5 \

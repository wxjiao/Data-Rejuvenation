WORK_SPACE=/data0/ywang/fairseq-12-24/data-bin/iwslt14.tokenized.de-en

CHECKPOINT=/data0/ywang/fairseq-12-24/checkpoints

if [ ! -d $CHECKPOINT/valid_test_txt ]; then
  mkdir -p $CHECKPOINT/valid_test_txt
  cp -r $WORK_SPACE/valid_test_txt $CHECKPOINT
fi

CUDA_VISIBLE_DEVICES=0,7 python train.py $WORK_SPACE \
    --task translation -s de -t en \
    --restore-file checkpoint_last.pt \
    --max-tokens 2048 --max-sentences-valid 256 \
    --lr 5e-4 --min-lr '1e-09' --clip-norm 0.0 --update-freq 1 --max-update 100000 \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed --dropout 0.3 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.0001 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --save-dir $CHECKPOINT \
    --save-interval-updates 1000 --save-interval 1 --validate-interval 1 \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --quiet --beam 1 --max-len-a 2 --max-len-b 0 --all-gather-list-size 522240 --tensorboard-logdir $CHECKPOINT \
    --num-ref iwslt=1 \
    --valid-decoding-path $CHECKPOINT/valid_test_txt \
    --multi-bleu-path /data0/ywang/fairseq-12-24/scripts --remove-bpe
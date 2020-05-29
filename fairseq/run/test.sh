WORK_SPACE=/data0/ywang/fairseq-12-24/data-bin/iwslt14.tokenized.de-en

CHECKPOINT=/data0/ywang/fairseq-12-24/checkpoints

datasets=("valid" "test")
for dataset in ${datasets[@]}
do
    DECODING=/data0/ywang/fairseq-12-24/checkpoints/decoding/$dataset

    echo $DECODING
    if [ ! -d $DECODING ]; then
      mkdir -p $DECODING
    fi

    CUDA_VISIBLE_DEVICES=7 python generate.py $WORK_SPACE \
      --task translation -s de -t en \
      --gen-subset $dataset \
      --path $CHECKPOINT/checkpoint_best.pt \
      --batch-size 256 \
      --quiet --beam 4 --max-len-a 0 --max-len-b 200 --remove-bpe \
      --decoding-path $DECODING \
      --num-ref iwslt=1 \
      --valid-decoding-path $WORK_SPACE/valid_test_txt \
      --multi-bleu-path /data0/ywang/fairseq-12-24/scripts
done

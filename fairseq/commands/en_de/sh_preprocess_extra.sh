DATA=wmt14_en_de_transfer_BaseBaseBase_trusted
TEXT=dataset/$DATA
# Preprocess
cd ../../
python preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref $TEXT/train_BaseBaseBase/train \
  --destdir data-bin/$DATA/train_BaseBaseBase \
  --workers 32 \
  --srcdict data-bin/$DATA/dict.en.txt \
  --tgtdict data-bin/$DATA/dict.de.txt \


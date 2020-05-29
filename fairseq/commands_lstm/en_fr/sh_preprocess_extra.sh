DATA=wmt14_en_de_transfer_TinyBig
TEXT=dataset/$DATA
# Preprocess
cd ../../
python preprocess.py \
  --source-lang en \
  --target-lang de \
  --validpref $TEXT/valid_probRepLev10 \
  --destdir data-bin/$DATA \
  --workers 32 \
  --srcdict data-bin/$DATA/dict.en.txt \
  --tgtdict data-bin/$DATA/dict.de.txt \


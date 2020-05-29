DATA=wmt14_en_fr_transfer_LightLight
TEXT=dataset/$DATA
# Preprocess
cd ../../
python preprocess.py \
  --source-lang en \
  --target-lang fr \
  --validpref $TEXT/valid_probRepLev10 \
  --destdir data-bin/$DATA/valid_probRepLev10 \
  --workers 32 \
  --srcdict data-bin/$DATA/dict.en.txt \
  --tgtdict data-bin/$DATA/dict.fr.txt \


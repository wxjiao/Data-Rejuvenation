DATA=wmt14_en_de_lightconv
TEXT=dataset/$DATA
# Preprocess
cd ../../
python preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test \
  --destdir data-bin/$DATA \
  --workers 32 \
  --joined-dictionary \
#  --srcdict data-bin/$DATA/dict.en.txt \
#  --tgtdict data-bin/$DATA/dict.de.txt \


DATA=wmt14_en_fr_base_untied_probdiff01rm
TEXT=dataset/$DATA
# Preprocess
cd ../../
python preprocess.py \
  --source-lang en \
  --target-lang fr \
  --validpref $TEXT/valid_probRepLev10 \
  --destdir data-bin/$DATA \
  --workers 32 \
  --srcdict data-bin/$DATA/dict.en.txt \
  --tgtdict data-bin/$DATA/dict.fr.txt \


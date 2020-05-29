# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
python preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
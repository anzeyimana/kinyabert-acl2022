
fairseq-preprocess \
    --only-source \
    --trainpref "datasets/RW_NEWS/parsed/train_input0_bpe.txt" \
    --validpref "datasets/RW_NEWS/parsed/dev_input0_bpe.txt" \
    --testpref "datasets/RW_NEWS/parsed/test_input0_bpe.txt" \
    --destdir "datasets/RW_NEWS/bpe/input0" \
    --workers 8 \
    --srcdict datasets/RW_NEWS/bpe/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/RW_NEWS/plain/train.label.txt" \
    --validpref "datasets/RW_NEWS/plain/dev.label.txt" \
    --destdir "datasets/RW_NEWS/bpe/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/RW_NEWS/parsed/train_input0_spm.txt" \
    --validpref "datasets/RW_NEWS/parsed/dev_input0_spm.txt" \
    --testpref "datasets/RW_NEWS/parsed/test_input0_spm.txt" \
    --destdir "datasets/RW_NEWS/spm/input0" \
    --workers 8 \
    --srcdict datasets/RW_NEWS/spm/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/RW_NEWS/plain/train.label.txt" \
    --validpref "datasets/RW_NEWS/plain/dev.label.txt" \
    --destdir "datasets/RW_NEWS/spm/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/RW_NEWS/parsed/train_input0_morpho.txt" \
    --validpref "datasets/RW_NEWS/parsed/dev_input0_morpho.txt" \
    --testpref "datasets/RW_NEWS/parsed/test_input0_morpho.txt" \
    --destdir "datasets/RW_NEWS/morpho/input0" \
    --workers 8 \
    --srcdict datasets/RW_NEWS/morpho/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/RW_NEWS/plain/train.label.txt" \
    --validpref "datasets/RW_NEWS/plain/dev.label.txt" \
    --destdir "datasets/RW_NEWS/morpho/label" \
    --workers 8

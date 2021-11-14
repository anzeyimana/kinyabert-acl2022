
fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_SST-2/parsed/train_input0_bpe.txt" \
    --validpref "datasets/KIN_SST-2/parsed/dev_input0_bpe.txt" \
    --testpref "datasets/KIN_SST-2/parsed/test_input0_bpe.txt" \
    --destdir "datasets/KIN_SST-2/bpe/input0" \
    --workers 8 \
    --srcdict datasets/KIN_SST-2/bpe/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_SST-2/to_translate/train_label.txt" \
    --validpref "datasets/KIN_SST-2/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_SST-2/bpe/label" \
    --workers 8


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_SST-2/parsed/train_input0_spm.txt" \
    --validpref "datasets/KIN_SST-2/parsed/dev_input0_spm.txt" \
    --testpref "datasets/KIN_SST-2/parsed/test_input0_spm.txt" \
    --destdir "datasets/KIN_SST-2/spm/input0" \
    --workers 8 \
    --srcdict datasets/KIN_SST-2/spm/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_SST-2/to_translate/train_label.txt" \
    --validpref "datasets/KIN_SST-2/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_SST-2/spm/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_SST-2/parsed/train_input0_morpho.txt" \
    --validpref "datasets/KIN_SST-2/parsed/dev_input0_morpho.txt" \
    --testpref "datasets/KIN_SST-2/parsed/test_input0_morpho.txt" \
    --destdir "datasets/KIN_SST-2/morpho/input0" \
    --workers 8 \
    --srcdict datasets/KIN_SST-2/morpho/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_SST-2/to_translate/train_label.txt" \
    --validpref "datasets/KIN_SST-2/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_SST-2/morpho/label" \
    --workers 8

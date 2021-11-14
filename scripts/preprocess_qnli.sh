
fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_QNLI/parsed/train_input0_bpe.txt" \
    --validpref "datasets/KIN_QNLI/parsed/dev_input0_bpe.txt" \
    --testpref "datasets/KIN_QNLI/parsed/test_input0_bpe.txt" \
    --destdir "datasets/KIN_QNLI/bpe/input0" \
    --workers 8 \
    --srcdict datasets/KIN_QNLI/bpe/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_QNLI/parsed/train_input1_bpe.txt" \
    --validpref "datasets/KIN_QNLI/parsed/dev_input1_bpe.txt" \
    --testpref "datasets/KIN_QNLI/parsed/test_input1_bpe.txt" \
    --destdir "datasets/KIN_QNLI/bpe/input1" \
    --workers 8 \
    --srcdict datasets/KIN_QNLI/bpe/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_QNLI/to_translate/train_label.txt" \
    --validpref "datasets/KIN_QNLI/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_QNLI/bpe/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_QNLI/parsed/train_input0_spm.txt" \
    --validpref "datasets/KIN_QNLI/parsed/dev_input0_spm.txt" \
    --testpref "datasets/KIN_QNLI/parsed/test_input0_spm.txt" \
    --destdir "datasets/KIN_QNLI/spm/input0" \
    --workers 8 \
    --srcdict datasets/KIN_QNLI/spm/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_QNLI/parsed/train_input1_spm.txt" \
    --validpref "datasets/KIN_QNLI/parsed/dev_input1_spm.txt" \
    --testpref "datasets/KIN_QNLI/parsed/test_input1_spm.txt" \
    --destdir "datasets/KIN_QNLI/spm/input1" \
    --workers 8 \
    --srcdict datasets/KIN_QNLI/spm/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_QNLI/to_translate/train_label.txt" \
    --validpref "datasets/KIN_QNLI/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_QNLI/spm/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_QNLI/parsed/train_input0_morpho.txt" \
    --validpref "datasets/KIN_QNLI/parsed/dev_input0_morpho.txt" \
    --testpref "datasets/KIN_QNLI/parsed/test_input0_morpho.txt" \
    --destdir "datasets/KIN_QNLI/morpho/input0" \
    --workers 8 \
    --srcdict datasets/KIN_QNLI/morpho/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_QNLI/parsed/train_input1_morpho.txt" \
    --validpref "datasets/KIN_QNLI/parsed/dev_input1_morpho.txt" \
    --testpref "datasets/KIN_QNLI/parsed/test_input1_morpho.txt" \
    --destdir "datasets/KIN_QNLI/morpho/input1" \
    --workers 8 \
    --srcdict datasets/KIN_QNLI/morpho/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_QNLI/to_translate/train_label.txt" \
    --validpref "datasets/KIN_QNLI/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_QNLI/morpho/label" \
    --workers 8


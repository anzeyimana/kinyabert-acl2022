
fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_MRPC/parsed/train_input0_bpe.txt" \
    --validpref "datasets/KIN_MRPC/parsed/dev_input0_bpe.txt" \
    --testpref "datasets/KIN_MRPC/parsed/test_input0_bpe.txt" \
    --destdir "datasets/KIN_MRPC/bpe/input0" \
    --workers 8 \
    --srcdict datasets/KIN_MRPC/bpe/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_MRPC/parsed/train_input1_bpe.txt" \
    --validpref "datasets/KIN_MRPC/parsed/dev_input1_bpe.txt" \
    --testpref "datasets/KIN_MRPC/parsed/test_input1_bpe.txt" \
    --destdir "datasets/KIN_MRPC/bpe/input1" \
    --workers 8 \
    --srcdict datasets/KIN_MRPC/bpe/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_MRPC/to_translate/train_label.txt" \
    --validpref "datasets/KIN_MRPC/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_MRPC/bpe/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_MRPC/parsed/train_input0_spm.txt" \
    --validpref "datasets/KIN_MRPC/parsed/dev_input0_spm.txt" \
    --testpref "datasets/KIN_MRPC/parsed/test_input0_spm.txt" \
    --destdir "datasets/KIN_MRPC/spm/input0" \
    --workers 8 \
    --srcdict datasets/KIN_MRPC/spm/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_MRPC/parsed/train_input1_spm.txt" \
    --validpref "datasets/KIN_MRPC/parsed/dev_input1_spm.txt" \
    --testpref "datasets/KIN_MRPC/parsed/test_input1_spm.txt" \
    --destdir "datasets/KIN_MRPC/spm/input1" \
    --workers 8 \
    --srcdict datasets/KIN_MRPC/spm/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_MRPC/to_translate/train_label.txt" \
    --validpref "datasets/KIN_MRPC/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_MRPC/spm/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_MRPC/parsed/train_input0_morpho.txt" \
    --validpref "datasets/KIN_MRPC/parsed/dev_input0_morpho.txt" \
    --testpref "datasets/KIN_MRPC/parsed/test_input0_morpho.txt" \
    --destdir "datasets/KIN_MRPC/morpho/input0" \
    --workers 8 \
    --srcdict datasets/KIN_MRPC/morpho/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_MRPC/parsed/train_input1_morpho.txt" \
    --validpref "datasets/KIN_MRPC/parsed/dev_input1_morpho.txt" \
    --testpref "datasets/KIN_MRPC/parsed/test_input1_morpho.txt" \
    --destdir "datasets/KIN_MRPC/morpho/input1" \
    --workers 8 \
    --srcdict datasets/KIN_MRPC/morpho/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_MRPC/to_translate/train_label.txt" \
    --validpref "datasets/KIN_MRPC/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_MRPC/morpho/label" \
    --workers 8

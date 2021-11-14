
fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_RTE/parsed/train_input0_bpe.txt" \
    --validpref "datasets/KIN_RTE/parsed/dev_input0_bpe.txt" \
    --testpref "datasets/KIN_RTE/parsed/test_input0_bpe.txt" \
    --destdir "datasets/KIN_RTE/bpe/input0" \
    --workers 8 \
    --srcdict datasets/KIN_RTE/bpe/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_RTE/parsed/train_input1_bpe.txt" \
    --validpref "datasets/KIN_RTE/parsed/dev_input1_bpe.txt" \
    --testpref "datasets/KIN_RTE/parsed/test_input1_bpe.txt" \
    --destdir "datasets/KIN_RTE/bpe/input1" \
    --workers 8 \
    --srcdict datasets/KIN_RTE/bpe/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_RTE/to_translate/train_label.txt" \
    --validpref "datasets/KIN_RTE/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_RTE/bpe/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_RTE/parsed/train_input0_spm.txt" \
    --validpref "datasets/KIN_RTE/parsed/dev_input0_spm.txt" \
    --testpref "datasets/KIN_RTE/parsed/test_input0_spm.txt" \
    --destdir "datasets/KIN_RTE/spm/input0" \
    --workers 8 \
    --srcdict datasets/KIN_RTE/spm/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_RTE/parsed/train_input1_spm.txt" \
    --validpref "datasets/KIN_RTE/parsed/dev_input1_spm.txt" \
    --testpref "datasets/KIN_RTE/parsed/test_input1_spm.txt" \
    --destdir "datasets/KIN_RTE/spm/input1" \
    --workers 8 \
    --srcdict datasets/KIN_RTE/spm/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_RTE/to_translate/train_label.txt" \
    --validpref "datasets/KIN_RTE/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_RTE/spm/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_RTE/parsed/train_input0_morpho.txt" \
    --validpref "datasets/KIN_RTE/parsed/dev_input0_morpho.txt" \
    --testpref "datasets/KIN_RTE/parsed/test_input0_morpho.txt" \
    --destdir "datasets/KIN_RTE/morpho/input0" \
    --workers 8 \
    --srcdict datasets/KIN_RTE/morpho/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_RTE/parsed/train_input1_morpho.txt" \
    --validpref "datasets/KIN_RTE/parsed/dev_input1_morpho.txt" \
    --testpref "datasets/KIN_RTE/parsed/test_input1_morpho.txt" \
    --destdir "datasets/KIN_RTE/morpho/input1" \
    --workers 8 \
    --srcdict datasets/KIN_RTE/morpho/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_RTE/to_translate/train_label.txt" \
    --validpref "datasets/KIN_RTE/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_RTE/morpho/label" \
    --workers 8

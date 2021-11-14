
fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_WNLI/parsed/train_input0_bpe.txt" \
    --validpref "datasets/KIN_WNLI/parsed/dev_input0_bpe.txt" \
    --testpref "datasets/KIN_WNLI/parsed/test_input0_bpe.txt" \
    --destdir "datasets/KIN_WNLI/bpe/input0" \
    --workers 8 \
    --srcdict datasets/KIN_WNLI/bpe/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_WNLI/parsed/train_input1_bpe.txt" \
    --validpref "datasets/KIN_WNLI/parsed/dev_input1_bpe.txt" \
    --testpref "datasets/KIN_WNLI/parsed/test_input1_bpe.txt" \
    --destdir "datasets/KIN_WNLI/bpe/input1" \
    --workers 8 \
    --srcdict datasets/KIN_WNLI/bpe/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_WNLI/to_translate/train_label.txt" \
    --validpref "datasets/KIN_WNLI/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_WNLI/bpe/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_WNLI/parsed/train_input0_spm.txt" \
    --validpref "datasets/KIN_WNLI/parsed/dev_input0_spm.txt" \
    --testpref "datasets/KIN_WNLI/parsed/test_input0_spm.txt" \
    --destdir "datasets/KIN_WNLI/spm/input0" \
    --workers 8 \
    --srcdict datasets/KIN_WNLI/spm/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_WNLI/parsed/train_input1_spm.txt" \
    --validpref "datasets/KIN_WNLI/parsed/dev_input1_spm.txt" \
    --testpref "datasets/KIN_WNLI/parsed/test_input1_spm.txt" \
    --destdir "datasets/KIN_WNLI/spm/input1" \
    --workers 8 \
    --srcdict datasets/KIN_WNLI/spm/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_WNLI/to_translate/train_label.txt" \
    --validpref "datasets/KIN_WNLI/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_WNLI/spm/label" \
    --workers 8

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_WNLI/parsed/train_input0_morpho.txt" \
    --validpref "datasets/KIN_WNLI/parsed/dev_input0_morpho.txt" \
    --testpref "datasets/KIN_WNLI/parsed/test_input0_morpho.txt" \
    --destdir "datasets/KIN_WNLI/morpho/input0" \
    --workers 8 \
    --srcdict datasets/KIN_WNLI/morpho/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_WNLI/parsed/train_input1_morpho.txt" \
    --validpref "datasets/KIN_WNLI/parsed/dev_input1_morpho.txt" \
    --testpref "datasets/KIN_WNLI/parsed/test_input1_morpho.txt" \
    --destdir "datasets/KIN_WNLI/morpho/input1" \
    --workers 8 \
    --srcdict datasets/KIN_WNLI/morpho/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_WNLI/to_translate/train_label.txt" \
    --validpref "datasets/KIN_WNLI/to_translate/dev_label.txt" \
    --destdir "datasets/KIN_WNLI/morpho/label" \
    --workers 8

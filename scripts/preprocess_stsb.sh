
fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_STS-B/parsed/train_input0_bpe.txt" \
    --validpref "datasets/KIN_STS-B/parsed/dev_input0_bpe.txt" \
    --testpref "datasets/KIN_STS-B/parsed/test_input0_bpe.txt" \
    --destdir "datasets/KIN_STS-B/bpe/input0" \
    --workers 8 \
    --srcdict datasets/KIN_STS-B/bpe/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_STS-B/parsed/train_input1_bpe.txt" \
    --validpref "datasets/KIN_STS-B/parsed/dev_input1_bpe.txt" \
    --testpref "datasets/KIN_STS-B/parsed/test_input1_bpe.txt" \
    --destdir "datasets/KIN_STS-B/bpe/input1" \
    --workers 8 \
    --srcdict datasets/KIN_STS-B/bpe/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_STS-B/parsed/train_input0_spm.txt" \
    --validpref "datasets/KIN_STS-B/parsed/dev_input0_spm.txt" \
    --testpref "datasets/KIN_STS-B/parsed/test_input0_spm.txt" \
    --destdir "datasets/KIN_STS-B/spm/input0" \
    --workers 8 \
    --srcdict datasets/KIN_STS-B/spm/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_STS-B/parsed/train_input1_spm.txt" \
    --validpref "datasets/KIN_STS-B/parsed/dev_input1_spm.txt" \
    --testpref "datasets/KIN_STS-B/parsed/test_input1_spm.txt" \
    --destdir "datasets/KIN_STS-B/spm/input1" \
    --workers 8 \
    --srcdict datasets/KIN_STS-B/spm/dict.txt


fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_STS-B/parsed/train_input0_morpho.txt" \
    --validpref "datasets/KIN_STS-B/parsed/dev_input0_morpho.txt" \
    --testpref "datasets/KIN_STS-B/parsed/test_input0_morpho.txt" \
    --destdir "datasets/KIN_STS-B/morpho/input0" \
    --workers 8 \
    --srcdict datasets/KIN_STS-B/morpho/dict.txt



fairseq-preprocess \
    --only-source \
    --trainpref "datasets/KIN_STS-B/parsed/train_input1_morpho.txt" \
    --validpref "datasets/KIN_STS-B/parsed/dev_input1_morpho.txt" \
    --testpref "datasets/KIN_STS-B/parsed/test_input1_morpho.txt" \
    --destdir "datasets/KIN_STS-B/morpho/input1" \
    --workers 8 \
    --srcdict datasets/KIN_STS-B/morpho/dict.txt


mkdir -p "datasets/KIN_STS-B/morpho/label"
awk '{print $1 / 5.0 }' "datasets/KIN_STS-B/to_translate/train_score.txt" > "datasets/KIN_STS-B/morpho/label/train.label"
awk '{print $1 / 5.0 }' "datasets/KIN_STS-B/to_translate/dev_score.txt" > "datasets/KIN_STS-B/morpho/label/valid.label"

mkdir -p "datasets/KIN_STS-B/bpe/label"
awk '{print $1 / 5.0 }' "datasets/KIN_STS-B/to_translate/train_score.txt" > "datasets/KIN_STS-B/bpe/label/train.label"
awk '{print $1 / 5.0 }' "datasets/KIN_STS-B/to_translate/dev_score.txt" > "datasets/KIN_STS-B/bpe/label/valid.label"

mkdir -p "datasets/KIN_STS-B/spm/label"
awk '{print $1 / 5.0 }' "datasets/KIN_STS-B/to_translate/train_score.txt" > "datasets/KIN_STS-B/spm/label/train.label"
awk '{print $1 / 5.0 }' "datasets/KIN_STS-B/to_translate/dev_score.txt" > "datasets/KIN_STS-B/spm/label/valid.label"

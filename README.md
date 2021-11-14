# KinyaBERT

## KinyaBERT: a Morphology-aware Kinyarwanda Language Model

## Brief Introduction

KinyaBERT implements a two-tier BERT architecture for modeling morphologically rich languages.
Current implementation is tailored to Kinyarwanda language, a language spoken Central and Eastern Africa.
This code repository has been anonymized for ACL 2022 blind review and might not work straight out of the repository.
Once the paper gets published, the code will be properly open sourced and provided with sufficient documentation for adapting the model to other languages.

## Repository guide:
- code: main python codebase
- conf: vocabulary files for KinyaBERT
- datasets: evaluation datasets for Translated GLUE benchmark, Named Entity Recognition (NER) and NEWS categorization tasks.
- fairseq-tupe-tpu-pytorch-v1.9: TPU-optimized fairseq code for baseline models. The package has been customized to use TUPE-R position encoding
- lib: shared library for Kinyarwanda morphological analysis and part-of-speech tagging
- results: Fine-tuning results in raw format
- scripts: data pre-processing scripts

## Important files:
1. code/morpho_model.py : KinyaBERT model implementation in PyTorch
2. code/train_exploratory_distributed_model.py : KinyaBERT pre-training process
3. code/pretrained_kinyabert_model_fine_tune_eval.py : KinyaBERT fine-tuning process
4. code/pretrained_roberta_model_fine_tune_eval.py : Baseline models fine-tuning process
5. lib/libkinlp.so : Morphological analyzer/POS Tagger shared library
6. results/FINAL_AVERAGED_RESULTS.xlsx : All experimental results aggregated in a spreadsheet
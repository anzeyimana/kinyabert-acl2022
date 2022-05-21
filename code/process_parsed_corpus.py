# Copyright (c) Antoine Nzeyimana.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import progressbar
import torch
import youtokentome as yttm

from morpho_common import SimpleKBVocab, build_kinlp_morpho_lib, RichParsedToken, parse_raw_text_lines

from kinyabert_utils import read_lines,time_now

def parse_documents_corpus(home_path, in_file_path, out_file_path):
    import fastBPE
    import sentencepiece

    print(time_now(),'Processing', in_file_path, '...')

    codes_path = "fastBPE/rw_codes"
    vocab_path = "fastBPE/vocab.rw.40000"
    spm_model_path = "spm/sentencepiece.bpe.model"

    bpe = fastBPE.fastBPE(codes_path, vocab_path)
    spm = sentencepiece.SentencePieceProcessor(model_file=spm_model_path)

    BPE_model_path = (home_path + "data/BPE-30k.mdl")
    bpe_encoder = yttm.BPE(model=BPE_model_path)

    kb_vocab = SimpleKBVocab()
    kbvocab_state_dict_file_path = (home_path + "data/kb_vocab_state_dict_2021-02-07.pt")
    kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

    f = open(in_file_path, 'r')
    Lines = f.readlines()
    f.close()

    parsed_file = open(out_file_path+'_parsed.txt', 'w')
    morpho_file = open(out_file_path+'_morpho.txt', 'w')
    bpe_file = open(out_file_path+'_bpe.txt', 'w')
    spm_file = open(out_file_path+'_spm.txt', 'w')

    doc_idx = [i for i in range(len(Lines)) if (len(Lines[i]) == 1)]
    if doc_idx[-1] < (len(Lines)-1):
        doc_idx.append(len(Lines))


    start_idx = 0
    all_docs = len(doc_idx)

    print(time_now(), 'Loaded', len(Lines), 'lines', '({} documents)'.format(all_docs))

    with progressbar.ProgressBar(max_value=(all_docs), redirect_stdout=True) as bar:
        bar.update(0)
        for i,end_idx in enumerate(doc_idx):
            lines_batch = Lines[start_idx:end_idx]
            start_idx = end_idx + 1
            if (len(lines_batch) > 0):
                # Morphological Analysis
                parsed_tokens = parse_raw_text_lines(lines_batch, kb_vocab, bpe_encoder)
                # PARSED:
                parsed_file.write('; '.join([pt.to_parsed_format() for pt in parsed_tokens]) + "\n")
                # MORPHO:
                # ' '.join([' '.join(pt.morpho_tokens) for pt in parsed_tokens])
                morpho_file.write((' '.join([' '.join(pt.morpho_tokens) for pt in parsed_tokens])) + "\n")
                # BPE: accepts list of strings and return list of strings
                # bpe.apply(lines_batch)[0]
                bpe_file.write((bpe.apply(lines_batch)[0]) + "\n")
                # SPM: accepts string and returns list of tokens
                # ' '.join(spm.encode(lines_batch[0], out_type=str))
                spm_file.write((' '.join(spm.encode(lines_batch[0], out_type=str))) + "\n")
            if (((i+1) % 1000) == 0):
                print(time_now(),'Processed {}K/{}K docs'.format(int(i/1000.0), int(all_docs/1000.0)))
                bar.update(i+1)
    parsed_file.close()
    morpho_file.close()
    bpe_file.close()
    spm_file.close()

def parse_sentences_corpus(home_path, in_file_path, out_file_path):
    import fastBPE
    import sentencepiece

    print('Processing', in_file_path, '...')

    codes_path = "fastBPE/rw_codes"
    vocab_path = "fastBPE/vocab.rw.40000"
    spm_model_path = "xlmr.base/sentencepiece.bpe.model"

    bpe = fastBPE.fastBPE(codes_path, vocab_path)
    spm = sentencepiece.SentencePieceProcessor(model_file=spm_model_path)

    BPE_model_path = (home_path + "data/BPE-30k.mdl")
    bpe_encoder = yttm.BPE(model=BPE_model_path)

    kb_vocab = SimpleKBVocab()
    kbvocab_state_dict_file_path = (home_path + "data/kb_vocab_state_dict_2021-02-07.pt")
    kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

    Lines = read_lines(in_file_path)

    parsed_file = open(out_file_path+'_parsed.txt', 'w')
    morpho_file = open(out_file_path+'_morpho.txt', 'w')
    bpe_file = open(out_file_path+'_bpe.txt', 'w')
    spm_file = open(out_file_path+'_spm.txt', 'w')
    with progressbar.ProgressBar(max_value=(len(Lines)), redirect_stdout=True) as bar:
        bar.update(0)
        for i in range(len(Lines)):
            lines_batch = Lines[i:(i+1)]
            if (len(lines_batch[0]) > 0):
                # Morphological Analysis
                parsed_tokens = parse_raw_text_lines(lines_batch, kb_vocab, bpe_encoder)
                # PARSED:
                parsed_file.write('; '.join([pt.to_parsed_format() for pt in parsed_tokens]) + "\n")
                # MORPHO:
                # ' '.join([' '.join(pt.morpho_tokens) for pt in parsed_tokens])
                morpho_file.write((' '.join([' '.join(pt.morpho_tokens) for pt in parsed_tokens])) + "\n")
                # BPE: accepts list of strings and return list of strings
                # bpe.apply(lines_batch)[0]
                bpe_file.write((bpe.apply(lines_batch)[0]) + "\n")
                # SPM: accepts string and returns list of tokens
                # ' '.join(spm.encode(lines_batch[0], out_type=str))
                spm_file.write((' '.join(spm.encode(lines_batch[0], out_type=str))) + "\n")

            if (((i+1) % 1000) == 0):
                bar.update(i+1)
    parsed_file.close()
    morpho_file.close()
    bpe_file.close()
    spm_file.close()

if __name__ == '__main__':
    build_kinlp_morpho_lib()
    from kinlpmorpholib import ffi, lib
    conf = "data/kb_config_kinlp.conf"
    # conf = "data/config_kinlp.conf"
    lib.start_kinlp_lib(conf.encode('utf-8'))
    
    data_home_path = "./"

    parse_documents_corpus(data_home_path,
                           'data/better-valid-kinlp-latest.txt',
                           'data/parsed_corpus_2021-07-28')

    # # WNLI -------------------------------------------------------------------------------------------------------------------
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_WNLI/rw_translated/wnli_input_dev_input0_rw_translations.txt",
    #                        "datasets/KIN_WNLI/parsed/dev_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_WNLI/rw_translated/wnli_input_dev_input1_rw_translations.txt",
    #                        "datasets/KIN_WNLI/parsed/dev_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_WNLI/rw_translated/wnli_input_test_input0_rw_translations.txt",
    #                        "datasets/KIN_WNLI/parsed/test_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_WNLI/rw_translated/wnli_input_test_input1_rw_translations.txt",
    #                        "datasets/KIN_WNLI/parsed/test_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_WNLI/rw_translated/wnli_input_train_input0_rw_translations.txt",
    #                        "datasets/KIN_WNLI/parsed/train_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_WNLI/rw_translated/wnli_input_train_input1_rw_translations.txt",
    #                        "datasets/KIN_WNLI/parsed/train_input1")
    #
    # # MRPC -------------------------------------------------------------------------------------------------------------------
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_MRPC/rw_translated/mrpc_input_dev_input0_rw_translations.txt",
    #                        "datasets/KIN_MRPC/parsed/dev_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_MRPC/rw_translated/mrpc_input_dev_input1_rw_translations.txt",
    #                        "datasets/KIN_MRPC/parsed/dev_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_MRPC/rw_translated/mrpc_input_test_input0_rw_translations.txt",
    #                        "datasets/KIN_MRPC/parsed/test_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_MRPC/rw_translated/mrpc_input_test_input1_rw_translations.txt",
    #                        "datasets/KIN_MRPC/parsed/test_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_MRPC/rw_translated/mrpc_input_train_input0_rw_translations.txt",
    #                        "datasets/KIN_MRPC/parsed/train_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_MRPC/rw_translated/mrpc_input_train_input1_rw_translations.txt",
    #                        "datasets/KIN_MRPC/parsed/train_input1")
    #
    # # RTE -------------------------------------------------------------------------------------------------------------------
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_RTE/rw_translated/rte_input_dev_input0_rw_translations.txt",
    #                        "datasets/KIN_RTE/parsed/dev_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_RTE/rw_translated/rte_input_dev_input1_rw_translations.txt",
    #                        "datasets/KIN_RTE/parsed/dev_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_RTE/rw_translated/rte_input_test_input0_rw_translations.txt",
    #                        "datasets/KIN_RTE/parsed/test_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_RTE/rw_translated/rte_input_test_input1_rw_translations.txt",
    #                        "datasets/KIN_RTE/parsed/test_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_RTE/rw_translated/rte_input_train_input0_rw_translations.txt",
    #                        "datasets/KIN_RTE/parsed/train_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_RTE/rw_translated/rte_input_train_input1_rw_translations.txt",
    #                        "datasets/KIN_RTE/parsed/train_input1")
    #
    # # QNLI -------------------------------------------------------------------------------------------------------------------
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_QNLI/rw_translated/qnli_input_dev_input0_rw_translations.txt",
    #                        "datasets/KIN_QNLI/parsed/dev_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_QNLI/rw_translated/qnli_input_dev_input1_rw_translations.txt",
    #                        "datasets/KIN_QNLI/parsed/dev_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_QNLI/rw_translated/qnli_input_test_input0_rw_translations.txt",
    #                        "datasets/KIN_QNLI/parsed/test_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_QNLI/rw_translated/qnli_input_test_input1_rw_translations.txt",
    #                        "datasets/KIN_QNLI/parsed/test_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_QNLI/rw_translated/qnli_input_train_input0_rw_translations.txt",
    #                        "datasets/KIN_QNLI/parsed/train_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_QNLI/rw_translated/qnli_input_train_input1_rw_translations.txt",
    #                        "datasets/KIN_QNLI/parsed/train_input1")
    #
    # # STS-B -------------------------------------------------------------------------------------------------------------------
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_STS-B/rw_translated/stsb_input_dev_input0_rw_translations.txt",
    #                        "datasets/KIN_STS-B/parsed/dev_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_STS-B/rw_translated/stsb_input_dev_input1_rw_translations.txt",
    #                        "datasets/KIN_STS-B/parsed/dev_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_STS-B/rw_translated/stsb_input_test_input0_rw_translations.txt",
    #                        "datasets/KIN_STS-B/parsed/test_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_STS-B/rw_translated/stsb_input_test_input1_rw_translations.txt",
    #                        "datasets/KIN_STS-B/parsed/test_input1")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_STS-B/rw_translated/stsb_input_train_input0_rw_translations.txt",
    #                        "datasets/KIN_STS-B/parsed/train_input0")
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_STS-B/rw_translated/stsb_input_train_input1_rw_translations.txt",
    #                        "datasets/KIN_STS-B/parsed/train_input1")
    #
    # # SST-2 -------------------------------------------------------------------------------------------------------------------
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_SST-2/rw_translated/sst2_input_dev_input0_rw_translations.txt",
    #                        "datasets/KIN_SST-2/parsed/dev_input0")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_SST-2/rw_translated/sst2_input_test_input0_rw_translations.txt",
    #                        "datasets/KIN_SST-2/parsed/test_input0")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/KIN_SST-2/rw_translated/sst2_input_train_input0_rw_translations.txt",
    #                        "datasets/KIN_SST-2/parsed/train_input0")
    #
    # # RW_NEWS -------------------------------------------------------------------------------------------------------------------
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/RW_NEWS/plain/dev.input0.txt",
    #                        "datasets/RW_NEWS/parsed/dev_input0")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/RW_NEWS/plain/test.input0.txt",
    #                        "datasets/RW_NEWS/parsed/test_input0")
    #
    # parse_sentences_corpus(data_home_path,
    #                        "datasets/RW_NEWS/plain/train.input0.txt",
    #                        "datasets/RW_NEWS/parsed/train_input0")

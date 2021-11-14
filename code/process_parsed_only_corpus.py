import progressbar
import torch
import youtokentome as yttm

from morpho_common import SimpleKBVocab, build_kinlp_morpho_lib, RichParsedToken, parse_raw_text_lines

from kinyabert_utils import read_lines,time_now

def parse_documents_corpus(home_path, in_file_path, out_file_path):
    print(time_now(),'Processing', in_file_path, '...')

    BPE_model_path = (home_path + "data/BPE-30k.mdl")
    bpe_encoder = yttm.BPE(model=BPE_model_path)

    kb_vocab = SimpleKBVocab()
    kbvocab_state_dict_file_path = (home_path + "data/kb_vocab_state_dict_2021-02-07.pt")
    kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

    f = open(in_file_path, 'r')
    Lines = f.readlines()
    f.close()

    parsed_file = open(out_file_path+'_parsed.txt', 'w')

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
                _tk_list, grouped_parsed_tokens = parse_raw_text_lines(lines_batch, kb_vocab, bpe_encoder)
                # SAVE PARSED:
                for sentence_tokens in grouped_parsed_tokens:
                    parsed_file.write('; '.join([pt.to_parsed_format() for pt in sentence_tokens]) + "\n")
                parsed_file.write("\n")
                parsed_file.flush()

            if (((i+1) % 1000) == 0):
                print(time_now(),'Processed {}K/{}K docs'.format(int(i/1000.0), int(all_docs/1000.0)))
                bar.update(i+1)
    parsed_file.close()

def parse_sentences_corpus(home_path, in_file_path, out_file_path):
    print('Processing', in_file_path, '...')

    BPE_model_path = (home_path + "data/BPE-30k.mdl")
    bpe_encoder = yttm.BPE(model=BPE_model_path)

    kb_vocab = SimpleKBVocab()
    kbvocab_state_dict_file_path = (home_path + "data/kb_vocab_state_dict_2021-02-07.pt")
    kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

    Lines = read_lines(in_file_path)

    parsed_file = open(out_file_path+'_parsed.txt', 'w')
    with progressbar.ProgressBar(max_value=(len(Lines)), redirect_stdout=True) as bar:
        bar.update(0)
        for i in range(len(Lines)):
            lines_batch = Lines[i:(i+1)]
            if (len(lines_batch[0]) > 0):
                # Morphological Analysis
                parsed_tokens, _grpd = parse_raw_text_lines(lines_batch, kb_vocab, bpe_encoder)
                # PARSED:
                parsed_file.write('; '.join([pt.to_parsed_format() for pt in parsed_tokens]) + "\n")
                parsed_file.flush()

            if (((i+1) % 1000) == 0):
                bar.update(i+1)
    parsed_file.close()

if __name__ == '__main__':
    build_kinlp_morpho_lib()
    from kinlpmorpholib import ffi, lib
    conf = "/home/user/KINLP/data/kb_config_kinlp.conf"
    # conf = "/mnt/NVM/KINLP/data/config_kinlp.conf"
    lib.start_kinlp_lib(conf.encode('utf-8'))
    
    data_home_path = "/home/user/KINLP/"

    # parse_documents_corpus(data_home_path,
    #                        '/home/user/KINLP/data/full_valid_kinlp_corpus_2021-11-05.txt',
    #                        '/home/user/KINLP/data/full_valid_kinlp_corpus_2021-11-05')

    parse_sentences_corpus(data_home_path,
                           '/home/user/KINLP/data/kinmt2021_train_rw.txt',
                           '/home/user/KINLP/data/kinmt2021_train_rw')
    parse_sentences_corpus(data_home_path,
                           '/home/user/KINLP/data/kinmt2021_test_rw.txt',
                           '/home/user/KINLP/data/kinmt2021_test_rw')
    parse_sentences_corpus(data_home_path,
                           '/home/user/KINLP/data/kinmt2021_valid_rw.txt',
                           '/home/user/KINLP/data/kinmt2021_valid_rw')

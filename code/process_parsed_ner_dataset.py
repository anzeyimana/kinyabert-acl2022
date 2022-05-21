# Copyright (c) Antoine Nzeyimana.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import progressbar
import torch
import youtokentome as yttm

from morpho_common import SimpleKBVocab, build_kinlp_morpho_lib, RichParsedToken, parse_raw_text_lines

def parse_ner_dataset(home_path, in_file_path, out_file_path):
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

    f = open(in_file_path, 'r')
    Lines = [line.rstrip('\n') for line in f]
    Lines = ['"' if (l=='""""') else l for l in Lines]
    Lines = [l.replace('“','"').replace('‘‘','"').replace('’’','"').replace('’','\'').replace('‘','\'') for l in Lines]
    f.close()

    doc_idx = [i for i in range(len(Lines)) if (len(Lines[i]) == 0)]
    if doc_idx[-1] < (len(Lines)-1):
        doc_idx.append(len(Lines))
    start_idx = 0

    text_file = open(out_file_path+'_plain.txt', 'w')
    label_file = open(out_file_path+'_labels.txt', 'w')

    parsed_file = open(out_file_path+'_parsed.txt', 'w')

    morpho_file = open(out_file_path+'_morpho.txt', 'w')
    morpho_labels_file = open(out_file_path+'_morpho_labels.txt', 'w')

    bpe_file = open(out_file_path+'_bpe.txt', 'w')
    bpe_labels_file = open(out_file_path+'_bpe_labels.txt', 'w')

    spm_file = open(out_file_path+'_spm.txt', 'w')
    spm_labels_file = open(out_file_path+'_spm_labels.txt', 'w')

    all_docs = len(doc_idx)
    with progressbar.ProgressBar(max_value=(all_docs+10), redirect_stdout=True) as bar:
        bar.update(0)
        for iter,end_idx in enumerate(doc_idx):
            lines_batch = Lines[start_idx:end_idx]
            start_idx = end_idx + 1
            if (len(lines_batch) > 1):
                words = [l.split(' ')[0] for l in lines_batch]
                labels = [l.split(' ')[1] for l in lines_batch]
                sentence = ' '.join(words)
                parsed_tokens = parse_raw_text_lines([sentence], kb_vocab, bpe_encoder)
                assert (len(words) <= len(parsed_tokens))
                start = 0
                new_labels = []
                new_parsed_tokens = []
                for wrd,lbl in zip(words,labels):
                    end = start
                    token = ''
                    while end < len(parsed_tokens):
                        token += parsed_tokens[end].raw_surface_form
                        end += 1
                        if wrd == token:
                            if (end-start) > 1:
                                pts = parsed_tokens[start:end]
                                if (pts[0].is_apostrophed) and lbl.startswith('B-'):
                                    new_labels.extend(['O', lbl])
                                    new_parsed_tokens.extend(pts[:2])
                                    if (len(pts) > 2):
                                        nlb = 'I'+lbl[1:]
                                        new_labels.extend([nlb] * (len(pts)-2))
                                        new_parsed_tokens.extend(pts[2:])
                                else:
                                    new_labels.extend([lbl] * len(pts))
                                    new_parsed_tokens.extend(pts)
                            else:
                                new_labels.append(lbl)
                                new_parsed_tokens.append(parsed_tokens[start])
                            start = end
                            break
                    # Check match
                    if not ((wrd == token) and (start == end)):
                        print('Error with:',sentence)
                        print('Got parse::',' '.join([p.raw_surface_form for p in parsed_tokens]))
                        print('Mismatch: ','words: \'{}\' ~ \'{}\''.format(wrd,token), 'start:' + str(start) + ' end: ' + str(end))
                        break
                # Now assemble new dataset
                parsed_file.write('; '.join([pt.to_parsed_format() for pt in new_parsed_tokens]) + "\n")
                text_file.write(' '.join([pt.raw_surface_form for pt in new_parsed_tokens]) + "\n")
                label_file.write(' '.join([lbl for lbl in new_labels]) + "\n")
                # MORPHO
                morpho_file.write(' '.join([' '.join(pt.morpho_tokens) for pt in new_parsed_tokens]) + "\n")
                morpho_labels_file.write(' '.join([
                    ' '.join([lbl]+(['I'+(lbl[1:])] * (len(pt.morpho_tokens)-1))) if lbl.startswith('B-')
                    else ' '.join([lbl] * len(pt.morpho_tokens))
                    for pt,lbl in zip(new_parsed_tokens,new_labels)
                ]) + "\n")
                # BPE: List[str] --> List[str (space-separated)]
                bpe_tokens = bpe.apply([pt.raw_surface_form for pt in new_parsed_tokens])
                bpe_tokens = [t.split(' ') for t in bpe_tokens]
                bpe_file.write(' '.join([' '.join(bt) for bt in bpe_tokens]) + "\n")
                bpe_labels_file.write(' '.join([
                    ' '.join([lbl]+(['I'+(lbl[1:])] * (len(bt)-1))) if lbl.startswith('B-')
                    else ' '.join([lbl] * len(bt))
                    for bt,lbl in zip(bpe_tokens,new_labels)
                ]) + "\n")
                # SPM: str --> List[str]
                spm_tokens = [spm.encode(pt.raw_surface_form, out_type=str) for pt in new_parsed_tokens]
                spm_file.write(' '.join([' '.join(st) for st in spm_tokens]) + "\n")
                spm_labels_file.write(' '.join([
                    ' '.join([lbl]+(['I'+(lbl[1:])] * (len(st)-1))) if lbl.startswith('B-')
                    else ' '.join([lbl] * len(st))
                    for st,lbl in zip(spm_tokens,new_labels)
                ]) + "\n")

    parsed_file.close()
    text_file.close()
    label_file.close()
    morpho_file.close()
    morpho_labels_file.close()
    bpe_file.close()
    bpe_labels_file.close()
    spm_file.close()
    spm_labels_file.close()

if __name__ == '__main__':
    build_kinlp_morpho_lib()
    from kinlpmorpholib import ffi, lib
    #conf = "data/kb_config_kinlp.conf"
    conf = "data/config_kinlp.conf"
    lib.start_kinlp_lib(conf.encode('utf-8'))

    data_home_path = "./"

    parse_ner_dataset(data_home_path,
                       "datasets/KIN_NER/original/dev.txt",
                       "datasets/KIN_NER/parsed/dev")
    parse_ner_dataset(data_home_path,
                       "datasets/KIN_NER/original/test.txt",
                       "datasets/KIN_NER/parsed/test")
    parse_ner_dataset(data_home_path,
                       "datasets/KIN_NER/original/train.txt",
                       "datasets/KIN_NER/parsed/train")
    lib.stop_kinlp_lib()

from __future__ import print_function, division

# Ignore warnings
import warnings
from typing import List

import numpy as np

warnings.filterwarnings("ignore")

from morpho_data_loaders import ParsedToken, KBVocab, AffixSetVocab, parse_raw_text_lines

def inf_process_parsed_sentence(args, parsed_tokens_list: List[ParsedToken], kv : KBVocab, affix_set_vocab : AffixSetVocab, rel_pos_dict, rel_pos_dmax, mask_ids):
    # Input data
    pos_tags = []
    stems = []
    afsets = [] if args.use_afsets else None
    affixes = []
    tokens_lengths = []

    # Predicted data
    predicted_stems = []
    predicted_afsets = [] if args.use_afsets else None
    predicted_affixes = [] if args.predict_affixes else None

    predicted_tokens_idx = []
    predicted_tokens_affixes_idx = [] if args.predict_affixes else None
    predicted_tokens_affixes_lengths = [] if args.predict_affixes else None

    add_cls = True

    pos_tags.append(kv.pos_tag_vocab['<CLS>'])
    stems.append(kv.reduced_stem_vocab['<CLS>'])
    if args.use_afsets:
        afsets.append(affix_set_vocab.affix_set_to_idx('<CLS>'))
    tokens_lengths.append(0)

    for tidx,ptoken in enumerate(parsed_tokens_list):
        for sidx in ptoken.stem_idx:
            predict = False
            unchanged = True
            if (tidx+1) in mask_ids:
                predict = True
                unchanged = False
                pos_tags.append(kv.pos_tag_vocab['<MSK>'])
                stems.append(kv.reduced_stem_vocab['<MSK>'])
                if args.use_afsets:
                    afsets.append(affix_set_vocab.affix_set_to_idx('<MSK>'))
                tokens_lengths.append(0)

            if(unchanged):
                pos_tags.append(ptoken.pos_tag_idx)
                stems.append(kv.mapped_stem_vocab_idx[sidx])
                if args.use_afsets:
                    afsets.append(affix_set_vocab.affix_set_to_idx(ptoken.affix_set_key()))

                affixes.extend([(v) for v in ptoken.affixes_idx])
                tokens_lengths.append(len(ptoken.affixes_idx))

            if(predict):
                predicted_stems.append(kv.mapped_stem_vocab_idx[sidx])
                predicted_tokens_idx.append(len(tokens_lengths) - 1)
                if args.use_afsets:
                    predicted_afsets.append(affix_set_vocab.affix_set_to_idx(ptoken.affix_set_key()))
                if args.predict_affixes:
                    predicted_affixes.extend([(v) for v in ptoken.affixes_idx])
                    if (len(ptoken.affixes_idx) > 0):
                        predicted_tokens_affixes_idx.append(len(predicted_tokens_idx) - 1)
                        predicted_tokens_affixes_lengths.append(len(ptoken.affixes_idx))

    rel_pos_arr = np.zeros((len(pos_tags), len(pos_tags))).astype(int) if (rel_pos_dict is not None) else None
    if rel_pos_arr is not None:
        for i,pi in enumerate(pos_tags):
            for j,pj in enumerate(pos_tags):
                d = i - j
                if ((d != 0) and (d >= -rel_pos_dmax) and (d <= rel_pos_dmax) and ((not add_cls) or ((i != 0) and (j != 0)))):
                    rel_pos_arr[i, j] = rel_pos_dict[(pi, pj, d)]
    return (rel_pos_arr,
            pos_tags,
            stems,
            afsets,
            affixes,
            tokens_lengths,
            predicted_stems,
            predicted_afsets,
            predicted_affixes,
            predicted_tokens_idx,
            predicted_tokens_affixes_idx,
            predicted_tokens_affixes_lengths)

def inf_gather_replicated_itemized_data(args, input_line, max_seq_len, kb_vocab : KBVocab, affix_set_vocab : AffixSetVocab, bpe, rel_pos_dict, rel_pos_dmax, mask_ids):
    itemized_data = []
    itemized_parsed_tokens = []

    seq_pos_tags = []
    seq_stems = []
    seq_afsets = [] if args.use_afsets else None
    seq_affixes = []
    seq_tokens_lengths = []
    seq_predicted_stems = []
    seq_predicted_afsets = [] if args.use_afsets else None
    seq_predicted_affixes = [] if args.predict_affixes else None
    seq_predicted_tokens_idx = []
    seq_predicted_tokens_affixes_idx = [] if args.predict_affixes else None
    seq_predicted_tokens_affixes_lengths = [] if args.predict_affixes else None

    seq_rel_pos_arr = np.zeros((max_seq_len, max_seq_len)).astype(int) if (rel_pos_dict is not None) else None

    seq_parsed_tokens = []

    ptoken = ParsedToken('<CLS>', decode_prob=1.0, tf_idf=0.001, pos_tag_id=kb_vocab.pos_tag_vocab['<CLS>'], stem_ids=[kb_vocab.reduced_stem_vocab['<CLS>']])
    ptoken.append_morpheme(kb_vocab.morpheme_slot_vocab['<EOT>'], kb_vocab.affix_vocab['<EOT>'])
    seq_parsed_tokens.append(ptoken)

    parsed_tokens_line = parse_raw_text_lines(input_line, kb_vocab, bpe)

    seq_parsed_tokens.extend(parsed_tokens_line)

    (rel_pos_arr,
     pos_tags,
     stems,
     afsets,
     affixes,
     tokens_lengths,
     predicted_stems,
     predicted_afsets,
     predicted_affixes,
     predicted_tokens_idx,
     predicted_tokens_affixes_idx,
     predicted_tokens_affixes_lengths) = inf_process_parsed_sentence(args, parsed_tokens_line, kb_vocab, affix_set_vocab, rel_pos_dict, rel_pos_dmax, mask_ids)

    if args.predict_affixes:
        seq_predicted_tokens_affixes_idx.extend([len(seq_predicted_tokens_idx) + idx for idx in predicted_tokens_affixes_idx])

    seq_predicted_tokens_idx.extend([len(seq_tokens_lengths)+idx for idx in predicted_tokens_idx])

    lt = len(seq_tokens_lengths)
    if seq_rel_pos_arr is not None:
        seq_rel_pos_arr[lt:(lt+rel_pos_arr.shape[0]), lt:(lt+rel_pos_arr.shape[1])] = rel_pos_arr

    seq_pos_tags.extend(pos_tags)
    seq_stems.extend(stems)
    if args.use_afsets:
        seq_afsets.extend(afsets)
    seq_affixes.extend(affixes)
    seq_tokens_lengths.extend(tokens_lengths)
    seq_predicted_stems.extend(predicted_stems)
    if args.use_afsets:
        seq_predicted_afsets.extend(predicted_afsets)
    if args.predict_affixes:
        seq_predicted_affixes.extend(predicted_affixes)
        seq_predicted_tokens_affixes_lengths.extend(predicted_tokens_affixes_lengths)

    data_item = (max_seq_len,
                 seq_rel_pos_arr,
                 seq_pos_tags,
                 seq_stems,
                 seq_afsets,
                 seq_affixes,
                 seq_tokens_lengths,
                 seq_predicted_stems,
                 seq_predicted_afsets,
                 seq_predicted_affixes,
                 seq_predicted_tokens_idx,
                 seq_predicted_tokens_affixes_idx,
                 seq_predicted_tokens_affixes_lengths)
    itemized_data.append(data_item)
    itemized_parsed_tokens.append(seq_parsed_tokens)

    return itemized_data, itemized_parsed_tokens

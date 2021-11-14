
from __future__ import print_function, division

from datetime import datetime
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
import random

def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from morpho_data_loaders import KBVocab, AffixSetVocab, ParsedToken

def process_input_sequence(args, input0: List[ParsedToken], input1: List[ParsedToken], tags, kv: KBVocab, affix_set_vocab: AffixSetVocab, rel_pos_dict, rel_pos_dmax, max_seq_len):
    # Input data
    pos_tags = []
    stems = []
    afsets = [] if args.use_afsets else None
    affixes = []
    tokens_lengths = []

    # Add <CLS> token
    pos_tags.append(kv.pos_tag_vocab['<CLS>'])
    stems.append(kv.reduced_stem_vocab['<CLS>'])
    if args.use_afsets:
        afsets.append(affix_set_vocab.affix_set_to_idx('<CLS>'))
    tokens_lengths.append(0)

    for ptoken in input0:
        if(len(tokens_lengths) >= max_seq_len):
            break
        for sidx in ptoken.stem_idx:
            if (len(tokens_lengths) >= max_seq_len):
                break
            pos_tags.append(ptoken.pos_tag_idx)
            stems.append(kv.mapped_stem_vocab_idx[sidx])
            if args.use_afsets:
                afsets.append(affix_set_vocab.affix_set_to_idx(ptoken.affix_set_key()))
            affixes.extend([(v) for v in ptoken.affixes_idx])
            tokens_lengths.append(len(ptoken.affixes_idx))

    br = len(pos_tags) # Break ID
    if ((len(input1) > 0) and (len(tokens_lengths) < max_seq_len)): # New document
        pos_tags.append(kv.pos_tag_vocab['<SEP>'])
        stems.append(kv.reduced_stem_vocab['<SEP>'])
        if args.use_afsets:
            afsets.append(affix_set_vocab.affix_set_to_idx('<SEP>'))
        tokens_lengths.append(0)

        for ptoken in input1:
            if (len(tokens_lengths) >= max_seq_len):
                break
            for sidx in ptoken.stem_idx:
                if (len(tokens_lengths) >= max_seq_len):
                    break
                pos_tags.append(ptoken.pos_tag_idx)
                stems.append(kv.mapped_stem_vocab_idx[sidx])
                if args.use_afsets:
                    afsets.append(affix_set_vocab.affix_set_to_idx(ptoken.affix_set_key()))
                affixes.extend([(v) for v in ptoken.affixes_idx])
                tokens_lengths.append(len(ptoken.affixes_idx))

    rel_pos_arr = np.zeros((max_seq_len, max_seq_len)).astype(int) if (rel_pos_dict is not None) else None
    if rel_pos_arr is not None:
        for i,pi in enumerate(pos_tags):
            for j,pj in enumerate(pos_tags):
                d = i - j
                if ((d != 0) and (d >= -rel_pos_dmax) and (d <= rel_pos_dmax) and ((i != 0) and (j != 0)) and (((i<=br) and (j<=br)) or ((i>=br) and (j>=br)))):
                    rel_pos_arr[i, j] = rel_pos_dict[(pi, pj, d)]

    return (tags,
            max_seq_len,
            rel_pos_arr,
            pos_tags,
            stems,
            afsets,
            affixes,
            tokens_lengths)

def collate_input_sequences(batch_items):
    batch_pos_tags = []
    batch_stems = []
    batch_afsets = []
    batch_affixes = []
    batch_tokens_lengths = []

    batch_tags = []

    batch_input_sequence_lengths = []

    max_sequence_len = batch_items[0][1]
    first_seq_rel_pos_arr = batch_items[0][2]

    batch_rel_pos_arr = np.zeros((len(batch_items), max_sequence_len, max_sequence_len)).astype(int) if (first_seq_rel_pos_arr is not None) else None
    for bidx,data_item in enumerate(batch_items):
        (seq_tags,
         max_seq_len,
         seq_rel_pos_arr,
         seq_pos_tags,
         seq_stems,
         seq_afsets,
         seq_affixes,
         seq_tokens_lengths) = data_item

        if batch_rel_pos_arr is not None:
            batch_rel_pos_arr[bidx,:,:] = seq_rel_pos_arr

        batch_pos_tags.extend(seq_pos_tags)
        batch_stems.extend(seq_stems)
        if seq_afsets is not None:
            batch_afsets.extend(seq_afsets)
        batch_affixes.extend(seq_affixes)
        batch_tokens_lengths.extend(seq_tokens_lengths)

        batch_tags.extend(seq_tags)

        batch_input_sequence_lengths.append(len(seq_tokens_lengths))

    data_item = (batch_tags,
                 batch_input_sequence_lengths,
                 batch_rel_pos_arr,
                 batch_pos_tags,
                 batch_stems,
                 batch_afsets,
                 batch_affixes,
                 batch_tokens_lengths)
    return data_item


from morpho_data_loaders import ParsedToken
import progressbar

class KBTagCorpusDataset(Dataset):

    def __init__(self, args,
                 kb_vocab : KBVocab, affix_set_vocab : AffixSetVocab,
                 rel_pos_dict, rel_pos_dmax,
                 tags_dict, tag_lines,
                 lines_input0, max_lines = 12000,
                 max_seq_len = 512):
        self.max_seq_len = max_seq_len
        self.itemized_data = []
        assert (len(lines_input0)==len(tag_lines))
        num_lines = min(min(len(lines_input0), max_lines), len(tag_lines))
        shuffle = False
        if (num_lines < len(tag_lines)):
            print('Have to shuffle, why?')
            shuffle = True
        with progressbar.ProgressBar(initial_value=0, max_value=(2*num_lines), redirect_stdout=True) as bar:
            bar.update(0)
            for it in range(num_lines):
                idx = it
                if shuffle:
                    idx = random.randint(0, len(lines_input0) - 1) % len(lines_input0)
                line0 = lines_input0[idx]
                tags_line = tag_lines[idx]
                tags_idx = [tags_dict[tg] for tg in tags_line.split(' ')]
                input0 = [ParsedToken('_', parsed_token=t) for t in line0.split('; ')]
                assert (len(tags_idx) == sum([len(t.stem_idx) for t in input0]))
                input1 = []
                if (len(input0) > 510):
                    print('Some data are very long:',len(input0))
                    input0 = input0[:510]
                    tags_idx = tags_idx[:510]
                self.itemized_data.append(process_input_sequence(args, input0, input1, tags_idx, kb_vocab, affix_set_vocab, rel_pos_dict, rel_pos_dmax, self.max_seq_len))
                if ((it % 1000) == 0):
                    bar.update(it)

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

from morpho_model import KinyaBERTSequenceTagger

def tagger_morpho_model_forward(args, batch_data_item, tagger_model : KinyaBERTSequenceTagger, device):
    (batch_tags,
     batch_input_sequence_lengths,
     batch_rel_pos_arr,
     batch_pos_tags,
     batch_stems,
     batch_afsets,
     batch_affixes,
     batch_tokens_lengths) = batch_data_item

    tokens_lengths = batch_tokens_lengths  # torch.tensor(batch_tokens_lengths).to(device)
    input_sequence_lengths = batch_input_sequence_lengths  # torch.tensor(batch_input_sequence_lengths).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    stems = torch.tensor(batch_stems).to(device)
    afsets = torch.tensor(batch_afsets).to(device) if args.use_afsets else None
    affixes = torch.tensor(batch_affixes).to(device)

    rel_pos_arr = torch.from_numpy(batch_rel_pos_arr).to(device) if (batch_rel_pos_arr is not None) else None

    output_scores =  tagger_model(args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes)
    return output_scores, batch_tags

def tagger_morpho_model_predict(args, seq_data_item, tagger_model : KinyaBERTSequenceTagger, device):
    (tags,
     max_seq_len,
     seq_rel_pos_arr,
     seq_pos_tags,
     seq_stems,
     batch_afsets,
     seq_affixes,
     seq_tokens_lengths) = seq_data_item

    tokens_lengths = seq_tokens_lengths
    input_sequence_lengths = [len(seq_tokens_lengths)]
    pos_tags = torch.tensor(seq_pos_tags).to(device)
    stems = torch.tensor(seq_stems).to(device)
    afsets = torch.tensor(batch_afsets).to(device) if args.use_afsets else None
    affixes = torch.tensor(seq_affixes).to(device)

    rel_pos_arr = torch.from_numpy(seq_rel_pos_arr).unsqueeze(0).to(device) if (seq_rel_pos_arr is not None) else None

    output_scores =  tagger_model(args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes)
    predicted_labels = torch.argmax(output_scores, dim=1)
    return output_scores, predicted_labels, tags

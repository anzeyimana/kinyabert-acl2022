# Copyright (c) Antoine Nzeyimana.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import print_function, division

import math
import random
import sys
# Ignore warnings
import warnings
from typing import List

import numpy as np
import progressbar
import torch
import youtokentome as yttm
from torch.utils.data import Dataset

from kinyabert_utils import time_now

warnings.filterwarnings("ignore")

def read_corpus(fn):
    f = open(fn, 'r+')
    corpus_lines = [line.rstrip('\n') for line in f]
    f.close()
    corpus_lines.append("\n")
    return corpus_lines

def read_vocab_idx(fn, voc, voc_idx):
    f = open(fn, 'r')
    v_lines = f.readlines()
    f.close()
    for v in v_lines:
        if(len(v)>1):
            vt = v.split('\t')
            if(len(vt) == 2):
                voc_idx[int(vt[1])] = vt[0]
                voc[vt[0]] = int(vt[1])
    return voc, voc_idx

def read_vocab_counts(fn, voc, voc_idx_counts):
    f = open(fn, 'r')
    v_lines = f.readlines()
    f.close()
    for v in v_lines:
        if(len(v)>1):
            vt = v.split('\t')
            if(len(vt) == 2):
                id = voc[vt[0]]
                voc_idx_counts[id] = int(vt[1])
    return voc, voc_idx_counts

def reduce_stem_vocab(_stem_vocab, _stem_vocab_idx, _stem_vocab_idx_counts,
                      noun_min_count, verb_min_count, np_min_count,
                      other_morpho_min_count, other_cls_min_count, other_token_min_count):
    reduced_stem_vocab = dict()
    mapped_stem_vocab_idx = dict()
    reduced_stem_vocab_idx_counts = dict()
    unk_idx = _stem_vocab['<UNK>']
    for i in range(1,len(_stem_vocab_idx)+1):
        if (i <= unk_idx):
            mapped_stem_vocab_idx[i] = i
            reduced_stem_vocab[_stem_vocab_idx[i]] = i
            reduced_stem_vocab_idx_counts[i] = 1000000
        else:
            key = _stem_vocab_idx[i]
            if (key.startswith('N:')):
                min_count = noun_min_count
            elif (key.startswith('V:')):
                min_count = verb_min_count
            elif (key.startswith('QA:') or key.startswith('PO:') or key.startswith('DE:') or key.startswith('NU:') or key.startswith('OT:')):
                min_count = other_morpho_min_count
            elif (key.startswith('NP:')):
                min_count = np_min_count
            elif (key.startswith('T:')):
                min_count = other_token_min_count
            else:
                min_count = other_cls_min_count
            if _stem_vocab_idx_counts[i] >= min_count:
                idx = len(reduced_stem_vocab)+1
                mapped_stem_vocab_idx[i] = idx
                reduced_stem_vocab[_stem_vocab_idx[i]] = idx
                reduced_stem_vocab_idx_counts[idx] = _stem_vocab_idx_counts[i]
            else:
                mapped_stem_vocab_idx[i] = reduced_stem_vocab['<UNK>']

    return reduced_stem_vocab, mapped_stem_vocab_idx, reduced_stem_vocab_idx_counts

def sigmoid_score(x, min_val, max_val):
    s = pow((1.0 + (math.exp(((-8.0) * (x - min_val)) / (max_val - min_val)))), -8.0)
    return s

class KBVocab:

    def __init__(self, noun_min_count = 200,
                 verb_min_count = 100,
                 np_min_count = 200,
                 other_morpho_min_count = 20,
                 other_cls_min_count = 20,
                 other_token_min_count = 200,
                 pos_tag_vocab_idx_tsv="data/pos_tag_vocab_idx_2021-02-07.tsv",
                 pos_tag_vocab_tsv="data/pos_tag_vocab_2021-02-07.tsv",
                 stem_vocab_idx_tsv="data/stem_vocab_idx_2021-02-07.tsv",
                 stem_vocab_tsv="data/stem_vocab_2021-02-07.tsv",
                 morpheme_slot_vocab_idx_tsv="data/morpheme_slot_vocab_idx_2021-02-07.tsv",
                 morpheme_slot_vocab_tsv="data/morpheme_slot_vocab_2021-02-07.tsv",
                 affix_vocab_idx_tsv="data/affix_vocab_idx_2021-02-07.tsv",
                 affix_vocab_tsv="data/affix_vocab_2021-02-07.tsv",
                 read_vocab_files = False):
        self.pos_tag_vocab = dict()
        self.pos_tag_vocab_idx = dict()
        self.pos_tag_vocab_idx_counts = dict()
        self.pos_tag_vocab_idx_subsample_weights = dict()
        self.pos_tag_vocab_idx_subsample_weights_max = 0.1
        self.pos_tag_vocab_idx_subsample_weights_min = 0.1
        if(read_vocab_files):
            read_vocab_idx(pos_tag_vocab_idx_tsv, self.pos_tag_vocab, self.pos_tag_vocab_idx)
            read_vocab_counts(pos_tag_vocab_tsv, self.pos_tag_vocab, self.pos_tag_vocab_idx_counts)

        self._stem_vocab = dict()
        self._stem_vocab_idx = dict()
        self._stem_vocab_idx_counts = dict()
        if(read_vocab_files):
            read_vocab_idx(stem_vocab_idx_tsv, self._stem_vocab, self._stem_vocab_idx)
            read_vocab_counts(stem_vocab_tsv, self._stem_vocab, self._stem_vocab_idx_counts)

        self.reduced_stem_vocab = dict()
        self.reduced_stem_vocab_idx = dict()
        self.mapped_stem_vocab_idx = dict()
        self.reduced_stem_vocab_idx_counts = dict()
        self.reduced_stem_vocab_idx_subsample_weights = dict()
        self.reduced_stem_vocab_idx_subsample_weights_max = 0.1
        self.reduced_stem_vocab_idx_subsample_weights_min = 0.1
        if(read_vocab_files):
            (self.reduced_stem_vocab,
             self.mapped_stem_vocab_idx,
             self.reduced_stem_vocab_idx_counts) = reduce_stem_vocab(self._stem_vocab,
                                                                     self._stem_vocab_idx,
                                                                     self._stem_vocab_idx_counts,
                                                                     noun_min_count,
                                                                     verb_min_count,
                                                                     np_min_count,
                                                                     other_morpho_min_count,
                                                                     other_cls_min_count,
                                                                     other_token_min_count)
            for k in self.reduced_stem_vocab:
                self.reduced_stem_vocab_idx[self.reduced_stem_vocab[k]] = k

        self.morpheme_slot_vocab = dict()
        self.morpheme_slot_vocab_idx = dict()
        self.morpheme_slot_vocab_idx_counts = dict()
        self.morpheme_slot_vocab_idx_subsample_weights = dict()
        self.morpheme_slot_vocab_idx_subsample_weights_max = 0.1
        self.morpheme_slot_vocab_idx_subsample_weights_min = 0.1
        if(read_vocab_files):
            read_vocab_idx(morpheme_slot_vocab_idx_tsv, self.morpheme_slot_vocab, self.morpheme_slot_vocab_idx)
            read_vocab_counts(morpheme_slot_vocab_tsv, self.morpheme_slot_vocab, self.morpheme_slot_vocab_idx_counts)

        self.affix_vocab = dict()
        self.affix_vocab_idx = dict()
        self.affix_vocab_idx_counts = dict()
        self.affix_vocab_idx_subsample_weights = dict()
        self.affix_vocab_idx_subsample_weights_max = 0.1
        self.affix_vocab_idx_subsample_weights_min = 0.1
        if(read_vocab_files):
            read_vocab_idx(affix_vocab_idx_tsv, self.affix_vocab, self.affix_vocab_idx)
            read_vocab_counts(affix_vocab_tsv, self.affix_vocab, self.affix_vocab_idx_counts)

    def compute_subsampling_weights(self):
        eps = 1e-5
        sum_pos_tags = sum([self.pos_tag_vocab_idx_counts[k] for k in self.pos_tag_vocab_idx_counts])
        for k in self.pos_tag_vocab_idx_counts:
            z = float(self.pos_tag_vocab_idx_counts[k])/float(sum_pos_tags)
            z = z + eps
            self.pos_tag_vocab_idx_subsample_weights[k] = (math.sqrt(z/0.001)+1.0)*(0.001/z)
        self.pos_tag_vocab_idx_subsample_weights_max = max([self.pos_tag_vocab_idx_subsample_weights[k] for k in self.pos_tag_vocab_idx_subsample_weights])
        self.pos_tag_vocab_idx_subsample_weights_min = min([self.pos_tag_vocab_idx_subsample_weights[k] for k in self.pos_tag_vocab_idx_subsample_weights])
        for k in self.pos_tag_vocab_idx_subsample_weights:
            if (k <= self.pos_tag_vocab['<UNK>']):
                self.pos_tag_vocab_idx_subsample_weights[k] = self.pos_tag_vocab_idx_subsample_weights_min

        sum_morpheme_slots = sum([self.morpheme_slot_vocab_idx_counts[k] for k in self.morpheme_slot_vocab_idx_counts])
        for k in self.morpheme_slot_vocab_idx_counts:
            z = float(self.morpheme_slot_vocab_idx_counts[k])/float(sum_morpheme_slots)
            z = z + eps
            self.morpheme_slot_vocab_idx_subsample_weights[k] = (math.sqrt(z/0.001)+1.0)*(0.001/z)
        self.morpheme_slot_vocab_idx_subsample_weights_max = max([self.morpheme_slot_vocab_idx_subsample_weights[k] for k in self.morpheme_slot_vocab_idx_subsample_weights])
        self.morpheme_slot_vocab_idx_subsample_weights_min = min([self.morpheme_slot_vocab_idx_subsample_weights[k] for k in self.morpheme_slot_vocab_idx_subsample_weights])
        for k in self.morpheme_slot_vocab_idx_subsample_weights:
            if (k <= self.morpheme_slot_vocab['<UNK>']):
                self.morpheme_slot_vocab_idx_subsample_weights[k] = self.morpheme_slot_vocab_idx_subsample_weights_min

        sum_affixes = sum([self.affix_vocab_idx_counts[k] for k in self.affix_vocab_idx_counts])
        for k in self.affix_vocab_idx_counts:
            z = float(self.affix_vocab_idx_counts[k])/float(sum_affixes)
            z = z + eps
            self.affix_vocab_idx_subsample_weights[k] = (math.sqrt(z/0.001)+1.0)*(0.001/z)
        self.affix_vocab_idx_subsample_weights_max = max([self.affix_vocab_idx_subsample_weights[k] for k in self.affix_vocab_idx_subsample_weights])
        self.affix_vocab_idx_subsample_weights_min = min([self.affix_vocab_idx_subsample_weights[k] for k in self.affix_vocab_idx_subsample_weights])
        for k in self.affix_vocab_idx_subsample_weights:
            if (k <= self.affix_vocab['<UNK>']):
                self.affix_vocab_idx_subsample_weights[k] = self.affix_vocab_idx_subsample_weights_min

        sum_stems = sum([self.reduced_stem_vocab_idx_counts[k] for k in self.reduced_stem_vocab_idx_counts])
        for k in self.reduced_stem_vocab_idx_counts:
            z = float(self.reduced_stem_vocab_idx_counts[k])/float(sum_stems)
            z = z + eps
            self.reduced_stem_vocab_idx_subsample_weights[k] = (math.sqrt(z/0.001)+1.0)*(0.001/z)
        self.reduced_stem_vocab_idx_subsample_weights_max = max([self.reduced_stem_vocab_idx_subsample_weights[k] for k in self.reduced_stem_vocab_idx_subsample_weights])
        self.reduced_stem_vocab_idx_subsample_weights_min = min([self.reduced_stem_vocab_idx_subsample_weights[k] for k in self.reduced_stem_vocab_idx_subsample_weights])
        for k in self.reduced_stem_vocab_idx_subsample_weights:
            if (k <= self.reduced_stem_vocab['<UNK>']):
                self.reduced_stem_vocab_idx_subsample_weights[k] = self.reduced_stem_vocab_idx_subsample_weights_min

    def state_dict(self):
        return {'pos_tag_vocab':self.pos_tag_vocab,
                'pos_tag_vocab_idx':self.pos_tag_vocab_idx,
                'pos_tag_vocab_idx_counts':self.pos_tag_vocab_idx_counts,
                '_stem_vocab':self._stem_vocab,
                '_stem_vocab_idx':self._stem_vocab_idx,
                '_stem_vocab_idx_counts':self._stem_vocab_idx_counts,
                'reduced_stem_vocab':self.reduced_stem_vocab,
                'mapped_stem_vocab_idx':self.mapped_stem_vocab_idx,
                'reduced_stem_vocab_idx_counts':self.reduced_stem_vocab_idx_counts,
                'morpheme_slot_vocab':self.morpheme_slot_vocab,
                'morpheme_slot_vocab_idx':self.morpheme_slot_vocab_idx,
                'morpheme_slot_vocab_idx_counts':self.morpheme_slot_vocab_idx_counts,
                'affix_vocab':self.affix_vocab,
                'affix_vocab_idx':self.affix_vocab_idx,
                'affix_vocab_idx_counts':self.affix_vocab_idx_counts,

                'morpheme_slot_vocab_idx_subsample_weights': self.morpheme_slot_vocab_idx_subsample_weights,
                'morpheme_slot_vocab_idx_subsample_weights_max': self.morpheme_slot_vocab_idx_subsample_weights_max,
                'morpheme_slot_vocab_idx_subsample_weights_min': self.morpheme_slot_vocab_idx_subsample_weights_min,

                'affix_vocab_idx_subsample_weights':self.affix_vocab_idx_subsample_weights,
                'affix_vocab_idx_subsample_weights_max':self.affix_vocab_idx_subsample_weights_max,
                'affix_vocab_idx_subsample_weights_min':self.affix_vocab_idx_subsample_weights_min,

                'pos_tag_vocab_idx_subsample_weights': self.pos_tag_vocab_idx_subsample_weights,
                'pos_tag_vocab_idx_subsample_weights_max': self.pos_tag_vocab_idx_subsample_weights_max,
                'pos_tag_vocab_idx_subsample_weights_min': self.pos_tag_vocab_idx_subsample_weights_min,

                'reduced_stem_vocab_idx_subsample_weights':self.reduced_stem_vocab_idx_subsample_weights,
                'reduced_stem_vocab_idx_subsample_weights_max':self.reduced_stem_vocab_idx_subsample_weights_max,
                'reduced_stem_vocab_idx_subsample_weights_min':self.reduced_stem_vocab_idx_subsample_weights_min}

    def load_state_dict(self, d):
        self.pos_tag_vocab = d['pos_tag_vocab']
        self.pos_tag_vocab_idx = d['pos_tag_vocab_idx']
        self.pos_tag_vocab_idx_counts = d['pos_tag_vocab_idx_counts']
        self._stem_vocab = d['_stem_vocab']
        self._stem_vocab_idx = d['_stem_vocab_idx']
        self._stem_vocab_idx_counts = d['_stem_vocab_idx_counts']
        self.reduced_stem_vocab = d['reduced_stem_vocab']
        self.mapped_stem_vocab_idx = d['mapped_stem_vocab_idx']
        self.reduced_stem_vocab_idx_counts = d['reduced_stem_vocab_idx_counts']
        self.morpheme_slot_vocab = d['morpheme_slot_vocab']
        self.morpheme_slot_vocab_idx = d['morpheme_slot_vocab_idx']
        self.morpheme_slot_vocab_idx_counts = d['morpheme_slot_vocab_idx_counts']
        self.affix_vocab = d['affix_vocab']
        self.affix_vocab_idx = d['affix_vocab_idx']
        self.affix_vocab_idx_counts = d['affix_vocab_idx_counts']

        self.morpheme_slot_vocab_idx_subsample_weights = d['morpheme_slot_vocab_idx_subsample_weights']
        self.morpheme_slot_vocab_idx_subsample_weights_max = d['morpheme_slot_vocab_idx_subsample_weights_max']
        self.morpheme_slot_vocab_idx_subsample_weights_min = d['morpheme_slot_vocab_idx_subsample_weights_min']

        self.pos_tag_vocab_idx_subsample_weights = d['pos_tag_vocab_idx_subsample_weights']
        self.pos_tag_vocab_idx_subsample_weights_max = d['pos_tag_vocab_idx_subsample_weights_max']
        self.pos_tag_vocab_idx_subsample_weights_min = d['pos_tag_vocab_idx_subsample_weights_min']

        self.affix_vocab_idx_subsample_weights = d['affix_vocab_idx_subsample_weights']
        self.affix_vocab_idx_subsample_weights_max = d['affix_vocab_idx_subsample_weights_max']
        self.affix_vocab_idx_subsample_weights_min = d['affix_vocab_idx_subsample_weights_min']

        self.reduced_stem_vocab_idx_subsample_weights = d['reduced_stem_vocab_idx_subsample_weights']
        self.reduced_stem_vocab_idx_subsample_weights_max = d['reduced_stem_vocab_idx_subsample_weights_max']
        self.reduced_stem_vocab_idx_subsample_weights_min = d['reduced_stem_vocab_idx_subsample_weights_min']

        for k in self.reduced_stem_vocab:
            self.reduced_stem_vocab_idx[self.reduced_stem_vocab[k]] = k
        self.reduced_stem_vocab_idx_counts = dict()
        for i in self._stem_vocab_idx_counts:
            self.reduced_stem_vocab_idx_counts[self.mapped_stem_vocab_idx[i]] = self._stem_vocab_idx_counts[i]

class AffixSetVocab:
    def __init__(self, reduced_affix_dict_file = None, reduced_affix_dict_map_file = None):
        self.affix_set_vocab_idx = dict()
        self.affix_set_vocab = dict()
        self.reduced_affix_dict_counts = dict()
        self.reduced_affix_dict_map = dict()

        if reduced_affix_dict_file is not None:
            f = open(reduced_affix_dict_file, 'r')
            dict_lines = [line.rstrip('\n') for line in f]
            f.close()
            idx = 1
            for l in dict_lines:
                if len(l) > 0:
                    spl = l.split(',')
                    if (len(spl) == 2):
                        self.reduced_affix_dict_counts[spl[0]] = int(spl[1])
                        self.affix_set_vocab_idx[spl[0]] = idx
                        self.affix_set_vocab[idx] = spl[0]
                        idx += 1

        if reduced_affix_dict_map_file is not None:
            f = open(reduced_affix_dict_map_file, 'r')
            dict_lines = [line.rstrip('\n') for line in f]
            f.close()
            for l in dict_lines:
                if len(l) > 0:
                    spl = l.split(',')
                    if (len(spl) == 2):
                        self.reduced_affix_dict_map[spl[0]] = spl[1]
                        if (spl[1]) in self.affix_set_vocab_idx:
                            self.affix_set_vocab_idx[spl[0]] = self.affix_set_vocab_idx[spl[1]]
                        else:
                            self.affix_set_vocab_idx[spl[0]] = 1

    def affix_set_to_idx(self, key):
        if key in self.affix_set_vocab_idx:
            return self.affix_set_vocab_idx[key]
        else:
            return 1 # N/A

    def affix_set_idx_to_txt(self, idx, kb_vocab: KBVocab):
        if idx == 1:
            return "N/A"
        elif idx in self.affix_set_vocab:
            return '-'.join([kb_vocab.affix_vocab_idx[int(x)] for x in self.affix_set_vocab[idx].split('-')])
        else:
            return "UNK" # N/A

    def random_idx(self):
        return random.randint(1, len(self.affix_set_vocab_idx))

    def state_dict(self):
        return {'affix_set_vocab_idx':self.affix_set_vocab_idx,
                'reduced_affix_dict_counts':self.reduced_affix_dict_counts,
                'reduced_affix_dict_map':self.reduced_affix_dict_map}

    def load_state_dict(self, d):
        self.affix_set_vocab_idx = d['affix_set_vocab_idx']
        self.reduced_affix_dict_counts = d['reduced_affix_dict_counts']
        self.reduced_affix_dict_map = d['reduced_affix_dict_map']

class ParsedToken:
    def __init__(self, surface_form, parsed_token=None, decode_prob=None, tf_idf=0.0, pos_tag_id=None, stem_ids=None, line_num=0):
        self.surface_form = surface_form
        self.tf_idf = tf_idf
        if parsed_token is not None:
            parts = parsed_token.split('/')
            self.decode_prob = float(parts[0])
            self.tf_idf = float(parts[1])
            morphs = parts[2].split(',')
            pos_stem = morphs[0].split(':')
            stem_parts = pos_stem[1].split('*')
            if(len(stem_parts[0]) < 1):
                # print('\nParsing wrong token: /{}/ at line # {}'.format(parsed_token, line_num))
                self.pos_tag_idx = int(pos_stem[0])
                self.stem_idx = [6]
                self.morpho_slots_idx = []
                self.affixes_idx = []
            else:
                self.pos_tag_idx = int(pos_stem[0])
                self.stem_idx = [int(v) for v in stem_parts]
                self.morpho_slots_idx = [int(morphs[i].split(':')[0]) for i in range(1, len(morphs))]
                self.affixes_idx = [int(morphs[i].split(':')[1]) for i in range(1, len(morphs))]
        else:
            self.decode_prob = decode_prob
            self.pos_tag_idx = pos_tag_id
            self.stem_idx = stem_ids
            self.morpho_slots_idx = []
            self.affixes_idx = []

    def append_morpheme(self, morpho_slot_id, affix_id):
        self.morpho_slots_idx.append(morpho_slot_id)
        self.affixes_idx.append(affix_id)

    def to_parsed_format(self):
        st = ['{}:{}'.format(self.pos_tag_idx,'*'.join([str(i) for i in self.stem_idx]))]
        for i in range(len(self.morpho_slots_idx)):
            st.append('{}:{}'.format(self.morpho_slots_idx[i], self.affixes_idx[i]))
        return '{:.3g}/{:.3g}/{}'.format(self.decode_prob, self.tf_idf, ','.join(st))

    def affix_set_key(self):
        key = '-'.join([str(af) for af in self.affixes_idx]) if (len(self.affixes_idx) > 0) else 'N/A'
        return key

def update_tf_idf_from_idf(doc_sentences):
    doc_voc = dict()
    doc_size = 0.0
    for parsed_tokens in doc_sentences:
        for t in parsed_tokens:
            doc_size += 1.0
            v = 0.0
            if(t.stem_idx[0] in doc_voc):
                v = doc_voc[t.stem_idx[0]]
            doc_voc[t.stem_idx[0]] = v + 1.0
    for parsed_tokens in doc_sentences:
        for t in parsed_tokens:
            t.tf_idf = sigmoid_score(t.tf_idf * doc_voc[t.stem_idx[0]] / doc_size, 0.01, 0.24)

def pre_process_parsed_corpus_compute_tfidf(input_corpus, output_corpus):
    f = open(input_corpus, 'r')
    Lines = f.readlines()
    f.close()

    outfile = open(output_corpus, 'w')
    doc_idx = [i for i in range(len(Lines)) if (len(Lines[i]) == 1)]
    if doc_idx[-1] < (len(Lines) - 1):
        doc_idx.append(len(Lines))
    start_idx = 0
    all_docs = len(doc_idx)
    print_docs = 0

    tot = 0
    for end_idx in doc_idx:
        tot = tot + 1
        lines_batch = Lines[start_idx:end_idx]
        start_idx = end_idx + 1
        if (len(lines_batch) > 0):
            doc_sentences = []
            for ln in lines_batch:
                line = ln.strip()
                line = line.strip('\n')
                line = line.strip('\t')
                line = line.strip('\r')
                splits = line.split('; ')
                if (len(splits) > 0):
                    if ((len(splits[0]) > 4) and ('/' in splits[0]) and (':' in splits[0])):
                        doc_sentences.append([ParsedToken('_', parsed_token=t) for t in splits])
            update_tf_idf_from_idf(doc_sentences)
            for sent in doc_sentences:
                outfile.write('; '.join([tok.to_parsed_format() for tok in sent]) + "\n")
            outfile.write("\n")
            outfile.flush()
            print_docs += 1
    outfile.close()
    print('Exported: {} / {} / {}'.format(print_docs, tot, all_docs))

def parse_raw_text_lines(doc_lines, kb_vocab, bpe):
    from kinlpmorpholib import ffi, lib
    parsed_tokens = []
    num_sent = ffi.new("int[1]")
    sentences = lib.parse_sentences_batch(doc_lines.encode('utf-8'), num_sent)
    for i in range(num_sent[0]):
        sent = sentences[i]
        for j in range(sent.words_len):
            w = sent.words[j]
            POS_TAG = ffi.string(w.pos_tag).decode("utf-8")
            WORD_TYPE = ffi.string(w.pos_group).decode("utf-8")
            STEM = ffi.string(w.stem).decode("utf-8")
            SURFACE_FORM = ffi.string(w.surface_form).decode("utf-8")
            DECODE_PROB = w.decode_prob
            TF_IDF = w.tf_idf
            pti = kb_vocab.pos_tag_vocab['<UNK>']
            if POS_TAG in kb_vocab.pos_tag_vocab.keys():
               pti = kb_vocab.pos_tag_vocab[POS_TAG]

            sids = []
            if ((STEM == SURFACE_FORM) and (w.morphemes_len <= 0)):
                list_sub_words = bpe.encode(SURFACE_FORM, output_type=yttm.OutputType.SUBWORD)
                for sub_word in list_sub_words:
                    stem_key = WORD_TYPE + ":" + sub_word
                    si = kb_vocab._stem_vocab['<UNK>']
                    if stem_key in kb_vocab._stem_vocab.keys():
                        si = kb_vocab._stem_vocab[stem_key]
                    sids.append(si)
            else:
                stem_key = WORD_TYPE + ":" + STEM
                si = kb_vocab._stem_vocab['<UNK>']
                if stem_key in kb_vocab._stem_vocab.keys():
                    si = kb_vocab._stem_vocab[stem_key]
                sids.append(si)
            ptoken = ParsedToken(SURFACE_FORM, parsed_token=None, decode_prob=DECODE_PROB, tf_idf=TF_IDF, pos_tag_id=pti, stem_ids=sids)
            if (w.morphemes_len > 0):
                for k in range(w.morphemes_len):
                    if ((k != w.stem_start_index) and (k != w.stem_end_index)):
                        MORPHEME_SLOT = WORD_TYPE + ":" + str(w.morphemes[k].slot_id)
                        MORPHEME = MORPHEME_SLOT + ":" + ffi.string(w.morphemes[k].morph_token).decode("utf-8")

                        msi = kb_vocab.morpheme_slot_vocab['<UNK>']
                        if MORPHEME_SLOT in kb_vocab.morpheme_slot_vocab.keys():
                            msi = kb_vocab.morpheme_slot_vocab[MORPHEME_SLOT]

                        mi = kb_vocab.affix_vocab['<UNK>']
                        if MORPHEME in kb_vocab.affix_vocab.keys():
                            mi = kb_vocab.affix_vocab[MORPHEME]

                        ptoken.append_morpheme(msi, mi)
            parsed_tokens.append(ptoken)
    lib.release_sentence(sentences, num_sent)

    return parsed_tokens

def process_parsed_sentence(args, parsed_tokens_list: List[ParsedToken], add_cls, kv : KBVocab, affix_set_vocab : AffixSetVocab, rel_pos_dict, rel_pos_dmax):
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
    # Add <CLS> Token
    if add_cls:
        pos_tags.append(kv.pos_tag_vocab['<CLS>'])
        stems.append(kv.reduced_stem_vocab['<CLS>'])
        if args.use_afsets:
            afsets.append(affix_set_vocab.affix_set_to_idx('<CLS>'))
        tokens_lengths.append(0)

    if (len(parsed_tokens_list) == 0): # New document
        pos_tags.append(kv.pos_tag_vocab['<SEP>'])
        stems.append(kv.reduced_stem_vocab['<SEP>'])
        if args.use_afsets:
            afsets.append(affix_set_vocab.affix_set_to_idx('<SEP>'))
        tokens_lengths.append(0)

    else:
        for ptoken in parsed_tokens_list:
            for sidx in ptoken.stem_idx:
                unchanged = True
                predict = False
                rval = random.random()
                if (rval <= 0.15): # 15% of tokens are predicted
                    predict = True
                    rval /= 0.15
                    if(rval < 0.8): # 80% of predicted tokens are masked
                        unchanged = False
                        pos_tags.append(kv.pos_tag_vocab['<MSK>'])
                        stems.append(kv.reduced_stem_vocab['<MSK>'])
                        if args.use_afsets:
                            afsets.append(affix_set_vocab.affix_set_to_idx('<MSK>'))

                        vv = rval/0.8
                        if vv < 0.3: # Include Affixes for 30% of the time to enforce morphology learning
                            affixes.extend([(v) for v in ptoken.affixes_idx])
                            tokens_lengths.append(len(ptoken.affixes_idx))
                        else:
                            tokens_lengths.append(0)
                    elif (rval < 0.9): # 10% are replaced by random tokens, 10% are left unchanged
                        unchanged = False
                        rnd_pos = random.randint(kv.pos_tag_vocab['<UNK>'], len(kv.pos_tag_vocab)-1)
                        rnd_stem = random.randint(kv.reduced_stem_vocab['<UNK>'], len(kv.reduced_stem_vocab)-1)

                        pos_tags.append(rnd_pos)
                        stems.append(rnd_stem)
                        if args.use_afsets:
                            afsets.append(affix_set_vocab.random_idx())
                        elif args.inference_model_file is not None:
                            affix_set_vocab.random_idx()
                        vv = rval/0.8
                        if vv < 0.3: # Include Affixes for 30% of the time to enforce morphology learning
                            affixes.extend([(v) for v in ptoken.affixes_idx])
                            tokens_lengths.append(len(ptoken.affixes_idx))
                        else:
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
                        if(len(ptoken.affixes_idx) > 0):
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

def gather_replicated_itemized_data(args, corpus_lines, doc_ends, is_corpus_parsed, max_seq_len, start_line, max_batch_items, kb_vocab : KBVocab, affix_set_vocab : AffixSetVocab, bpe: yttm.BPE, rel_pos_dict, rel_pos_dmax, rank=0, bar=None, num_lines=sys.maxsize,shuffle=False,are_parsed_tokens_split=False):
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

    # if (rank==0):
    #     print(time_now(), 'Gathering itemized input sequence data from line # {}'.format(start_line + 1), flush=True)
    lcount = 0
    while (True):
        lcount += 1
        add_cls = (len(seq_parsed_tokens) == 0)
        if add_cls:
            ptoken = ParsedToken('<CLS>', decode_prob=1.0, tf_idf=0.001, pos_tag_id=kb_vocab.pos_tag_vocab['<CLS>'], stem_ids=[kb_vocab.reduced_stem_vocab['<CLS>']])
            ptoken.append_morpheme(kb_vocab.morpheme_slot_vocab['<EOT>'], kb_vocab.affix_vocab['<EOT>'])
            seq_parsed_tokens.append(ptoken)

        line = corpus_lines[start_line % len(corpus_lines)]
        start_line = (start_line + 1) % len(corpus_lines)
        parsed_tokens_line = []
        if (is_corpus_parsed == True):
            if are_parsed_tokens_split:
                parsed_tokens_line = line
            else:
                splits = line.split('; ')
                if (len(splits) > 0):
                    if ((len(splits[0]) > 4) and ('/' in splits[0]) and (':' in splits[0])):
                        parsed_tokens_line = [ParsedToken('_', parsed_token=t, line_num=(start_line + 1)) for t in splits]
        else:
            parsed_tokens_line = parse_raw_text_lines(line, kb_vocab, bpe)

        if (len(parsed_tokens_line) == 0):
            ptoken = ParsedToken('<SEP>', decode_prob=1.0, tf_idf=0.001, pos_tag_id=kb_vocab.pos_tag_vocab['<SEP>'], stem_ids=[kb_vocab.reduced_stem_vocab['<SEP>']])
            ptoken.append_morpheme(kb_vocab.morpheme_slot_vocab['<EOT>'], kb_vocab.affix_vocab['<EOT>'])
            seq_parsed_tokens.append(ptoken)
        else:
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
         predicted_tokens_affixes_lengths) = process_parsed_sentence(args, parsed_tokens_line, add_cls, kb_vocab, affix_set_vocab, rel_pos_dict, rel_pos_dmax)
        if (len(seq_tokens_lengths) + len(tokens_lengths)) > max_seq_len:
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
            if (shuffle):
                dcx = random.randint(0, len(doc_ends) - 1) % len(doc_ends)
                start_line = (doc_ends[dcx]+1) % len(corpus_lines)
                if random.random() < 0.8: # 80% of the time, start from anywhere within the corpus.
                    start_line = random.randint(0, len(corpus_lines) - 1) % len(corpus_lines)

            if (len(itemized_data) >= max_batch_items):
                if ((rank == 0) and (bar is not None)):
                    bar.update(len(itemized_data))
                    sys.stdout.flush()
                return itemized_data, itemized_parsed_tokens, start_line

            if(lcount >= num_lines):
                return itemized_data, itemized_parsed_tokens, start_line

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

            if ((rank == 0) and ((len(itemized_data) % (math.floor(0.1 * max_batch_items) + 1)) == 0) and (bar is not None)):
                bar.update(len(itemized_data))
                sys.stdout.flush()
        else:
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

            if(lcount >= num_lines):
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
                return itemized_data, itemized_parsed_tokens, start_line

def morpho_seq_collate_wrapper(batch_items):
    batch_pos_tags = []
    batch_stems = []
    batch_afsets = []
    batch_affixes = []
    batch_tokens_lengths = []
    batch_predicted_stems = []
    batch_predicted_afsets = []
    batch_predicted_affixes = []
    batch_predicted_tokens_idx = []
    batch_predicted_tokens_affixes_idx = []
    batch_predicted_tokens_affixes_lengths = []

    batch_input_sequence_lengths = []

    max_sequence_len = batch_items[0][0]
    first_seq_rel_pos_arr = batch_items[0][1]

    batch_rel_pos_arr = np.zeros((len(batch_items), max_sequence_len, max_sequence_len)).astype(int) if (first_seq_rel_pos_arr is not None) else None

    for bidx,data_item in enumerate(batch_items):
        (max_seq_len,
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
         seq_predicted_tokens_affixes_lengths) = data_item

        if batch_rel_pos_arr is not None:
            batch_rel_pos_arr[bidx,:,:] = seq_rel_pos_arr
        # Need to offset from total number of predicted token indices
        if seq_predicted_tokens_affixes_idx is not None:
            batch_predicted_tokens_affixes_idx.extend([(len(batch_predicted_tokens_idx) + t) for t in seq_predicted_tokens_affixes_idx])

        batch_predicted_tokens_idx.extend([(t, len(batch_input_sequence_lengths)) for t in seq_predicted_tokens_idx])

        batch_pos_tags.extend(seq_pos_tags)
        batch_stems.extend(seq_stems)
        if seq_afsets is not None:
            batch_afsets.extend(seq_afsets)
        batch_affixes.extend(seq_affixes)
        batch_tokens_lengths.extend(seq_tokens_lengths)
        batch_predicted_stems.extend(seq_predicted_stems)
        if seq_predicted_afsets is not None:
            batch_predicted_afsets.extend(seq_predicted_afsets)
        if seq_predicted_affixes is not None:
            batch_predicted_affixes.extend(seq_predicted_affixes)
            batch_predicted_tokens_affixes_lengths.extend(seq_predicted_tokens_affixes_lengths)

        batch_input_sequence_lengths.append(len(seq_tokens_lengths))

    data_item = (batch_input_sequence_lengths,
                 batch_rel_pos_arr,
                 batch_pos_tags,
                 batch_stems,
                 batch_afsets,
                 batch_affixes,
                 batch_tokens_lengths,
                 batch_predicted_stems,
                 batch_predicted_afsets,
                 batch_predicted_affixes,
                 batch_predicted_tokens_idx,
                 batch_predicted_tokens_affixes_idx,
                 batch_predicted_tokens_affixes_lengths)
    return data_item

class KBCorpusDataset(Dataset):

    def __init__(self,args,
                 kb_vocab : KBVocab, affix_set_vocab : AffixSetVocab, bpe_encoder: yttm.BPE,
                 rel_pos_dict, rel_pos_dmax,
                 parsed_corpus_lines, doc_ends,  is_corpus_parsed,
                 start_line, max_batch_items,
                 max_seq_len = 512,
                 rank = 0):
        self.max_seq_len = max_seq_len
        self.start_line = start_line
        self.max_batch_items = max_batch_items
        if (rank==0):
            with progressbar.ProgressBar(max_value=max_batch_items, redirect_stdout=True) as bar:
                self.itemized_data, self.itemized_parsed_tokens, self.start_line = gather_replicated_itemized_data(args, parsed_corpus_lines, doc_ends,
                                                                                                                    is_corpus_parsed,
                                                                                                                    self.max_seq_len,
                                                                                                                    self.start_line,
                                                                                                                    self.max_batch_items,
                                                                                                                    kb_vocab, affix_set_vocab, bpe_encoder,
                                                                                                                    rel_pos_dict, rel_pos_dmax,
                                                                                                                    rank = rank, bar=bar,
                                                                                                                    shuffle = True)
        else:
            self.itemized_data, self.itemized_parsed_tokens, self.start_line = gather_replicated_itemized_data(args, parsed_corpus_lines, doc_ends,
                                                                                                                is_corpus_parsed,
                                                                                                                self.max_seq_len,
                                                                                                                self.start_line,
                                                                                                                self.max_batch_items,
                                                                                                                kb_vocab, affix_set_vocab, bpe_encoder,
                                                                                                                rel_pos_dict, rel_pos_dmax,
                                                                                                                rank=rank, bar=None,
                                                                                                                shuffle = True)
    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

from itertools import accumulate

from morpho_model import KinyaBERT

def morpho_model_forward(args, data_item, model : KinyaBERT, device, tot_num_affixes):
    (batch_input_sequence_lengths,
     batch_rel_pos_arr,
     batch_pos_tags,
     batch_stems,
     batch_afsets,
     batch_affixes,
     batch_tokens_lengths,
     batch_predicted_stems,
     batch_predicted_afsets,
     batch_predicted_affixes,
     batch_predicted_tokens_idx,
     batch_predicted_tokens_affixes_idx,
     batch_predicted_tokens_affixes_lengths) = data_item

    tokens_lengths = batch_tokens_lengths  # torch.tensor(batch_tokens_lengths).to(device)
    input_sequence_lengths = batch_input_sequence_lengths  # torch.tensor(batch_input_sequence_lengths).to(device)
    pos_tags = torch.tensor(batch_pos_tags).to(device)
    stems = torch.tensor(batch_stems).to(device)
    afsets = torch.tensor(batch_afsets).to(device) if args.use_afsets else None
    affixes = torch.tensor(batch_affixes).to(device)

    predicted_tokens_idx = torch.tensor([s * max(batch_input_sequence_lengths) + t for t, s in batch_predicted_tokens_idx]).to(device)
    predicted_tokens_affixes_idx = torch.tensor(batch_predicted_tokens_affixes_idx).to(device) if args.predict_affixes else None

    predicted_affixes_prob = None
    if args.predict_affixes:
        pred_affixes_list = [batch_predicted_affixes[x - y: x] for x, y in zip(accumulate(batch_predicted_tokens_affixes_lengths), batch_predicted_tokens_affixes_lengths)]
        afx_prob = torch.zeros(len(pred_affixes_list), tot_num_affixes)
        for i,lst in enumerate(pred_affixes_list):
            assert (len(lst) > 0)
            afx_prob[i,lst] = 1 / len(lst)
        predicted_affixes_prob = afx_prob.to(device)

    predicted_stems = torch.tensor(batch_predicted_stems).to(device)
    predicted_afsets = torch.tensor(batch_predicted_afsets).to(device) if args.use_afsets else None
    rel_pos_arr = torch.from_numpy(batch_rel_pos_arr).to(device) if (batch_rel_pos_arr is not None) else None

    return model(args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes,
                 predicted_tokens_idx,
                 predicted_tokens_affixes_idx,
                 predicted_stems,
                 predicted_afsets,
                 predicted_affixes_prob)

def morpho_model_seq_predict(args, data_item, model : KinyaBERT, device,
                     max_predict_affixes, proposed_stem_ids=None):
    (max_seq_len,
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
     seq_predicted_tokens_affixes_lengths) = data_item

    tokens_lengths = seq_tokens_lengths
    input_sequence_lengths = [len(seq_tokens_lengths)]
    pos_tags = torch.tensor(seq_pos_tags).to(device)
    stems = torch.tensor(seq_stems).to(device)
    afsets = torch.tensor(seq_afsets).to(device) if args.use_afsets else None
    affixes = torch.tensor(seq_affixes).to(device)

    seq_predicted_token_idx = torch.tensor(seq_predicted_tokens_idx).to(device)

    rel_pos_arr = torch.from_numpy(seq_rel_pos_arr).unsqueeze(0).to(device) if (seq_rel_pos_arr is not None) else None

    return model.predict(args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes,
                seq_predicted_token_idx,
                max_predict_affixes, proposed_stem_ids=proposed_stem_ids)
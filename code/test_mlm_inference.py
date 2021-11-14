from __future__ import print_function, division

import random
# Ignore warnings
import warnings
from datetime import datetime

import numpy as np
import torch
from tqdm.notebook import tqdm
from morpho_common import setup_common_args

warnings.filterwarnings("ignore")


def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# %%

from kinlpmorpho import build_kinlp_morpho_lib

build_kinlp_morpho_lib()

from kinlpmorpholib import lib

print(time_now(), 'KINLP Morpho library Ready!', flush=True)

# %%

conf = "/mnt/NVM/KINLP/data/kb_config_kinlp.conf"
lib.start_kinlp_lib(conf.encode('utf-8'))

print(time_now(), 'KINLP Ready!', flush=True)

# %%

import importlib
import morpho_data_loaders
import morpho_model

importlib.reload(morpho_data_loaders)
importlib.reload(morpho_model)

import youtokentome as yttm
from morpho_data_loaders import KBVocab, AffixSetVocab, morpho_model_seq_predict
from morpho_model import kinyabert_base
from morpho_data_loaders import gather_replicated_itemized_data

home_path = "/mnt/NVM/KINLP/"
USE_GPU = False

device = torch.device('cpu')
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using device: ', device, flush=True)

BPE_model_path = (home_path + "data/BPE-30k.mdl")
bpe_encoder = yttm.BPE(model=BPE_model_path)

kb_vocab = KBVocab()
kbvocab_state_dict_file_path = (home_path + "data/kb_vocab_state_dict_2021-02-07.pt")
kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

args = setup_common_args()

print('Vocab ready!', flush=True)

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def morpho_test_mlm_inference(args, key_name, parsed, kb_vocab, affix_set_vocab, bpe_encoder, kinya_bert_model, morpho_rel_pos_dict, morpho_rel_pos_dmax, num_iter=1):
    stem_acc = 0.0
    afset_acc = 0.0
    affix_acc = 0.0

    stem_sum_acc = 0.0
    afset_sum_acc = 0.0
    affix_sum_acc = 0.0


    for docNum in range(100):
        input_file = "/home/user/Desktop/Desk/sample_docs/"+str(docNum)+".txt"
        for _ in tqdm(range(num_iter)):
            f = open(input_file, 'r+')
            text_lines = [line.rstrip('\n') for line in f]
            doc_ends = [i for i in range(len(text_lines)) if (len(text_lines[i]) == 0)]
            f.close()
            itemized_data, itemized_parsed_tokens, _ = gather_replicated_itemized_data(args, text_lines, doc_ends, parsed, 512, 0,
                                                                                       1, kb_vocab, affix_set_vocab, bpe_encoder,
                                                                                       morpho_rel_pos_dict,
                                                                                       morpho_rel_pos_dmax,
                                                                                       num_lines=len(text_lines))
            for myidx, data_item in enumerate(itemized_data):
                seq_parsed_tokens = itemized_parsed_tokens[myidx]
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

                (stem_predictions, stem_predictions_prob, afset_predictions, afset_predictions_prob, affix_predictions) = morpho_model_seq_predict(args, data_item,
                                                                                                        kinya_bert_model,
                                                                                                        device, 10)

                if args.debug:
                    np.set_printoptions(precision=3, linewidth=np.inf)
                    print('\nstem_predictions:', stem_predictions.detach().numpy().tolist())
                    print('\nseq_predicted_stems:', seq_predicted_stems)
                    print('\nstem_predictions_prob:', stem_predictions_prob.detach().numpy())

                    if args.use_afsets:
                        print('\nafset_predictions:', afset_predictions.detach().numpy().tolist())
                        print('\nseq_predicted_afsets:', seq_predicted_afsets)
                        print('\nafset_predictions_prob:', afset_predictions_prob.detach().numpy())

                    print('\nseq_predicted_tokens_affixes_lengths:', seq_predicted_tokens_affixes_lengths)
                    print('\nseq_predicted_tokens_affixes_idx:', seq_predicted_tokens_affixes_idx)
                    print('\naffix_predictions:', affix_predictions)
                    print('\nseq_predicted_affixes:', seq_predicted_affixes, '\n')

                stems_count = 0
                affixes_count = 0
                affixes_total = 0
                afx = 0
                ptk = 0
                next_ptk_idx = 0
                ptoken = seq_parsed_tokens[0]
                for i in range(len(seq_tokens_lengths)):
                    # 1. Print Token
                    # Keep the same token until all stem_idx are handled
                    if (next_ptk_idx == i):
                        ptoken = seq_parsed_tokens[ptk]
                        next_ptk_idx = i + len(ptoken.stem_idx)
                        ptk += 1
                    if args.debug:
                        if args.use_afsets:
                            print('\n{} @ --> {} {} / {} {} ==> {}'.format(ptoken.surface_form,
                                                                      kb_vocab.pos_tag_vocab_idx[ptoken.pos_tag_idx],
                                                                      [kb_vocab.affix_vocab_idx[k] for k in ptoken.affixes_idx],
                                                                      ptoken.affix_set_key(),
                                                                      [kb_vocab.affix_vocab_idx[k] for k in ptoken.affixes_idx],
                                                                      [kb_vocab._stem_vocab_idx[k] for k in ptoken.stem_idx],
                                                                      [kb_vocab.reduced_stem_vocab_idx[
                                                                           kb_vocab.mapped_stem_vocab_idx[k]] for k in
                                                                       ptoken.stem_idx]))
                            # 1.5 Print input stem,pos & affixes
                            print('Input:', '{}/{}/{}'.format(kb_vocab.pos_tag_vocab_idx[seq_pos_tags[i]],
                                                           kb_vocab.reduced_stem_vocab_idx[seq_stems[i]],
                                                           affix_set_vocab.affix_set_idx_to_txt(seq_afsets[i], kb_vocab)),
                                  ['{}'.format(kb_vocab.affix_vocab_idx[a]) for a in
                                   seq_affixes[afx:(afx + seq_tokens_lengths[i])]])
                        else:
                            print('\n{} @ --> {} {} {} ==> {}'.format(ptoken.surface_form,
                                                                      kb_vocab.pos_tag_vocab_idx[ptoken.pos_tag_idx],
                                                                      [kb_vocab.affix_vocab_idx[k] for k in ptoken.affixes_idx],
                                                                      [kb_vocab._stem_vocab_idx[k] for k in ptoken.stem_idx],
                                                                      [kb_vocab.reduced_stem_vocab_idx[
                                                                           kb_vocab.mapped_stem_vocab_idx[k]] for k in
                                                                       ptoken.stem_idx]))
                            # 1.5 Print input stem,pos & affixes
                            print('Input:', '{}/{}'.format(kb_vocab.pos_tag_vocab_idx[seq_pos_tags[i]],
                                                           kb_vocab.reduced_stem_vocab_idx[seq_stems[i]]),
                                  ['{}'.format(kb_vocab.affix_vocab_idx[a]) for a in
                                   seq_affixes[afx:(afx + seq_tokens_lengths[i])]])

                    afx += seq_tokens_lengths[i]

                    # 2. Stem Prediction
                    if i in seq_predicted_tokens_idx:
                        pstem = kb_vocab.reduced_stem_vocab_idx[seq_predicted_stems[stems_count]]
                        _pstem = kb_vocab.reduced_stem_vocab_idx[stem_predictions[stems_count].item()]
                        _pstem_prob = stem_predictions_prob[stems_count].item()
                        if args.debug:
                            print('{} [STEM]>>>> Gold: {} --> Pred: {} @ {:.3}'.format((pstem == _pstem), pstem, _pstem,
                                                                                       _pstem_prob))
                        stem_sum_acc += 1
                        if (pstem == _pstem):
                            stem_acc += 1

                        if (seq_predicted_afsets is not None) and args.use_afsets:
                            pafset = affix_set_vocab.affix_set_idx_to_txt(seq_predicted_afsets[stems_count], kb_vocab)
                            _pafset = affix_set_vocab.affix_set_idx_to_txt(afset_predictions[stems_count].item(), kb_vocab)
                            _pafset_prob = afset_predictions_prob[stems_count].item()
                            if args.debug:
                                print(
                                    '{} [AFSET]>>>> Gold: {} --> Pred: {} @ {:.3}'.format((pafset == _pafset), pafset, _pafset,
                                                                                          _pafset_prob))
                            afset_sum_acc += 1
                            if (pafset == _pafset):
                                afset_acc += 1

                        # 3. Affix Prediction
                        if args.predict_affixes:
                            if stems_count in seq_predicted_tokens_affixes_idx:
                                flen = seq_predicted_tokens_affixes_lengths[affixes_count]
                                paffixes = set([kb_vocab.affix_vocab_idx[a] for a in
                                                seq_predicted_affixes[affixes_total:(affixes_total + flen)]])
                                _paffixes = set([kb_vocab.affix_vocab_idx[a] for a in affix_predictions[affixes_count]])
                                incr = float(len(paffixes.intersection(_paffixes))) / float(len(paffixes))
                                if args.debug:
                                    print('{} @ {:.2f} [AFFIX]>>> Gold: {} --> Pred: {}'.format((paffixes == _paffixes), incr,
                                                                                                paffixes, _paffixes))
                                affixes_count += 1
                                affixes_total += flen
                                affix_sum_acc += 1
                                affix_acc += incr
                                # if (paffixes == _paffixes):
                                #     affix_acc += 1
                        stems_count += 1
                if args.debug:
                    print(
                        '\n-------------------------------------------------------------------- NEW SEQUENCE ---------------------------------------------------------------------\n')
    print(key_name, 'LM ACCURACY:>> STEM: {:.2f}%({:.0f}/{:.0f}) \t AFSET: {:.2f}%({:.0f}/{:.0f}) \t AFFIX: {:.2f}%({:.2f}/{:.2f})'.format(
        (100.0 * stem_acc / (stem_sum_acc + 1e-7)), stem_acc, stem_sum_acc,
        (100.0 * afset_acc / (afset_sum_acc + 1e-7)), afset_acc, afset_sum_acc,
        100.0 * affix_acc / (affix_sum_acc + 1e-7), affix_acc, affix_sum_acc), flush=True)

def better_mlm_inference_run(args, key_name, parsed_data, parsed_data_doc_ends, kb_vocab, affix_set_vocab, bpe_encoder, kinya_bert_model, morpho_rel_pos_dict, morpho_rel_pos_dmax, num_iter=1):
    import progressbar
    stem_acc = 0.0
    afset_acc = 0.0
    affix_acc = 0.0

    stem_sum_acc = 0.0
    afset_sum_acc = 0.0
    affix_sum_acc = 0.0

    progressCount = 0
    with progressbar.ProgressBar(max_value=(args.num_inference_runs * len(parsed_data)), redirect_stdout=True) as bar:
        bar.update(progressCount)
        for _i_ in range(args.num_inference_runs):
            for di, parsed_tokens_lines in enumerate(parsed_data):
                itemized_data, itemized_parsed_tokens, _ = gather_replicated_itemized_data(args, parsed_tokens_lines,
                                                                                           parsed_data_doc_ends[di],
                                                                                           True, 512, 0,
                                                                                           1, kb_vocab, affix_set_vocab,
                                                                                           bpe_encoder,
                                                                                           morpho_rel_pos_dict,
                                                                                           morpho_rel_pos_dmax,
                                                                                           num_lines=len(parsed_tokens_lines),
                                                                                           are_parsed_tokens_split=True)
                for myidx, data_item in enumerate(itemized_data):
                    seq_parsed_tokens = itemized_parsed_tokens[myidx]
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

                    (stem_predictions, stem_predictions_prob, afset_predictions, afset_predictions_prob, affix_predictions) = morpho_model_seq_predict(args, data_item,
                                                                                                            kinya_bert_model,
                                                                                                            device, 10)

                    if args.debug:
                        np.set_printoptions(precision=3, linewidth=np.inf)
                        print('\nstem_predictions:', stem_predictions.detach().numpy().tolist())
                        print('\nseq_predicted_stems:', seq_predicted_stems)
                        print('\nstem_predictions_prob:', stem_predictions_prob.detach().numpy())

                        if args.use_afsets:
                            print('\nafset_predictions:', afset_predictions.detach().numpy().tolist())
                            print('\nseq_predicted_afsets:', seq_predicted_afsets)
                            print('\nafset_predictions_prob:', afset_predictions_prob.detach().numpy())

                        print('\nseq_predicted_tokens_affixes_lengths:', seq_predicted_tokens_affixes_lengths)
                        print('\nseq_predicted_tokens_affixes_idx:', seq_predicted_tokens_affixes_idx)
                        print('\naffix_predictions:', affix_predictions)
                        print('\nseq_predicted_affixes:', seq_predicted_affixes, '\n')

                    stems_count = 0
                    affixes_count = 0
                    affixes_total = 0
                    afx = 0
                    ptk = 0
                    next_ptk_idx = 0
                    ptoken = seq_parsed_tokens[0]
                    for i in range(len(seq_tokens_lengths)):
                        # 1. Print Token
                        # Keep the same token until all stem_idx are handled
                        if (next_ptk_idx == i):
                            ptoken = seq_parsed_tokens[ptk]
                            next_ptk_idx = i + len(ptoken.stem_idx)
                            ptk += 1
                        if args.debug:
                            if args.use_afsets:
                                print('\n{} @ --> {} {} / {} {} ==> {}'.format(ptoken.surface_form,
                                                                          kb_vocab.pos_tag_vocab_idx[ptoken.pos_tag_idx],
                                                                          [kb_vocab.affix_vocab_idx[k] for k in ptoken.affixes_idx],
                                                                          ptoken.affix_set_key(),
                                                                          [kb_vocab.affix_vocab_idx[k] for k in ptoken.affixes_idx],
                                                                          [kb_vocab._stem_vocab_idx[k] for k in ptoken.stem_idx],
                                                                          [kb_vocab.reduced_stem_vocab_idx[
                                                                               kb_vocab.mapped_stem_vocab_idx[k]] for k in
                                                                           ptoken.stem_idx]))
                                # 1.5 Print input stem,pos & affixes
                                print('Input:', '{}/{}/{}'.format(kb_vocab.pos_tag_vocab_idx[seq_pos_tags[i]],
                                                               kb_vocab.reduced_stem_vocab_idx[seq_stems[i]],
                                                               affix_set_vocab.affix_set_idx_to_txt(seq_afsets[i], kb_vocab)),
                                      ['{}'.format(kb_vocab.affix_vocab_idx[a]) for a in
                                       seq_affixes[afx:(afx + seq_tokens_lengths[i])]])
                            else:
                                print('\n{} @ --> {} {} {} ==> {}'.format(ptoken.surface_form,
                                                                          kb_vocab.pos_tag_vocab_idx[ptoken.pos_tag_idx],
                                                                          [kb_vocab.affix_vocab_idx[k] for k in ptoken.affixes_idx],
                                                                          [kb_vocab._stem_vocab_idx[k] for k in ptoken.stem_idx],
                                                                          [kb_vocab.reduced_stem_vocab_idx[
                                                                               kb_vocab.mapped_stem_vocab_idx[k]] for k in
                                                                           ptoken.stem_idx]))
                                # 1.5 Print input stem,pos & affixes
                                print('Input:', '{}/{}'.format(kb_vocab.pos_tag_vocab_idx[seq_pos_tags[i]],
                                                               kb_vocab.reduced_stem_vocab_idx[seq_stems[i]]),
                                      ['{}'.format(kb_vocab.affix_vocab_idx[a]) for a in
                                       seq_affixes[afx:(afx + seq_tokens_lengths[i])]])

                        afx += seq_tokens_lengths[i]

                        # 2. Stem Prediction
                        if i in seq_predicted_tokens_idx:
                            pstem = kb_vocab.reduced_stem_vocab_idx[seq_predicted_stems[stems_count]]
                            _pstem = kb_vocab.reduced_stem_vocab_idx[stem_predictions[stems_count].item()]
                            _pstem_prob = stem_predictions_prob[stems_count].item()
                            if args.debug:
                                print('{} [STEM]>>>> Gold: {} --> Pred: {} @ {:.3}'.format((pstem == _pstem), pstem, _pstem,
                                                                                           _pstem_prob))
                            stem_sum_acc += 1
                            if (pstem == _pstem):
                                stem_acc += 1

                            if (seq_predicted_afsets is not None) and args.use_afsets:
                                pafset = affix_set_vocab.affix_set_idx_to_txt(seq_predicted_afsets[stems_count], kb_vocab)
                                _pafset = affix_set_vocab.affix_set_idx_to_txt(afset_predictions[stems_count].item(), kb_vocab)
                                _pafset_prob = afset_predictions_prob[stems_count].item()
                                if args.debug:
                                    print(
                                        '{} [AFSET]>>>> Gold: {} --> Pred: {} @ {:.3}'.format((pafset == _pafset), pafset, _pafset,
                                                                                              _pafset_prob))
                                afset_sum_acc += 1
                                if (pafset == _pafset):
                                    afset_acc += 1

                            # 3. Affix Prediction
                            if args.predict_affixes:
                                if stems_count in seq_predicted_tokens_affixes_idx:
                                    flen = seq_predicted_tokens_affixes_lengths[affixes_count]
                                    paffixes = set([kb_vocab.affix_vocab_idx[a] for a in
                                                    seq_predicted_affixes[affixes_total:(affixes_total + flen)]])
                                    _paffixes = set([kb_vocab.affix_vocab_idx[a] for a in affix_predictions[affixes_count]])
                                    incr = float(len(paffixes.intersection(_paffixes))) / float(len(paffixes))
                                    if args.debug:
                                        print('{} @ {:.2f} [AFFIX]>>> Gold: {} --> Pred: {}'.format((paffixes == _paffixes), incr,
                                                                                                    paffixes, _paffixes))
                                    affixes_count += 1
                                    affixes_total += flen
                                    affix_sum_acc += 1
                                    affix_acc += incr
                                    # if (paffixes == _paffixes):
                                    #     affix_acc += 1
                            stems_count += 1
                progressCount += 1
                bar.update(progressCount)
    print('\n\n', time_now(), key_name, 'MLM ACCURACY:>> STEM: {:.2f}%({:.0f}/{:.0f}) \t AFSET: {:.2f}%({:.0f}/{:.0f}) \t AFFIX: {:.2f}%({:.2f}/{:.2f})'.format(
        (100.0 * stem_acc / (stem_sum_acc + 1e-7)), stem_acc, stem_sum_acc,
        (100.0 * afset_acc / (afset_sum_acc + 1e-7)), afset_acc, afset_sum_acc,
        100.0 * affix_acc / (affix_sum_acc + 1e-7), affix_acc, affix_sum_acc), '\n\n', flush=True)


print(time_now(), 'Functions Ready!', flush=True)

# %%
def MLM_main_run(args, parsed_data, parsed_data_doc_ends):
    from torch.nn.parallel import DistributedDataParallel as DDP

    print('\nPROCESSING ', args.model_keyword, ' ...', '\n', flush=True)
    affix_set_vocab = AffixSetVocab(
        reduced_affix_dict_file=home_path + "data/reduced_affix_dict_" + str(args.afset_dict_size) + ".csv",
        reduced_affix_dict_map_file=home_path + "data/reduced_affix_dict_map_" + str(args.afset_dict_size) + ".csv")

    args.world_size = args.gpus
    args.num_pos_m_embeddings = args.pos
    args.num_stem_m_embeddings = args.stem
    args.use_affix_bow_m_embedding = args.use_affix_bow
    args.use_pos_aware_rel_pos_bias = args.use_pos_aware_rel
    args.use_tupe_rel_pos_bias = args.use_tupe_rel
    args.num_inference_iters = args.inference_iters
    args.num_inference_runs = args.inference_runs

    morpho_rel_pos_dict = None
    morpho_rel_pos_dmax = 5
    if args.use_pos_aware_rel_pos_bias:
        morpho_rel_pos_dict_file_path = (home_path+"data/morpho_rel_pos_dict_2021-03-24.pt")
        saved_pos_rel_dict = torch.load(morpho_rel_pos_dict_file_path)
        morpho_rel_pos_dict = saved_pos_rel_dict['morpho_rel_pos_dict']
        morpho_rel_pos_dmax = saved_pos_rel_dict['morpho_rel_pos_dmax']

    kb_model = kinyabert_base(kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                           device, args, saved_model_file=None)
    ddp_model = DDP(kb_model)

    args.num_inference_runs = 10
    # for iter in range(args.num_inference_iters):
    rand_val = 349 # random.randint(1, 1000)
    kb_state_dict = torch.load(args.inference_model_file, map_location=device)
    ddp_model.load_state_dict(kb_state_dict['model_state_dict'])
    kb_model = ddp_model.module
    kb_model.eval()

    set_random_seeds(random_seed=rand_val)

    # morpho_test_mlm_inference(args, args.model_keyword, False,
    #                           kb_vocab, affix_set_vocab, bpe_encoder, kb_model, morpho_rel_pos_dict, morpho_rel_pos_dmax, num_iter=args.num_inference_runs)

    better_mlm_inference_run(args, args.model_keyword, parsed_data, parsed_data_doc_ends,
                              kb_vocab, affix_set_vocab, bpe_encoder, kb_model, morpho_rel_pos_dict, morpho_rel_pos_dmax, num_iter=args.num_inference_runs)

def test_all_MLM():
    from morpho_common import setup_common_args
    from morpho_data_loaders import parse_raw_text_lines
    import progressbar

    print(time_now(), 'Parsing data ...', flush=True)
    parsed_data = []
    parsed_data_doc_ends = []
    progressCount = 0
    with progressbar.ProgressBar(max_value=1000, redirect_stdout=True) as bar:
        bar.update(progressCount)
        for docNum in range(1000):
            input_file = "/home/user/Desktop/Desk/sample_docs/"+str(docNum)+".txt"
            f = open(input_file, 'r+')
            text_lines = [line.rstrip('\n') for line in f]
            doc_ends = [i for i in range(len(text_lines)) if (len(text_lines[i]) == 0)]
            f.close()
            parsed_tokens_lines = [parse_raw_text_lines(line, kb_vocab, bpe_encoder) for line in text_lines]
            parsed_data.append(parsed_tokens_lines)
            parsed_data_doc_ends.append(doc_ends)
            progressCount += 1
            bar.update(progressCount)

    print(time_now(), 'Done parsing data!', flush=True)

    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-04-23'
    # args.gpus = 0
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = True
    # args.use_tupe_rel=False
    # args.use_afsets=False
    # args.predict_affixes=True
    # args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_04_23_morpho_attentive_model_base_2021-04-19.pt'
    # MLM_main_run(args, parsed_data, parsed_data_doc_ends)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-tupe-07-26'
    # args.gpus = 0
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = False
    # args.predict_affixes = True
    # args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_07_26_morpho_attentive_model_base_2021-07-12.pt'
    # MLM_main_run(args, parsed_data, parsed_data_doc_ends)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-tupe-bowm-08-12'
    # args.gpus = 0
    # args.pos = 2
    # args.stem = 1
    # args.use_affix_bow = True
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = False
    # args.predict_affixes = True
    # args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_08_12_exploratory_kinyabert_model_2021-07-30_pos:2_stem:1_mbow:True_pawrel:False_tuperel:True.pt'
    # MLM_main_run(args, parsed_data, parsed_data_doc_ends)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-afset-tupe-08-29'
    # args.gpus = 0
    # args.pos = 2
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = True
    # args.predict_affixes = False
    # args.afset_dict_size = 10000
    # args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_08_29_exploratory_kinyabert_model_2021-08-15_pos@2_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True_predaffixes@False.pt'
    # MLM_main_run(args, parsed_data, parsed_data_doc_ends)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-afset-34008-tupe-09-29'
    # args.gpus = 0
    # args.pos = 2
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = True
    # args.predict_affixes = False
    # args.afset_dict_size = 34008
    # args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_09_29_exploratory_kinyabert_model_2021-08-29_pos@2_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True@34008_predaffixes@False.pt'
    # MLM_main_run(args, parsed_data, parsed_data_doc_ends)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-05-14'
    # args.gpus = 0
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = True
    # args.use_tupe_rel=False
    # args.use_afsets=False
    # args.predict_affixes=True
    # args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_05_14_morpho_attentive_model_base_2021-04-19.pt'
    # MLM_main_run(args, parsed_data, parsed_data_doc_ends)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-tupe-08-22'
    # args.gpus = 0
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = False
    # args.predict_affixes = True
    # args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_08_22_morpho_attentive_model_base_2021-07-12.pt'
    # MLM_main_run(args, parsed_data, parsed_data_doc_ends)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-07-24'
    # args.gpus = 0
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = True
    # args.use_tupe_rel=False
    # args.use_afsets=False
    # args.predict_affixes=True
    # args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_07_24_morpho_attentive_model_base_2021-04-19.pt'
    # MLM_main_run(args, parsed_data, parsed_data_doc_ends)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-08-27'
    # args.gpus = 0
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = True
    # args.use_tupe_rel=False
    # args.use_afsets=False
    # args.predict_affixes=True
    # args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_08_27_morpho_attentive_model_base_2021-04-19.pt'
    # MLM_main_run(args, parsed_data, parsed_data_doc_ends)

    args = setup_common_args()
    args.model_keyword = 'kinyabert-tupe-36K'
    args.gpus = 0
    args.pos = 3
    args.stem = 1
    args.use_affix_bow = False
    args.use_pos_aware_rel = False
    args.use_tupe_rel = True
    args.use_afsets = False
    args.predict_affixes = True
    args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/morpho_attentive_model_base_2021-07-12.pt'
    MLM_main_run(args, parsed_data, parsed_data_doc_ends)

    args = setup_common_args()
    args.model_keyword = 'kinyabert-afset-38K'
    args.gpus = 0
    args.pos = 2
    args.stem = 1
    args.use_affix_bow = False
    args.use_pos_aware_rel = False
    args.use_tupe_rel = True
    args.use_afsets = True
    args.predict_affixes = False
    args.afset_dict_size = 34008
    args.inference_model_file = '/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/exploratory_kinyabert_model_2021-08-29_pos@2_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True@34008_predaffixes@False.pt'
    MLM_main_run(args, parsed_data, parsed_data_doc_ends)

if __name__ == '__main__':
    import os
    import torch.distributed as dist

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '88599'
    dist.init_process_group(backend='gloo', init_method='env://', world_size=1, rank=0)

    test_all_MLM()

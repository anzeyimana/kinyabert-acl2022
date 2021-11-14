
from __future__ import print_function, division
import torch

def count_params(args):
    from morpho_data_loaders import KBVocab, AffixSetVocab
    from morpho_model import kinyabert_base

    args.world_size = args.gpus
    args.num_pos_m_embeddings = args.pos
    args.num_stem_m_embeddings = args.stem
    args.use_affix_bow_m_embedding = args.use_affix_bow
    args.use_pos_aware_rel_pos_bias = args.use_pos_aware_rel
    args.use_tupe_rel_pos_bias = args.use_tupe_rel
    args.num_inference_iters = args.inference_iters
    args.num_inference_runs = args.inference_runs

    device = torch.device('cpu')

    home_path = args.home_path

    kb_vocab = KBVocab()
    kbvocab_state_dict_file_path = (home_path+"data/kb_vocab_state_dict_2021-02-07.pt")
    kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

    affix_set_vocab = None
    if args.use_afsets:
        affix_set_vocab = AffixSetVocab(reduced_affix_dict_file=home_path+"data/reduced_affix_dict_"+str(args.afset_dict_size)+".csv",
                                    reduced_affix_dict_map_file=home_path+"data/reduced_affix_dict_map_"+str(args.afset_dict_size)+".csv")
    morpho_rel_pos_dict = None
    if args.use_pos_aware_rel_pos_bias:
        morpho_rel_pos_dict_file_path = (home_path+"data/morpho_rel_pos_dict_2021-03-24.pt")
        saved_pos_rel_dict = torch.load(morpho_rel_pos_dict_file_path)
        morpho_rel_pos_dict = saved_pos_rel_dict['morpho_rel_pos_dict']

    print("args.morpho_dim",args.morpho_dim)
    print("args.stem_dim",args.stem_dim)
    model = kinyabert_base(kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                           device, args, saved_model_file=None)
    total_params = sum(p.numel() for p in model.parameters())
    print(args.model_keyword,'Params: ', total_params)

def count_all_models():
    from morpho_common import setup_common_args

    args = setup_common_args()
    args.model_keyword = 'kinyabert-afset34008-32K'
    args.gpus = 1
    args.pos = 2
    args.stem = 1
    args.use_affix_bow = False
    args.use_pos_aware_rel = False
    args.use_tupe_rel = True
    args.use_afsets = True
    args.afset_dict_size = 34008
    args.predict_affixes = False
    args.pretrained_model_file = '/home/user/KINLP/data/backup_10_18_exploratory_kinyabert_model_2021-08-29_pos@2_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True@34008_predaffixes@False.pt'
    count_params(args)

    args = setup_common_args()
    args.model_keyword = 'kinyabert-tupe-08-22'
    args.gpus = 1
    args.pos = 3
    args.stem = 1
    args.use_affix_bow = False
    args.use_pos_aware_rel = False
    args.use_tupe_rel = True
    args.use_afsets = False
    args.predict_affixes = True
    args.pretrained_model_file = '/home/user/KINLP/data/backup_08_22_morpho_attentive_model_base_2021-07-12.pt'
    count_params(args)

    args = setup_common_args()
    args.model_keyword = 'kinyabert-stem-32K'
    args.gpus = 1
    args.pos = 0
    args.stem = 0
    args.use_affix_bow = False
    args.use_pos_aware_rel = False
    args.use_tupe_rel = True
    args.use_afsets = False
    args.predict_affixes = False
    args.use_morpho_encoder = False
    args.stem_dim = 768
    args.pretrained_model_file = '/home/user/KINLP/data/backup_10_23_exploratory_kinyabert_model_2021-09-29_pos@0_stem@0_mbow@False_pawrel@False_tuperel@True_afsets@False@34008_predaffixes@False_morpho@False.pt'
    count_params(args)

    args = setup_common_args()
    args.model_keyword = 'kinyabert-tupe-bowm-08-12'
    args.gpus = 1
    args.pos = 2
    args.stem = 1
    args.use_affix_bow = True
    args.use_pos_aware_rel = False
    args.use_tupe_rel = True
    args.use_afsets = False
    args.predict_affixes = True
    args.pretrained_model_file = '/home/user/KINLP/data/backup_08_12_exploratory_kinyabert_model_2021-07-30_pos:2_stem:1_mbow:True_pawrel:False_tuperel:True.pt'
    count_params(args)

    args = setup_common_args()
    args.model_keyword = 'kinyabert-04-23'
    args.gpus = 1
    args.pos = 3
    args.stem = 1
    args.use_affix_bow = False
    args.use_pos_aware_rel = True
    args.use_tupe_rel=False
    args.use_afsets=False
    args.predict_affixes=True
    args.pretrained_model_file = '/home/user/KINLP/data/backup_04_23_morpho_attentive_model_base_2021-04-19.pt'
    count_params(args)

    args = setup_common_args()
    args.model_keyword = 'kinyabert-afset-tupe-08-29'
    args.gpus = 1
    args.pos = 2
    args.stem = 1
    args.use_affix_bow = False
    args.use_pos_aware_rel = False
    args.use_tupe_rel = True
    args.use_afsets = True
    args.predict_affixes = False
    args.pretrained_model_file = '/home/user/KINLP/data/backup_08_29_exploratory_kinyabert_model_2021-08-15_pos@2_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True_predaffixes@False.pt'
    count_params(args)


if __name__ == '__main__':
    count_all_models()


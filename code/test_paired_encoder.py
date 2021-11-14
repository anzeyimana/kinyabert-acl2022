from __future__ import print_function, division

import os
import os.path
import random
from datetime import datetime

import numpy as np
import psutil
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def date_now():
    return datetime.now().strftime("%Y-%m-%d")

def cleanup_dist():
    dist.destroy_process_group()

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def test_fn(rank, args):
    from morpho_data_loaders import KBVocab, AffixSetVocab
    from morpho_model import kinyabert_base

    USE_GPU = (args.gpus > 0)

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # if USE_GPU and torch.cuda.is_available():
    #     dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    #     torch.cuda.set_device(rank)
    # else:
    #     dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)

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

    if (rank == 0):
        print('Vocab ready!')

    model = kinyabert_base(kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                           device, args, saved_model_file = None)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if (rank == 0):
        print('---------------------------------- Model Size ----------------------------------------')
        print(time_now(), 'Total params:', total_params, 'Trainable params:', trainable_params, psutil.virtual_memory(), flush=True)
        print('---------------------------------------------------------------------------------------')

def main():
    from morpho_common import setup_common_args
    args = setup_common_args()
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '88599'
    # if args.gpus > 0:
    #     mp.spawn(test_fn, nprocs=args.gpus, args=(args,))
    # else:
    args.world_size = 1
    test_fn(0,args)

if __name__ == '__main__':
    from kinlpmorpho import build_kinlp_morpho_lib
    build_kinlp_morpho_lib()
    main()

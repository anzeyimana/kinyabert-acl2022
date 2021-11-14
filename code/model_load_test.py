from __future__ import print_function, division
import torch
import random
import numpy as np
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

from kinlpmorpho import build_kinlp_morpho_lib

build_kinlp_morpho_lib()

# from kinlpmorpholib import ffi, lib

# print(time_now(), 'KINLP Morpho library Ready!', flush=True)
#
# # %%
#
# conf = "/mnt/NVM/KINLP/data/kb_config_kinlp.conf"
# lib.start_kinlp_lib(conf.encode('utf-8'))
#
# print(time_now(), 'KINLP Ready!', flush=True)

# %%

from morpho_data_loaders import KBVocab
from morpho_model import kinyabert_base

device = torch.device('cpu')

home_path = "/mnt/NVM/KINLP/"
kb_vocab = KBVocab()
kbvocab_state_dict_file_path = (home_path + "data/kb_vocab_state_dict_2021-02-07.pt")
kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

# %%
def main_run(args):
    import os
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '88599'
    dist.init_process_group(backend='gloo', init_method='env://', world_size=1, rank=0)

    morpho_rel_pos_dict = None
    if args.use_pos_aware_rel_pos_bias:
        morpho_rel_pos_dict_file_path = (home_path+"data/morpho_rel_pos_dict_2021-03-24.pt")
        saved_pos_rel_dict = torch.load(morpho_rel_pos_dict_file_path)
        morpho_rel_pos_dict = saved_pos_rel_dict['morpho_rel_pos_dict']

    kb_model = kinyabert_base(kb_vocab, morpho_rel_pos_dict,
                           device, args, saved_model_file=None)
    ddp_model = DDP(kb_model)

    # kb_state_dict = torch.load("/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_07_01_morpho_attentive_model_base_2021-04-19.pt", map_location=device)
    kb_state_dict = torch.load("/mnt/NVM/KinyaBERT_Checkpoints/kb_attentive/backup_tupe_07_15_morpho_attentive_model_base_2021-07-12.pt", map_location=device)
    ddp_model.load_state_dict(kb_state_dict['model_state_dict'])
    kb_model = ddp_model.module
    kb_model.eval()

    print('Successfully Loaded model!')
    exit()

if __name__ == '__main__':
    from morpho_common import setup_common_args
    main_run(setup_common_args())


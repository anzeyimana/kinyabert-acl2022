#
# Next run: Paired Transformer Encoder
#
# nohup python3 train_exploratory_distributed_model.py -g 1 -p 0 -s 1 --morpho-dim=384 --stem-dim=384 --embed-dim=768 --paired-encoder=true --seq-tr-nhead=8 --seq-tr-nlayers=8 --seq-tr-dim-feedforward=2048 --use-afsets=1 --afset-dict-size=34008 --predict-affixes=0 --use-affix-bow=0 --use-pos-aware-rel=0 --use-tupe-rel=1 --batch-size=16 --accumulation-steps=160 --number-of-load-batches=3200 --num-iters=200000 --warmup-iter=2000 --peak-lr=0.0004 --wd=0.01 &> log.out & tail -f log.out
#
from __future__ import print_function, division
import os
import torch
from torch.utils.data import DataLoader
from morpho_learning_rates import AnnealingLR

import torch.multiprocessing as mp

import numpy as np
import random

import psutil

import gc

import progressbar

from datetime import datetime

import math

import youtokentome as yttm

import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP
from shutil import copyfile
import os.path

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

def train_loop(args, rank, scaler, device, data_loder, start_line, tot_num_affixes, model, optimizer, lr_scheduler, save_file_path, accumulation_steps, loop, num_loops, bar, total_steps, total_loss, stem_loss, affix_loss, afset_loss):
    from morpho_data_loaders import morpho_model_forward
    for batch_idx, data_item in enumerate(data_loder):
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss, stem_loss_avg, afset_loss_avg, affix_loss_avg = morpho_model_forward(args, data_item, model, device, tot_num_affixes)
                loss = loss / accumulation_steps  # Normalize our loss (if averaged)
            scaler.scale(loss).backward()
        else:
            loss, stem_loss_avg, afset_loss_avg, affix_loss_avg = morpho_model_forward(args, data_item, model, device, tot_num_affixes)
            loss = loss / accumulation_steps  # Normalize our loss (if averaged)
            loss.backward()

        total_loss += loss.item()
        stem_loss += stem_loss_avg.item() / accumulation_steps
        affix_loss += affix_loss_avg.item() / accumulation_steps
        afset_loss += afset_loss_avg.item() / accumulation_steps

        total_steps += 1
        if (total_steps % accumulation_steps) == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            lr_scheduler.step()
            # Log
            if (rank == 0):
                print(time_now(), 'Loop:', '{}/{}'.format(loop,num_loops), 'Batch:', '{}/{}'.format(batch_idx+1,len(data_loder)), 'Batch size:', len(data_item[0]),
                      'TOTAL Loss: ', "{:.4f}".format(total_loss),
                      'STEM Loss: ', "{:.4f}".format(stem_loss),
                      'AFFIX Loss: ', "{:.4f}".format(affix_loss),
                      'AFSET Loss: ', "{:.4f}".format(afset_loss),
                      'Learning rate: ', "{:.8f}".format(lr_scheduler.get_lr()),
                      'Total iters: ', "{}".format(lr_scheduler.num_iters),
                      'start_lr: ', "{:.8f}".format(lr_scheduler.start_lr),
                      'warmup_iter: ', "{}".format(lr_scheduler.warmup_iter),
                      'end_iter: ', "{}".format(lr_scheduler.end_iter),
                      'decay_style: ', "{}".format(lr_scheduler.decay_style))
            if (rank == 0):
                bar.update(lr_scheduler.num_iters)

            total_loss = 0.0
            stem_loss = 0.0
            affix_loss = 0.0
            afset_loss = 0.0

    if (rank == 0):
        if os.path.exists(save_file_path):
            copyfile(save_file_path, save_file_path+"_prev_checkpoint.pt")
            print(time_now(), 'Prev model file checkpointed!')

        model.eval()
        torch.save({'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'start_line': start_line,
                    'loop': loop,
                    'num_loops': num_loops}, save_file_path)
        model.train()
    return total_steps, total_loss, stem_loss, affix_loss, afset_loss

def train_fn(rank, args):
    import apex

    from morpho_data_loaders import morpho_seq_collate_wrapper, KBCorpusDataset, KBVocab, AffixSetVocab, read_corpus
    from morpho_model import kinyabert_base

    USE_GPU = (args.gpus > 0)

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if USE_GPU and torch.cuda.is_available():
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
        torch.cuda.set_device(rank)
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
        dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)

    home_path = args.home_path

    BPE_model_path = (home_path+"data/BPE-30k.mdl")
    bpe_encoder = yttm.BPE(model=BPE_model_path)

    kb_vocab = KBVocab()
    kbvocab_state_dict_file_path = (home_path+"data/kb_vocab_state_dict_2021-02-07.pt")
    kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

    affix_set_vocab = None
    if args.use_afsets:
        affix_set_vocab = AffixSetVocab(reduced_affix_dict_file=home_path+"data/reduced_affix_dict_"+str(args.afset_dict_size)+".csv",
                                    reduced_affix_dict_map_file=home_path+"data/reduced_affix_dict_map_"+str(args.afset_dict_size)+".csv")

    morpho_rel_pos_dict = None
    morpho_rel_pos_dmax = 5
    if args.use_pos_aware_rel_pos_bias:
        morpho_rel_pos_dict_file_path = (home_path+"data/morpho_rel_pos_dict_2021-03-24.pt")
        saved_pos_rel_dict = torch.load(morpho_rel_pos_dict_file_path)
        morpho_rel_pos_dict = saved_pos_rel_dict['morpho_rel_pos_dict']
        morpho_rel_pos_dmax = saved_pos_rel_dict['morpho_rel_pos_dmax']

    if (rank == 0):
        print('Vocab ready!')

    num_iters =  args.num_iters # 200000
    warmup_iter =  args.warmup_iter # 2000

    # batch_size = 20 # 20 for 3090 # 40 for A6000 # 12 for V100
    # accumulation_steps = 128 # 128 for 3090 # 16 for A6000  # 24 for 8 x V100 (~2K batch size)
    # number_of_load_batches = 384 # 512 for 3090 # 800 for A6000 # 1200 for V100
    #
    # # Switch to TUPE version on 2080Ti
    # if args.use_tupe_rel_pos_bias:
    #     batch_size = 6 # 6 for 2080Ti # 20 for 3090 # 40 for A6000 # 12 for V100
    #     accumulation_steps = 428 # 428 for 2080Ti # 128 for 3090 # 16 for A6000  # 24 for 8 x V100 (~2K batch size)
    #     number_of_load_batches = 8560 # 8560 for 2080Ti # 6400 for 3090 # 800 for A6000 # 1200 for V100

    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    number_of_load_batches = args.number_of_load_batches

    peak_lr = args.peak_lr # 4e-4
    wd = args.wd # 0.01
    lr_decay_style = 'linear'
    init_step = 0

    if (rank == 0):
        print(time_now(), 'Forming model ...', flush=True)

    model = kinyabert_base(kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                           device, args, saved_model_file = None)
    tot_num_affixes = model.encoder.tot_num_affixes

    if USE_GPU and torch.cuda.is_available():
        model = DDP(model, device_ids=[rank])
        optimizer = apex.optimizers.FusedLAMB(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-06, weight_decay=wd)
    else:
        from lamb import Lamb
        model = DDP(model, device_ids=[])
        optimizer = Lamb(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-06, weight_decay=wd)

    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=peak_lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=lr_decay_style,
                               last_iter=init_step)

    curr_save_file_path = home_path+"data/exploratory_kinyabert_model_"+date_now()+"_pos@{}_stem@{}_mbow@{}_pawrel@{}_tuperel@{}_afsets@{}@{}_predaffixes@{}_morpho@{}_paired@{}.pt".format(args.num_pos_m_embeddings,
                           args.num_stem_m_embeddings,
                           args.use_affix_bow_m_embedding,
                           args.use_pos_aware_rel_pos_bias,
                           args.use_tupe_rel_pos_bias,
                           args.use_afsets,
                           args.afset_dict_size,
                           args.predict_affixes,
                           args.use_morpho_encoder,
                           args.paired_encoder)

    # curr_save_file_path = home_path+"data/exploratory_kinyabert_model_2021-08-29_pos@2_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True@34008_predaffixes@False.pt"
    if args.exploratory_model_load is not None:
        curr_save_file_path=args.exploratory_model_load
        prev_save_file_path = curr_save_file_path
        # Load saved state
        kb_state_dict = torch.load(prev_save_file_path, map_location=device)
        model.load_state_dict(kb_state_dict['model_state_dict'])
        optimizer.load_state_dict(kb_state_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])

    num_train_loops = math.ceil(num_iters * args.accumulation_steps / args.number_of_load_batches)

    start_line = 0

    curr_loops = math.floor(lr_scheduler.num_iters * num_train_loops / num_iters)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if (rank == 0):
        print('---------------------------------- Model Size ----------------------------------------')
        print(time_now(), 'Total params:', total_params, 'Trainable params:', trainable_params, psutil.virtual_memory(), flush=True)
        print('Saving model in:', curr_save_file_path)
        print('---------------------------------------------------------------------------------------')
        print('Model Arguments:', args)
        print('------------------ Train Config --------------------')
        print('start_line: ', start_line)
        print('curr_loops: ', curr_loops)
        print('num_train_loops: ', num_train_loops)
        print('number_of_load_batches: ', args.number_of_load_batches)
        print('accumulation_steps: ', args.accumulation_steps)
        print('batch_size: ', args.batch_size)
        print('effective_batch_size: ', args.batch_size*args.accumulation_steps)
        print('num_iters: ', num_iters)
        print('warmup_iter: ', warmup_iter)
        print('iters: ', lr_scheduler.num_iters)
        print('peak_lr: {:.8f}'.format(peak_lr))
        print('-----------------------------------------------------')

    if (rank == 0):
        print(time_now(), 'Reading corpus text ...', flush=True)
    parsed_corpus_file = (home_path+"data/parsed_tfidf_corpus_2021-02-07.txt")
    parsed_corpus_lines = read_corpus(parsed_corpus_file)
    parsed_corpus_doc_ends = [i for i in range(len(parsed_corpus_lines)) if (len(parsed_corpus_lines[i]) == 0)]

    # # Reduce number of items
    # parsed_corpus_lines = parsed_corpus_lines[0:parsed_corpus_doc_ends[2000]]
    # parsed_corpus_doc_ends = [i for i in range(len(parsed_corpus_lines)) if (len(parsed_corpus_lines[i]) == 0)]
    if (rank == 0):
        print(time_now(), 'Corpus text read!', flush=True)

    total_steps = 0
    total_loss = 0.0
    stem_loss = 0.0
    affix_loss = 0.0
    afset_loss = 0.0

    if (rank == 0):
        print(time_now(), 'Start training for', num_train_loops, 'loops ({} iterations)'.format(num_iters), flush=True)
    model.train()
    model.zero_grad()

    with progressbar.ProgressBar(initial_value=lr_scheduler.num_iters, max_value=lr_scheduler.end_iter, redirect_stdout=True) as bar:
        if (rank == 0):
            bar.update(lr_scheduler.num_iters)
        for loop in range(curr_loops, num_train_loops):
            if (rank == 0):
                print(time_now(), 'Loading dataset...', flush=True)

            kb_dataset = KBCorpusDataset(args, kb_vocab, affix_set_vocab, bpe_encoder,
                                         morpho_rel_pos_dict, morpho_rel_pos_dmax,
                                         parsed_corpus_lines, parsed_corpus_doc_ends, True, start_line,
                                         args.number_of_load_batches * args.batch_size,
                                         max_seq_len=512, rank=rank)

            start_line = kb_dataset.start_line
            if (rank == 0):
                print(time_now(), 'Next start_line:', start_line, flush=True)

            kb_data_loader = DataLoader(kb_dataset, batch_size=args.batch_size, collate_fn=morpho_seq_collate_wrapper, shuffle=True)

            if (rank == 0):
                print(time_now(), 'Memory status: ', psutil.virtual_memory(), flush=True)

            (total_steps,
             total_loss, stem_loss, affix_loss, afset_loss) = train_loop(args, rank,
                                                             scaler,
                                                             device,
                                                             kb_data_loader,
                                                             start_line,
                                                             tot_num_affixes,
                                                             model,
                                                             optimizer,
                                                             lr_scheduler,
                                                             curr_save_file_path,
                                                             args.accumulation_steps,
                                                             loop, num_train_loops,
                                                             bar, total_steps,
                                                             total_loss, stem_loss, affix_loss, afset_loss)

            if (rank == 0):
                print(time_now(), (loop+1), 'TRAINING LOOPS COMPLETE!', flush=True)

            del kb_data_loader
            del kb_dataset
            gc.collect()

            if (rank == 0):
                print(time_now(), "After Garbage collection:", psutil.virtual_memory(), flush=True)

def main():
    from morpho_common import setup_common_args
    args = setup_common_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '88599'
    if args.gpus == 0:
        args.world_size = 1
    mp.spawn(train_fn, nprocs=args.world_size, args=(args,))

if __name__ == '__main__':
    main()

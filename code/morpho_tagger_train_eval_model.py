# Copyright (c) Antoine Nzeyimana.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function, division

import math

import torch
import torch.nn.functional as F
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader

from morpho_learning_rates import AnnealingLR

from kinyabert_utils import time_now, read_lines

def tagger_model_eval(args, tagger_model, device, eval_dataset, tag_dict):
    from morpho_tagger_data_loaders import tagger_morpho_model_predict
    tagger_model.eval()
    inv_dict = {tag_dict[k]:k for k in tag_dict}
    y_true  = []
    y_pred = []

    with torch.no_grad():
        for itr,data_item in enumerate(eval_dataset.itemized_data):
            output_scores, predicted_labels, true_labels = tagger_morpho_model_predict(args, data_item, tagger_model, device)
            y_pred.append([inv_dict[predicted_labels[i].item()] for i in range(len(true_labels))])
            y_true.append([inv_dict[true_label] for true_label in true_labels])

    F1 = f1_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    return F1, 1.0

def train_loop(args, keyword, epoch, scaler, tagger_model, device, optimizer, lr_scheduler, train_data_loader, valid_dataset, best_F1, best_dev_loss, accumulation_steps, log_each_batch_num, save_file_path, label_dict, bar):
    from morpho_tagger_data_loaders import  tagger_morpho_model_forward
    tagger_model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for batch_idx, batch_data_item in enumerate(train_data_loader):
        with torch.cuda.amp.autocast():
            output_scores, batch_labels = tagger_morpho_model_forward(args, batch_data_item, tagger_model, device)
            output_scores = F.log_softmax(output_scores, dim=1)
            loss = F.nll_loss(output_scores, torch.tensor(batch_labels).to(device))
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        total_loss += loss.item()

        if ((batch_idx+1) % accumulation_steps) == 0:
            lr_scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if ((batch_idx+1) % log_each_batch_num) == 0:
                print('@'+keyword+':', time_now(), 'Epochs:', epoch, 'Batch:', '{}/{}'.format((batch_idx + 1), len(train_data_loader)),
                      'Batch size:', len(batch_data_item[1]),
                      'TOTAL Loss: ', "{:.4f}".format(total_loss),
                      'Learning rate: ', "{:.8f}".format(lr_scheduler.get_lr()),
                      'Total iters: ', "{}".format(lr_scheduler.num_iters),
                      'start_lr: ', "{:.8f}".format(lr_scheduler.start_lr),
                      'warmup_iter: ', "{}".format(lr_scheduler.warmup_iter),
                      'end_iter: ', "{}".format(lr_scheduler.end_iter),
                      'decay_style: ', "{}".format(lr_scheduler.decay_style), flush=True)
                bar.update(lr_scheduler.num_iters)

            total_loss = 0.0

    # print('Train Set eval:')
    # train_F1, train_loss = tagger_model_eval(tagger_model, device, train_data_loader.dataset, label_dict)
    print('Valid Set eval:')
    dev_F1, _ = tagger_model_eval(args, tagger_model, device, valid_dataset, label_dict)
    print('@'+keyword+': After', (epoch+1), 'epochs:', '==> Validation set F1:','{:.2f}%'.format(100.0 * dev_F1),  '==> Best valid F1:','{:.2f}%'.format(100.0 * max(best_F1,dev_F1)))
    if dev_F1 > best_F1:
        best_F1 = dev_F1
        best_dev_loss = 1.0
        torch.save({'iter': iter,
                    'model_state_dict': tagger_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_F1': best_F1},
                   save_file_path)
    return best_F1, best_dev_loss

def KIN_NER_train_main(args, keyword):
    from morpho_tagger_data_loaders import KBTagCorpusDataset, collate_input_sequences
    from morpho_data_loaders import KBVocab, AffixSetVocab
    from morpho_model import kinyabert_base_tagger_from_pretrained
    import progressbar
    from adamw import mAdamW

    # @Workstation-PC
    home_path = args.home_path
    USE_GPU = (args.gpus > 0)

    kb_vocab = KBVocab()
    kbvocab_state_dict_file_path = (home_path+"data/kb_vocab_state_dict_2021-02-07.pt")
    kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

    affix_set_vocab = None
    if args.use_afsets:
        affix_set_vocab = AffixSetVocab(reduced_affix_dict_file=args.home_path + "data/reduced_affix_dict_"+str(args.afset_dict_size)+".csv",
                                        reduced_affix_dict_map_file=args.home_path + "data/reduced_affix_dict_map_"+str(args.afset_dict_size)+".csv")

    morpho_rel_pos_dict = None
    morpho_rel_pos_dmax = 5
    if args.use_pos_aware_rel_pos_bias:
        morpho_rel_pos_dict_file_path = (home_path+"data/morpho_rel_pos_dict_2021-03-24.pt")
        saved_pos_rel_dict = torch.load(morpho_rel_pos_dict_file_path)
        morpho_rel_pos_dict = saved_pos_rel_dict['morpho_rel_pos_dict']
        morpho_rel_pos_dmax = saved_pos_rel_dict['morpho_rel_pos_dmax']

    label_dict = {}
    label_dict['B-PER'] = 0
    label_dict['I-PER'] = 1
    label_dict['B-ORG'] = 2
    label_dict['I-ORG'] = 3
    label_dict['B-LOC'] = 4
    label_dict['I-LOC'] = 5
    label_dict['B-DATE'] = 6
    label_dict['I-DATE'] = 7
    label_dict['O'] = 8

    print('Vocab ready!')

    device = torch.device('cpu')
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(time_now(), 'Reading datasets ...', flush=True)

    train_lines_input0 = read_lines("/home/user/datasets/KIN_NER/parsed/train_parsed.txt")
    train_label_lines = read_lines("/home/user/datasets/KIN_NER/parsed/train_parsed_labels.txt")

    valid_lines_input0 = read_lines("/home/user/datasets/KIN_NER/parsed/dev_parsed.txt")
    valid_label_lines = read_lines("/home/user/datasets/KIN_NER/parsed/dev_parsed_labels.txt")

    num_classes = len(label_dict)
    accumulation_steps = args.accumulation_steps # 16 batch size
    log_each_batch_num = accumulation_steps

    max_lines = args.max_input_lines

    print(time_now(), 'Preparing dev set ...', flush=True)
    valid_dataset = KBTagCorpusDataset(args, kb_vocab, affix_set_vocab, morpho_rel_pos_dict, morpho_rel_pos_dmax, label_dict, valid_label_lines, valid_lines_input0)

    print(time_now(), 'Preparing training set ...', flush=True)
    train_dataset = KBTagCorpusDataset(args, kb_vocab, affix_set_vocab, morpho_rel_pos_dict, morpho_rel_pos_dmax, label_dict, train_label_lines, train_lines_input0, max_lines=max_lines)

    print(time_now(), 'Forming model ...', flush=True)

    peak_lrs = [args.peak_lr] #[1e-5, 3e-5, 5e-5, 8e-5]
    batch_sizes = [args.batch_size] #[16, 32]
    max_num_epochs = [args.num_epochs] #[5, 10, 15, 20, 30]

    pretrained_model_file = args.pretrained_model_file # (home_path+"data/backup_07_24_morpho_attentive_model_base_2021-04-19.pt")

    wd = args.wd
    lr_decay_style = 'linear'
    init_step = 0

    best_F1 = -99999.99
    best_dev_loss = 99999.99

    tagger_model_save_file_path = args.devbest_cls_model_save_file_path # (home_path+"data/tagger_kin_ner_morpho_attentive_model_base_2021-07-24.pt")

    scaler = torch.cuda.amp.GradScaler()

    tagger_model = kinyabert_base_tagger_from_pretrained(num_classes, kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                           device, args, pretrained_model_file, ddp = True, pooler_dropout=0.2)
    for peak_lr in peak_lrs:
        for batch_size in batch_sizes:
            for num_epochs in max_num_epochs:
                train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_input_sequences, drop_last=True, shuffle=True)

                tagger_model = kinyabert_base_tagger_from_pretrained(num_classes, kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                           device, args, pretrained_model_file, ddp = True, pooler_dropout=0.3)

                num_iters = math.ceil(num_epochs * len(train_data_loader) / accumulation_steps)
                warmup_iter = math.ceil(num_iters * 0.06) # warm-up for first 6% of iterations

                # optimizer = torch.optim.AdamW(tagger_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd)

                # From: https://github.com/uds-lsv/bert-stable-fine-tuning
                # Paper: On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines
                # Marius Mosbach, Maksym Andriushchenko, Dietrich Klakow
                # https://arxiv.org/abs/2006.04884
                optimizer = mAdamW(tagger_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd,
                                   correct_bias=True,
                                   local_normalization=False, max_grad_norm=-1)

                lr_scheduler = AnnealingLR(optimizer,
                                           start_lr=peak_lr,
                                           warmup_iter=warmup_iter,
                                           num_iters=num_iters,
                                           decay_style=lr_decay_style,
                                           last_iter=init_step)

                print(time_now(), 'Start training ...', flush=True)

                with progressbar.ProgressBar(initial_value=0, max_value=(2*lr_scheduler.end_iter), redirect_stdout=True) as bar:
                    bar.update(0)
                    for epoch in range(num_epochs):
                        best_F1, best_dev_loss = train_loop(args, keyword, epoch, scaler, tagger_model, device, optimizer, lr_scheduler,
                                                train_data_loader, valid_dataset, best_F1, best_dev_loss,
                                                accumulation_steps, log_each_batch_num, tagger_model_save_file_path, label_dict, bar)

    print(time_now(), 'Training complete!',  flush=True)
    del valid_dataset
    del train_dataset
    del train_data_loader

    print(time_now(), 'Preparing test set ...', flush=True)
    test_lines_input0 = read_lines("/home/user/datasets/KIN_NER/parsed/test_parsed.txt")
    test_label_lines = read_lines("/home/user/datasets/KIN_NER/parsed/test_parsed_labels.txt")
    test_dataset = KBTagCorpusDataset(args, kb_vocab, affix_set_vocab, morpho_rel_pos_dict, morpho_rel_pos_dmax, label_dict, test_label_lines, test_lines_input0)

    print('Test Set eval:')
    final_test_F1, final_test_loss = tagger_model_eval(args, tagger_model, device, test_dataset, label_dict)

    print(keyword, '==> Final test set F1:', '{:.2f}%'.format(100.0 * final_test_F1))

    kb_state_dict = torch.load(tagger_model_save_file_path, map_location=device)
    tagger_model.load_state_dict(kb_state_dict['model_state_dict'])

    test_F1, test_loss = tagger_model_eval(args, tagger_model, device, test_dataset, label_dict)

    print(keyword, '==> Final test set F1 (using best dev):', '{:.2f}%'.format(100.0 * test_F1))
    return best_F1, best_dev_loss, final_test_F1, final_test_loss

if __name__ == '__main__':
    from kinlpmorpho import build_kinlp_morpho_lib
    build_kinlp_morpho_lib()
    #from morpho_common import setup_common_args
    #KIN_NER_train_main(setup_common_args())

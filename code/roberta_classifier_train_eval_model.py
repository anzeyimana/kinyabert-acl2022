# Copyright (c) Antoine Nzeyimana.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function, division

import torch
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from kinyabert_utils import time_now, read_lines

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from roberta_classifier_model import RobertaForSentenceClassification, RobertaForTokenClassification

import numpy as np
from scipy import stats

def spearman_corr(r_x, r_y):
    return stats.spearmanr(r_x, r_y)[0]

def pearson_corr(r_x, r_y):
    return stats.pearsonr(r_x, r_y)[0]

class RobertaTaggerDataset(Dataset):

    def __init__(self, tagger_model: RobertaForTokenClassification, input_lines, label_lines, labels_dict, args):
        super(RobertaTaggerDataset, self).__init__()
        self.token_ids = [tagger_model.encode_sentence(sentence) for sentence in input_lines]
        labels = [label_lines[i].split(' ') for i in range(len(label_lines))] if (label_lines is not None) else None
        self.label_ids_list = []  if (label_lines is not None) else None
        if labels is not None:
            for lbl_list in labels:
                self.label_ids_list.append([labels_dict[lbl] for lbl in lbl_list])

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return (self.token_ids[idx],
                self.label_ids_list[idx] if (self.label_ids_list is not None) else None)

class RobertaClassifierDataset(Dataset):

    def __init__(self, classifier_model: RobertaForSentenceClassification, input0_lines, input1_lines, label_lines, labels_dict, args):
        super(RobertaClassifierDataset, self).__init__()
        if input1_lines is not None:
            self.token_ids = [classifier_model.encode_sentences(s0,s1) for s0,s1 in zip(input0_lines,input1_lines)]
        else:
            self.token_ids = [classifier_model.encode_sentences(s0, None) for s0 in input0_lines]
        self.label_ids_list = [[((float(lbl) / args.regression_scale_factor) if args.regression_target else labels_dict[lbl])] for lbl in label_lines]  if (label_lines is not None) else None

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return (self.token_ids[idx],
                self.label_ids_list[idx] if (self.label_ids_list is not None) else None)

def collate_input_sequences(batch_items):
    batch_token_ids_list = []
    batch_label_ids_list = []
    input_sequence_lengths = []
    for bidx, data_item in enumerate(batch_items):
        token_ids, label_ids = data_item
        batch_token_ids_list.append(token_ids)
        batch_label_ids_list.extend(label_ids)
        input_sequence_lengths.append(token_ids.size(0))

    batch_token_ids = pad_sequence(batch_token_ids_list, batch_first=True, padding_value=1)
    return (batch_token_ids, batch_label_ids_list, input_sequence_lengths)


def model_eval_tagging(model, device, dataset, tag_dict, args):
    import seqeval.metrics
    model.eval()
    inv_dict = {tag_dict[k]:k for k in tag_dict}
    y_true  = []
    y_pred = []
    tot_loss = 0.0
    tot_loss_count = 0.0

    eval_data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_input_sequences, shuffle=False)

    with torch.no_grad():
        # Compute loss
        for batch_idx, batch_data_item in enumerate(eval_data_loader):
            (batch_token_ids, batch_label_ids_list, input_sequence_lengths) = batch_data_item
            batch_token_ids = batch_token_ids.to(device)
            batch_label_ids = torch.tensor(batch_label_ids_list, dtype=int).to(device)
            output_scores = model(batch_token_ids, input_sequence_lengths)
            output_scores = F.log_softmax(output_scores, dim=1)
            loss = F.nll_loss(output_scores, batch_label_ids, reduction = 'sum')
            tot_loss_count += len(batch_label_ids_list)
            tot_loss += loss.item()
        tot_loss = tot_loss / tot_loss_count
        # Predict labels
        for token_ids, true_labels in zip(dataset.token_ids,dataset.label_ids_list):
            token_ids = token_ids.to(device)
            output_scores, predicted_labels = model.predict(token_ids)
            y_pred.append([inv_dict[predicted_labels[i].item()] for i in range(len(true_labels))])
            y_true.append([inv_dict[true_label] for true_label in true_labels])

    F1 = seqeval.metrics.f1_score(y_true, y_pred)
    print(seqeval.metrics.classification_report(y_true, y_pred))
    return F1, tot_loss

def model_eval_classification(model, device, dataset, cls_dict, args):
    import sklearn.metrics
    model.eval()
    total = 0.0
    accurate = 0.0
    inv_dict = {cls_dict[k]:k for k in cls_dict}
    y_true  = []
    y_pred = []
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    with torch.no_grad():
        for token_ids, true_labels in zip(dataset.token_ids,dataset.label_ids_list):
            true_label = true_labels[0]
            token_ids = token_ids.to(device)
            output_scores, predicted_labels = model.predict(token_ids)
            predicted_label = predicted_labels.item()
            total += 1.0
            if (predicted_label == true_label):
                accurate += 1.0
            if (predicted_label == true_label) and (predicted_label == 1):
                TP += 1.0
            if (predicted_label == true_label) and (predicted_label == 0):
                TN += 1.0
            if (predicted_label != true_label) and (predicted_label == 1):
                FP += 1.0
            if (predicted_label != true_label) and (predicted_label == 0):
                FN += 1.0
            y_pred.append(predicted_label)
            y_true.append(true_label)
    cls_labels = [i for i in range(len(inv_dict))]
    F1 = sklearn.metrics.f1_score(y_true, y_pred, labels=cls_labels, average='macro')
    print('Macro F1: {:.2f}'.format(100.0 * F1))
    if len(inv_dict) == 2:
        F1 = TP / (TP + ((FP + FN) / 2.0))
        print('Binary F1: {:.2f}'.format(100.0*F1))
    return accurate/total, F1

def model_eval_regression(model, device, dataset, cls_dict, args):
    model.eval()
    true_vals = []
    hyp_vals = []

    with torch.no_grad():
        # Predict labels
        for token_ids, true_labels in zip(dataset.token_ids,dataset.label_ids_list):
            true_label = true_labels[0]
            token_ids = token_ids.to(device)
            output_scores, predicted_labels = model.predict(token_ids)
            hyp_vals.append(args.regression_scale_factor * output_scores.item())
            true_vals.append(args.regression_scale_factor * true_label)
    return pearson_corr(np.array(true_vals), np.array(hyp_vals)), spearman_corr(np.array(true_vals), np.array(hyp_vals))

def model_eval(model, device, dataset, tag_dict, args, is_tagging=False):
    if is_tagging:
        return model_eval_tagging(model, device, dataset, tag_dict, args)
    if args.regression_target:
        return model_eval_regression(model, device, dataset, tag_dict, args)
    else:
        return model_eval_classification(model, device, dataset, tag_dict, args)


def glue_model_inference_output(args, model: RobertaForSentenceClassification, test_lines0_file, test_lines1_file, cls_dict, outfile):
    test_lines0 = read_lines(test_lines0_file)
    test_lines1 = read_lines(test_lines1_file) if (test_lines1_file is not None) else None

    USE_GPU = (args.gpus > 0)

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = RobertaClassifierDataset(model, test_lines0, test_lines1, None, cls_dict, args)

    model.eval()  # disable dropout (or leave in train mode to finetune)

    out = open(outfile, 'w')
    out.write("index	prediction\n")

    print(time_now(), "Evaluating model ...", flush=True)

    with torch.no_grad():
        for idx, token_ids in enumerate(dataset.token_ids):
            token_ids = token_ids.to(device)
            output_scores, predicted_labels = model.predict(token_ids)
            if (idx > 0):
                out.write("\n")
            if args.regression_target:
                out.write(str(idx) + "	" + str(args.regression_scale_factor * output_scores.item()))
            else:
                predicted_label = predicted_labels.item()
                out.write(str(idx) + "	" + cls_dict[predicted_label])
    out.close()

def train_loop(kwd, epoch, scaler, model, device, optimizer, lr_scheduler, train_data_loader, valid_dataset, best_F1, best_dev_loss, accumulation_steps, log_each_batch_num, save_file_path, label_dict, args, bar, is_tagging=False):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for batch_idx, batch_data_item in enumerate(train_data_loader):
        with torch.cuda.amp.autocast():
            (batch_token_ids, batch_label_ids_list, input_sequence_lengths) = batch_data_item
            # print('Batch_max: ',batch_token_ids.max().item())
            # print('Batch_min: ',batch_token_ids.min().item())
            # print(model.encode_sentence('mu ko'))
            # print('MASK:', model.roberta.encode('<mask>'))
            # print('UNK:', model.roberta.encode('<unk>'))
            # print('Decoded: ', model.roberta.decode(torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 42610, 42611, 42612, 42613, 42614, 42615, 42616, 42617])))
            batch_token_ids = batch_token_ids.to(device)
            output_scores = model(batch_token_ids, input_sequence_lengths)
            if args.regression_target:
                batch_label_ids = torch.tensor(batch_label_ids_list).to(device)
                output_scores = output_scores.view(-1).float()
                targets = torch.tensor(batch_label_ids).to(device).float()
                loss = F.mse_loss(output_scores, targets)
            else:
                batch_label_ids = torch.tensor(batch_label_ids_list, dtype=int).to(device)
                output_scores = F.log_softmax(output_scores, dim=1)
                loss = F.nll_loss(output_scores, torch.tensor(batch_label_ids).to(device))
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        total_loss += loss.item()

        if ((batch_idx+1) % accumulation_steps) == 0:
            lr_scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if ((batch_idx+1) % log_each_batch_num) == 0:
                print('@'+kwd+': ', time_now(), 'Epochs:', epoch, 'Batch:', '{}/{}'.format((batch_idx + 1), len(train_data_loader)),
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

    print('Valid Set eval:')
    dev_F1, dev_loss = model_eval(model, device, valid_dataset, label_dict, args, is_tagging=is_tagging)
    if is_tagging:
        print('@'+kwd+': After', (epoch+1), 'epochs:', '==> Validation set F1:','{:.2f}'.format(100.0 * dev_F1), 'Validation set Loss:','{:.4f}'.format(dev_loss),  '==> Best valid F1:','{:.2f}'.format(100.0 * max(best_F1,dev_F1)), 'Best valid loss:', '{:.4f}'.format(min(best_dev_loss,dev_loss)))
    if args.regression_target:
        print('@'+kwd+': After', (epoch+1), 'epochs:', '==> Validation set Pearson-R:','{:.2f}'.format(100.0 * dev_F1), 'Validation set Spearman-R:','{:.2f}'.format(100.0*dev_loss),  '==> Best valid Pearson-R:','{:.2f}'.format(100.0 * max(best_F1,dev_F1)), 'Best valid Spearman-R:', '{:.2f}'.format(100.0*max(best_dev_loss,dev_loss)))
    else:
        print('@'+kwd+': After', (epoch+1), 'epochs:', '==> Validation set Accuracy:','{:.2f}'.format(100.0 * dev_F1), 'Validation set F1:','{:.2f}'.format(100.0*dev_loss),  '==> Best valid Accuracy:','{:.2f}'.format(100.0 * max(best_F1,dev_F1)), 'Best valid F1:', '{:.2f}'.format(100.0*max(best_dev_loss,dev_loss)))
    if dev_F1 > best_F1:
        best_F1 = dev_F1
        best_dev_loss = dev_loss
        torch.save({'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_F1': best_F1},
                   save_file_path)
    return best_F1, best_dev_loss

def KIN_NER_train_main(args, keyword):
    from roberta_classifier_model import RobertaForTokenClassification
    import progressbar
    from adamw import mAdamW
    import math
    from morpho_learning_rates import AnnealingLR

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

    USE_GPU = (args.gpus > 0)

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using device:', device)

    print(time_now(), 'Reading datasets ...', flush=True)

    train_lines_input0 = read_lines(args.cls_train_input0)
    train_label_lines = read_lines(args.cls_train_label)

    valid_lines_input0 = read_lines(args.cls_dev_input0)
    valid_label_lines = read_lines(args.cls_dev_label)

    num_classes = len(label_dict)
    accumulation_steps = 1 # 16 batch size
    log_each_batch_num = accumulation_steps

    print(time_now(), 'Forming model ...', flush=True)
    tagger_model = RobertaForTokenClassification(args.embed_dim, num_classes * 32, num_classes, args)
    tagger_model = tagger_model.to(device)

    print(time_now(), 'Preparing dev set ...', flush=True)
    valid_dataset = RobertaTaggerDataset(tagger_model, valid_lines_input0, valid_label_lines, label_dict, args)

    print(time_now(), 'Preparing training set ...', flush=True)
    train_dataset = RobertaTaggerDataset(tagger_model, train_lines_input0, train_label_lines, label_dict, args)

    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    log_each_batch_num = accumulation_steps

    peak_lr = args.peak_lr # 1e-5
    wd = args.wd # 0.1
    lr_decay_style = 'linear'
    init_step = 0

    num_epochs = args.num_epochs # 30 # ==> Corresponds to 10 epochs

    best_F1 = -999999999.99
    best_dev_loss = 999999999.99

    scaler = torch.cuda.amp.GradScaler()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_input_sequences, drop_last=True, shuffle=True)

    num_iters = math.ceil(num_epochs * len(train_data_loader) / accumulation_steps)
    warmup_iter = math.ceil(num_iters * 0.06) # warm-up for first 6% of iterations

    # optimizer = torch.optim.AdamW(tagger_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd)
    # From: https://github.com/uds-lsv/bert-stable-fine-tuning
    # Paper: On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines
    # Marius Mosbach, Maksym Andriushchenko, Dietrich Klakow
    # https://arxiv.org/abs/2006.04884
    optimizer = mAdamW(tagger_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd, correct_bias=True,
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
            best_F1, best_dev_loss = train_loop(keyword, epoch, scaler, tagger_model, device, optimizer, lr_scheduler,
                                    train_data_loader, valid_dataset, best_F1, best_dev_loss,
                                    accumulation_steps, log_each_batch_num, args.devbest_cls_model_save_file_path, label_dict, args, bar, is_tagging=True)

    print(time_now(), 'Training complete!',  flush=True)
    del valid_dataset
    del train_dataset
    del train_data_loader

    test_F1 = -1.0
    test_loss = -1.0
    final_test_F1 = -1.0
    final_test_loss = -1.0
    if args.cls_test_label is not None:
        print(time_now(), 'Preparing test set ...', flush=True)
        test_lines_input0 = read_lines(args.cls_test_input0)
        test_label_lines = read_lines(args.cls_test_label)
        test_dataset = RobertaTaggerDataset(tagger_model, test_lines_input0, test_label_lines, label_dict, args)

        final_test_F1, final_test_loss = model_eval(tagger_model, device, test_dataset, label_dict, args, is_tagging=True)
        print(keyword, '==> Final test set avg F1:', '{:.2f}'.format(100.0 * final_test_F1), 'Loss:', '{:.4f}'.format(final_test_loss))

        kb_state_dict = torch.load(args.devbest_cls_model_save_file_path, map_location=device)
        tagger_model.load_state_dict(kb_state_dict['model_state_dict'])

        test_F1, test_loss = model_eval(tagger_model, device, test_dataset, label_dict, args, is_tagging=True)
        print(keyword, '==> Final test set avg F1 (using best dev):', '{:.2f}'.format(100.0 * test_F1), 'Loss:', '{:.4f}'.format(test_loss))

    kb_state_dict = torch.load(args.devbest_cls_model_save_file_path, map_location=device)
    tagger_model.load_state_dict(kb_state_dict['model_state_dict'])
    return tagger_model, best_F1, best_dev_loss, final_test_F1, final_test_loss

def TEXT_CLS_train_main(args, keyword):
    import progressbar
    from adamw import mAdamW
    import math
    import os
    from morpho_learning_rates import AnnealingLR

    #cls_labels = "0,1"
    labels = args.cls_labels.split(',')
    label_dict = {v:k for k,v in enumerate(labels)}

    USE_GPU = (args.gpus > 0)

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(time_now(), 'Reading datasets ...', flush=True)

    train_lines_input0 = read_lines(args.cls_train_input0)
    train_lines_input1 = read_lines(args.cls_train_input1) if (args.cls_train_input1 is not None) else None
    train_label_lines = read_lines(args.cls_train_label)

    valid_lines_input0 = read_lines(args.cls_dev_input0)
    valid_lines_input1 = read_lines(args.cls_dev_input1) if (args.cls_dev_input1 is not None) else None
    valid_label_lines = read_lines(args.cls_dev_label)

    num_classes = len(label_dict)
    batch_size = args.batch_size
    accumulation_steps = args.accumulation_steps
    log_each_batch_num = accumulation_steps

    print(time_now(), 'Forming model ...', flush=True)
    classifier_model = RobertaForSentenceClassification(args.embed_dim, num_classes * 32, num_classes, args)
    classifier_model = classifier_model.to(device)

    print(time_now(), 'Preparing training set ...', flush=True)
    train_dataset = RobertaClassifierDataset(classifier_model, train_lines_input0, train_lines_input1, train_label_lines,label_dict, args)

    print(time_now(), 'Preparing dev set ...', flush=True)
    valid_dataset = RobertaClassifierDataset(classifier_model, valid_lines_input0, valid_lines_input1, valid_label_lines, label_dict, args)

    peak_lr = args.peak_lr # 1e-5
    wd = args.wd # 0.1
    lr_decay_style = 'linear'
    init_step = 0

    num_epochs = args.num_epochs # 30 # ==> Corresponds to 10 epochs

    dev_accuracy = -999999999.99
    dev_F1 = -999999999.99

    scaler = torch.cuda.amp.GradScaler()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_input_sequences, drop_last=True, shuffle=True)

    num_iters = math.ceil(num_epochs * len(train_data_loader) / accumulation_steps)
    warmup_iter = math.ceil(num_iters * 0.06) # warm-up for first 6% of iterations

    # optimizer = torch.optim.AdamW(tagger_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd)
    # From: https://github.com/uds-lsv/bert-stable-fine-tuning
    # Paper: On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines
    # Marius Mosbach, Maksym Andriushchenko, Dietrich Klakow
    # https://arxiv.org/abs/2006.04884
    optimizer = mAdamW(classifier_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd, correct_bias=True,
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
            dev_accuracy, dev_F1 = train_loop(keyword, epoch, scaler, classifier_model, device, optimizer, lr_scheduler,
                                    train_data_loader, valid_dataset, dev_accuracy, dev_F1,
                                    accumulation_steps, log_each_batch_num, args.devbest_cls_model_save_file_path, label_dict, args, bar)

    print(time_now(), 'Training complete!',  flush=True)
    del valid_dataset
    del train_dataset
    del train_data_loader

    test_accuracy = -1.0
    test_F1 = -1.0
    final_test_accuracy = -1.0
    final_test_F1 = -1.0
    if args.cls_test_label is not None:
        print(time_now(), 'Preparing test set ...', flush=True)
        test_lines_input0 = read_lines(args.cls_test_input0)
        test_lines_input1 = read_lines(args.cls_test_input1) if (args.cls_test_input1 is not None) else None
        test_label_lines = read_lines(args.cls_test_label)

        test_dataset = RobertaClassifierDataset(classifier_model, test_lines_input0, test_lines_input1, test_label_lines, label_dict, args)

        final_test_accuracy, final_test_F1 = model_eval(classifier_model, device, test_dataset, label_dict, args)
        if args.regression_target:
            print(keyword, '==> Final test set Pearson:', '{:.2f}'.format(100.0 * final_test_accuracy), 'Spearman:', '{:.2f}'.format(100.0 * final_test_F1))
        else:
            print(keyword, '==> Final test set accuracy:', '{:.2f}'.format(100.0 * final_test_accuracy), 'F1:', '{:.2f}'.format(100.0 * final_test_F1))

        kb_state_dict = torch.load(args.devbest_cls_model_save_file_path, map_location=device)
        classifier_model.load_state_dict(kb_state_dict['model_state_dict'])

        test_accuracy, test_F1 = model_eval(classifier_model, device, test_dataset, label_dict, args)
        if args.regression_target:
            print(keyword, '==> Final test set Pearson (using best dev):', '{:.2f}'.format(100.0 * test_accuracy), 'Spearman:', '{:.2f}'.format(100.0 * test_F1))
        else:
            print(keyword, '==> Final test set accuracy (using best dev):', '{:.2f}'.format(100.0 * test_accuracy), 'F1:', '{:.2f}'.format(100.0 * test_F1))

    kb_state_dict = torch.load(args.devbest_cls_model_save_file_path, map_location=device)
    classifier_model.load_state_dict(kb_state_dict['model_state_dict'])
    return classifier_model, dev_accuracy, dev_F1, test_accuracy, test_F1

if __name__ == '__main__':
    from kinlpmorpho import build_kinlp_morpho_lib
    build_kinlp_morpho_lib()
    from morpho_common import setup_common_args
    args = setup_common_args()

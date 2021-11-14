from __future__ import print_function, division
import torch
from torch.utils.data import DataLoader
from morpho_learning_rates import AnnealingLR

import numpy as np

import math

import torch.nn.functional as F

from kinyabert_utils import read_lines, time_now

from scipy import stats

def spearman_corr(r_x, r_y):
    return stats.spearmanr(r_x, r_y)[0]

def pearson_corr(r_x, r_y):
    return stats.pearsonr(r_x, r_y)[0]

def cls_model_eval_classification(args, cls_model, device, eval_dataset):
    import sklearn.metrics
    from morpho_classifier_data_loaders import cls_morpho_model_predict
    cls_model.eval()
    total = 0.0
    accurate = 0.0
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    labels = args.cls_labels.split(',')
    label_dict = {v: k for k, v in enumerate(labels)}
    inv_dict = {label_dict[k]: k for k in label_dict}
    y_true  = []
    y_pred = []
    with torch.no_grad():
        for data_item in eval_dataset.itemized_data:
            output_scores, predicted_label, true_label = cls_morpho_model_predict(args, data_item, cls_model, device)
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
        print('Binary F1: {:.2f}'.format(100.0 * F1))
    return accurate/total, F1

def cls_model_eval_regression(args, cls_model, device, eval_dataset, regression_scale_factor):
    from morpho_classifier_data_loaders import cls_morpho_model_predict
    cls_model.eval()
    true_vals = []
    hyp_vals = []
    with torch.no_grad():
        for data_item in eval_dataset.itemized_data:
            output_scores, predicted_label, true_label = cls_morpho_model_predict(args, data_item, cls_model, device)
            hyp_vals.append(regression_scale_factor * output_scores.item())
            true_vals.append(regression_scale_factor * true_label)
    return pearson_corr(np.array(true_vals), np.array(hyp_vals)), spearman_corr(np.array(true_vals), np.array(hyp_vals))

def train_loop(args, keyword, epoch, scaler, cls_model, device, optimizer, lr_scheduler, train_data_loader, valid_dataset, best_accur, best_F1, accumulation_steps, log_each_batch_num, save_file_path, regression_target, regression_scale_factor, bar):
    from morpho_classifier_data_loaders import  cls_morpho_model_forward
    cls_model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for batch_idx, batch_data_item in enumerate(train_data_loader):
        with torch.cuda.amp.autocast():
            output_scores, batch_labels = cls_morpho_model_forward(args, batch_data_item, cls_model, device)
            if regression_target:
                output_scores = output_scores.view(-1).float()
                targets = torch.tensor(batch_labels).to(device).float()
                loss = F.mse_loss(output_scores, targets)
            else:
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
                print(keyword, time_now(), 'Epochs:', epoch, 'Batch:', '{}/{}'.format((batch_idx + 1), len(train_data_loader)),
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

    if regression_target:
        # train_accur, train_F1 = cls_model_eval_regression(cls_model, device, train_data_loader.dataset, regression_scale_factor)
        # dev_accur, dev_F1 = cls_model_eval_regression(cls_model, device, valid_dataset, regression_scale_factor)
        # print(keyword, 'After', (epoch + 1), 'epochs:', '==> Training set Pearson', '{:.2f}%'.format(100.0 * train_accur),
        #       'Spearman:', '{:.2f}%'.format(100.0 * train_F1), '==> Validation set Pearson',
        #       '{:.2f}%'.format(100.0 * dev_accur), 'Spearman:', '{:.2f}%'.format(100.0 * dev_F1),
        #       '==> Best valid Pearson so far: {:.2f}%'.format(100.0 * max(best_accur, dev_accur)), 'Spearman:',
        #       '{:.2f}%'.format(100.0 * max(best_F1, dev_F1)))
        dev_accur, dev_F1 = cls_model_eval_regression(args, cls_model, device, valid_dataset, regression_scale_factor)
        print(keyword, 'After', (epoch + 1), 'epochs:', '==> Validation set Pearson',
              '{:.2f}%'.format(100.0 * dev_accur), 'Spearman:', '{:.2f}%'.format(100.0 * dev_F1),
              '==> Best valid Pearson so far: {:.2f}%'.format(100.0 * max(best_accur, dev_accur)), 'Spearman:',
              '{:.2f}%'.format(100.0 * max(best_F1, dev_F1)))
    else:
        # train_accur, train_F1 = cls_model_eval_classification(cls_model, device, train_data_loader.dataset)
        # dev_accur, dev_F1 = cls_model_eval_classification(cls_model, device, valid_dataset)
        # print(keyword, 'After', (epoch + 1), 'epochs:', '==> Training set accuracy', '{:.2f}%'.format(100.0 * train_accur),
        #       'F1:', '{:.2f}%'.format(100.0 * train_F1), '==> Validation set accuracy',
        #       '{:.2f}%'.format(100.0 * dev_accur), 'F1:', '{:.2f}%'.format(100.0 * dev_F1),
        #       '==> Best valid accuracy so far: {:.2f}%'.format(100.0 * max(best_accur, dev_accur)), 'F1:',
        #       '{:.2f}%'.format(100.0 * max(best_F1, dev_F1)))
        dev_accur, dev_F1 = cls_model_eval_classification(args, cls_model, device, valid_dataset)
        print(keyword, 'After', (epoch + 1), 'epochs:', '==> Validation set accuracy',
              '{:.2f}%'.format(100.0 * dev_accur), 'F1:', '{:.2f}%'.format(100.0 * dev_F1),
              '==> Best valid accuracy so far: {:.2f}%'.format(100.0 * max(best_accur, dev_accur)), 'F1:',
              '{:.2f}%'.format(100.0 * max(best_F1, dev_F1)))
    if dev_accur > best_accur:
        best_accur = dev_accur
        best_F1 = dev_F1
        torch.save({'iter': iter,
                    'model_state_dict': cls_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_dev_accuracy': best_accur,
                    'best_dev_F1': best_F1},
                   save_file_path)
    return best_accur, best_F1

def TEXT_CLS_train_main(args, keyword):
    from morpho_classifier_data_loaders import KBClsCorpusDataset, collate_input_sequences
    from morpho_data_loaders import KBVocab, AffixSetVocab
    from morpho_model import kinyabert_base_classifier_from_pretrained
    import progressbar
    from adamw import mAdamW

    kb_vocab = KBVocab()
    kbvocab_state_dict_file_path = (args.home_path+"data/kb_vocab_state_dict_2021-02-07.pt")
    kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

    affix_set_vocab = None
    if args.use_afsets:
        affix_set_vocab = AffixSetVocab(reduced_affix_dict_file=args.home_path+"data/reduced_affix_dict_"+str(args.afset_dict_size)+".csv",
                                        reduced_affix_dict_map_file=args.home_path+"data/reduced_affix_dict_map_"+str(args.afset_dict_size)+".csv")

    morpho_rel_pos_dict = None
    morpho_rel_pos_dmax = 5
    if args.use_pos_aware_rel_pos_bias:
        morpho_rel_pos_dict_file_path = (args.home_path+"data/morpho_rel_pos_dict_2021-03-24.pt")
        saved_pos_rel_dict = torch.load(morpho_rel_pos_dict_file_path)
        morpho_rel_pos_dict = saved_pos_rel_dict['morpho_rel_pos_dict']
        morpho_rel_pos_dmax = saved_pos_rel_dict['morpho_rel_pos_dmax']

    #cls_labels = "0,1"
    labels = args.cls_labels.split(',')
    label_dict = {v:k for k,v in enumerate(labels)}

    print(keyword, 'Vocab ready!')

    USE_GPU = (args.gpus > 0)

    device = torch.device('cpu')
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(keyword, time_now(), 'Reading datasets ...', flush=True)

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

    start_line = 0
    max_input_lines = args.max_input_lines

    print(keyword, time_now(), 'Preparing dev set ...', flush=True)
    valid_dataset = KBClsCorpusDataset(args, kb_vocab, affix_set_vocab,
                                       morpho_rel_pos_dict, morpho_rel_pos_dmax, label_dict, valid_label_lines, valid_lines_input0, lines_input1=valid_lines_input1, regression_target = args.regression_target, regression_scale_factor=args.regression_scale_factor)

    print(keyword, time_now(), 'Preparing training set ...', flush=True)
    train_dataset = KBClsCorpusDataset(args, kb_vocab, affix_set_vocab, morpho_rel_pos_dict, morpho_rel_pos_dmax, label_dict, train_label_lines, train_lines_input0, lines_input1=train_lines_input1, start_line=start_line, max_lines=max_input_lines, regression_target = args.regression_target, regression_scale_factor=args.regression_scale_factor)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_input_sequences, shuffle=True)

    print(keyword, time_now(), 'Forming model ...', flush=True)

    pretrained_model_file = args.pretrained_model_file #(args.home_path+"data/backup_07_08_morpho_attentive_model_base_2021-04-19.pt")

    cls_model = kinyabert_base_classifier_from_pretrained(num_classes, kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                           device, args, pretrained_model_file, ddp = True, pooler_dropout=0.2)

    peak_lr = args.peak_lr # 1e-5
    wd = args.wd # 0.1
    lr_decay_style = 'linear'
    init_step = 0

    num_epochs = args.num_epochs # 30 # ==> Corresponds to 10 epochs
    if len(train_lines_input0) > args.max_input_lines:
        num_loops = num_epochs * int(len(train_lines_input0) / args.max_input_lines)
        print(keyword, 'Adjusted training loops:', num_loops,'corresponding to ', num_epochs, 'epochs')
        num_epochs = num_loops

    num_iters = math.ceil(num_epochs * len(train_data_loader) / accumulation_steps)
    warmup_iter = math.ceil(num_iters * args.warmup_ratio) # warm-up for first 6% of iterations

    # optimizer = torch.optim.AdamW(cls_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd)

    # From: https://github.com/uds-lsv/bert-stable-fine-tuning
    # Paper: On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines
    # Marius Mosbach, Maksym Andriushchenko, Dietrich Klakow
    # https://arxiv.org/abs/2006.04884
    optimizer = mAdamW(cls_model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=wd, correct_bias=True,
                 local_normalization=False, max_grad_norm=-1)

    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=peak_lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=lr_decay_style,
                               last_iter=init_step)

    scaler = torch.cuda.amp.GradScaler()

    best_accur = -99999.99
    best_F1 = -99999.99
    curr_epochs = 0
    if args.resume_from_best_saved:
        kb_state_dict = torch.load(args.devbest_cls_model_save_file_path)
        best_accur = kb_state_dict['best_dev_accuracy']
        best_F1 = kb_state_dict['best_dev_F1']
        cls_model.load_state_dict(kb_state_dict['model_state_dict'])
        optimizer.load_state_dict(kb_state_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])
        curr_epochs = math.ceil(lr_scheduler.num_iters * accumulation_steps / len(train_data_loader))

    print(keyword, time_now(), 'Start training ...', flush=True)

    with progressbar.ProgressBar(initial_value=0, max_value=(2*lr_scheduler.end_iter), redirect_stdout=True) as bar:
        bar.update(0)
        for epoch in range(curr_epochs,num_epochs):
            best_accur, best_F1 = train_loop(args, keyword, epoch, scaler, cls_model, device, optimizer, lr_scheduler,
                                    train_data_loader, valid_dataset, best_accur, best_F1,
                                    accumulation_steps, log_each_batch_num, args.devbest_cls_model_save_file_path, args.regression_target, args.regression_scale_factor,bar)
            if max_input_lines < len(train_lines_input0):
                del train_dataset
                del train_data_loader
                if ((epoch+1) < num_epochs):
                    start_line = start_line + max_input_lines
                    print(keyword, time_now(), 'Preparing training set ...', flush=True)
                    train_dataset = KBClsCorpusDataset(args, kb_vocab, affix_set_vocab, morpho_rel_pos_dict, morpho_rel_pos_dmax, label_dict, train_label_lines, train_lines_input0, lines_input1=train_lines_input1, start_line=start_line, max_lines=max_input_lines, regression_target = args.regression_target, regression_scale_factor=args.regression_scale_factor)
                    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_input_sequences, shuffle=True)

    print(keyword, time_now(), keyword, 'Training complete!',  flush=True)
    if args.regression_target:
        print(keyword, '==> Final best dev set Pearson:', '{:.2f}%'.format(100.0 * best_accur), 'Spearman:', '{:.2f}%'.format(100.0 * best_F1))
    else:
        print(keyword, '==> Final best dev set accuracy:', '{:.2f}%'.format(100.0 * best_accur), 'F1:', '{:.2f}%'.format(100.0 * best_F1))
    del valid_dataset
    if max_input_lines >= len(train_lines_input0):
        del train_dataset
        del train_data_loader

    #torch.save({'model_state_dict': cls_model.state_dict()}, args.final_cls_model_save_file_path)
    test_accuracy = -1.0
    test_F1 = -1.0
    final_test_accuracy = -1.0
    final_test_F1 = -1.0
    if args.cls_test_label is not None:
        print(keyword, time_now(), 'Preparing test set ...', flush=True)
        test_lines_input0 = read_lines(args.cls_test_input0)
        test_lines_input1 = read_lines(args.cls_test_input1) if (args.cls_test_input1 is not None) else None
        test_label_lines = read_lines(args.cls_test_label)
        test_dataset = KBClsCorpusDataset(args, kb_vocab, affix_set_vocab, morpho_rel_pos_dict, morpho_rel_pos_dmax, label_dict, test_label_lines, test_lines_input0, lines_input1=test_lines_input1, regression_target = args.regression_target, regression_scale_factor=args.regression_scale_factor)

        if args.regression_target:
            final_test_accuracy, final_test_F1 = cls_model_eval_regression(args, cls_model, device, test_dataset, args.regression_scale_factor)
            print(keyword, '==> Final test set Pearson:', '{:.2f}%'.format(100.0 * final_test_accuracy), 'Spearman:', '{:.2f}%'.format(100.0 * final_test_F1))
        else:
            final_test_accuracy, final_test_F1 = cls_model_eval_classification(args, cls_model, device, test_dataset)
            print(keyword, '==> Final test set accuracy:', '{:.2f}%'.format(100.0 * final_test_accuracy), 'F1:', '{:.2f}%'.format(100.0 * final_test_F1))

        kb_state_dict = torch.load(args.devbest_cls_model_save_file_path, map_location=device)
        cls_model.load_state_dict(kb_state_dict['model_state_dict'])

        if args.regression_target:
            test_accuracy, test_F1 = cls_model_eval_regression(args, cls_model, device, test_dataset, args.regression_scale_factor)
            print(keyword, '==> Final test set Pearson (using best dev):', '{:.2f}%'.format(100.0 * test_accuracy), 'Spearman:', '{:.2f}%'.format(100.0 * test_F1))
        else:
            test_accuracy, test_F1 = cls_model_eval_classification(args, cls_model, device, test_dataset)
            print(keyword, '==> Final test set accuracy (using best dev):', '{:.2f}%'.format(100.0 * test_accuracy), 'F1:', '{:.2f}%'.format(100.0 * test_F1))

    return best_accur, best_F1, test_accuracy, test_F1

if __name__ == '__main__':
    from kinlpmorpho import build_kinlp_morpho_lib
    build_kinlp_morpho_lib()
    #from morpho_common import setup_common_args
    #train_main(setup_common_args())

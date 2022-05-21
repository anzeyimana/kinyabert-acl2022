# Copyright (c) Antoine Nzeyimana.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

def train_evals(args):
    import os
    import numpy as np
    from morpho_tagger_train_eval_model import KIN_NER_train_main
    from morpho_classifier_train_eval_model import TEXT_CLS_train_main
    from morpho_classifier_inference import GLUE_eval_main

    print('########################################################################################################')
    print('##########                       Evaluating:', args.model_keyword, '                         ##########')
    print('########################################################################################################')

    args.world_size = args.gpus
    args.num_pos_m_embeddings = args.pos
    args.num_stem_m_embeddings = args.stem
    args.use_affix_bow_m_embedding = args.use_affix_bow
    args.use_pos_aware_rel_pos_bias = args.use_pos_aware_rel
    args.use_tupe_rel_pos_bias = args.use_tupe_rel
    args.num_inference_iters = args.inference_iters
    args.num_inference_runs = args.inference_runs

    args.input_format = 'parsed'
    args.max_input_lines = 8000 if args.use_pos_aware_rel_pos_bias else 999999

    NUM_TRIALS = 10


    ###################################################
    # NER
    ###################################################
    args.cls_train_input0 = "/home/user/datasets/KIN_NER/parsed/train_"+args.input_format+".txt"
    args.cls_dev_input0 = "/home/user/datasets/KIN_NER/parsed/dev_"+args.input_format+".txt"
    args.cls_test_input0 = "/home/user/datasets/KIN_NER/parsed/test_"+args.input_format+".txt"

    args.cls_train_input1 = None
    args.cls_dev_input1 = None
    args.cls_test_input1 = None

    args.cls_train_label = "/home/user/datasets/KIN_NER/parsed/train_"+args.input_format+"_labels.txt"
    args.cls_dev_label = "/home/user/datasets/KIN_NER/parsed/dev_"+args.input_format+"_labels.txt"
    args.cls_test_label = "/home/user/datasets/KIN_NER/parsed/test_"+args.input_format+"_labels.txt"

    args.batch_size = 16
    args.accumulation_steps = 2
    args.embed_dim = 768
    args.peak_lr = 0.00005
    args.wd = 0.1
    args.num_epochs = 30
    args.task_keyword = "NER"

    F1_dev = []
    F1_test = []
    loss_dev = []
    loss_test = []
    for trial_iter in range(NUM_TRIALS):
        kwd = args.model_keyword+"_"+args.task_keyword+"_"+str(trial_iter)
        args.devbest_cls_model_save_file_path = "/home/user/fine-tuning/"+args.model_keyword+"/"+kwd+"_devbest.pt"
        best_F1, best_dev_loss, test_F1, test_loss = KIN_NER_train_main(args, kwd)
        F1_dev.append(best_F1)
        F1_test.append(test_F1)
        loss_dev.append(best_dev_loss)
        loss_test.append(test_loss)
        os.remove(args.devbest_cls_model_save_file_path)

    print('@END_RESULTS: {} @ {} : \t @DEV-->F1: AVG {:.2f} STD {:.2f}  @DEV-->Loss: AVG {:.4f} STD {:.4f} \t @TEST-->F1: AVG {:.2f} STD {:.2f}  @TEST-->Loss: AVG {:.4f} STD {:.4f}'.format(args.task_keyword, args.model_keyword,
                                                                                      100.0*np.mean(F1_dev), 100.0*np.std(F1_dev),
                                                                                      np.mean(loss_dev), np.std(loss_dev),
                                                                                      100.0*np.mean(F1_test), 100.0*np.std(F1_test),
                                                                                      np.mean(loss_test), np.std(loss_test)))

    ###################################################
    # GLUE
    ###################################################
    GLUE_TASKS = ['WNLI', 'STS-B', 'RTE', 'MRPC', 'SST-2', 'QNLI']
    is_regression = {'MRPC':False, 'QNLI':False, 'RTE':False, 'SST-2':False, 'STS-B':True, 'WNLI':False}
    num_inputs = {'MRPC':2, 'QNLI':2, 'RTE':2, 'SST-2':1, 'STS-B':2, 'WNLI':2}
    effective_batch_sizes = {'MRPC':16, 'QNLI':32, 'RTE':16, 'SST-2':32, 'STS-B':16, 'WNLI':16}
    accumulation_steps = {'MRPC':1, 'QNLI':4, 'RTE':1, 'SST-2':1, 'STS-B':1, 'WNLI':1}
    peak_lr = {'MRPC': 0.00001, 'QNLI': 0.00001, 'RTE': 0.00002, 'SST-2': 0.00001, 'STS-B': 0.00002, 'WNLI': 0.00001}
    cls_labels = dict()
    cls_labels['MRPC'] = ['0', '1']
    cls_labels['QNLI'] = ['entailment', 'not_entailment']
    cls_labels['RTE'] = ['entailment', 'not_entailment']
    cls_labels['SST-2'] = ['0', '1']
    cls_labels['STS-B'] = ['0']
    cls_labels['WNLI'] = ['0', '1']

    for task in GLUE_TASKS:
        args.batch_size = effective_batch_sizes[task] // accumulation_steps[task]
        args.accumulation_steps = accumulation_steps[task]
        args.embed_dim = 768
        args.peak_lr = peak_lr[task]
        args.wd = 0.1
        args.num_epochs = 15
        args.cls_labels = ','.join(cls_labels[task])

        cls_dict = {i:v for i,v in enumerate(cls_labels[task])}

        args.cls_train_input0 = "/home/user/datasets/KIN_"+task+"/parsed/train_input0_"+args.input_format+".txt"
        args.cls_train_input1 = "/home/user/datasets/KIN_"+task+"/parsed/train_input1_"+args.input_format+".txt" if (num_inputs[task] > 1) else None
        args.cls_train_label = "/home/user/datasets/KIN_"+task+"/to_translate/train_score.txt" if is_regression[task] else "/home/user/datasets/KIN_"+task+"/to_translate/train_label.txt"

        args.cls_dev_input0 = "/home/user/datasets/KIN_"+task+"/parsed/dev_input0_"+args.input_format+".txt"
        args.cls_dev_input1 = "/home/user/datasets/KIN_"+task+"/parsed/dev_input1_"+args.input_format+".txt" if (num_inputs[task] > 1) else None
        args.cls_dev_label = "/home/user/datasets/KIN_"+task+"/to_translate/dev_score.txt" if is_regression[task] else "/home/user/datasets/KIN_"+task+"/to_translate/dev_label.txt"

        args.cls_test_input0 = "/home/user/datasets/KIN_"+task+"/parsed/test_input0_"+args.input_format+".txt"
        args.cls_test_input1 = "/home/user/datasets/KIN_"+task+"/parsed/test_input1_"+args.input_format+".txt" if (num_inputs[task] > 1) else None
        args.cls_test_label = None

        args.task_keyword = task

        args.regression_target = is_regression[task]
        args.regression_scale_factor = 5.0

        accuracy_dev = []
        F1_dev = []
        accuracy_test = []
        F1_test = []
        for trial_iter in range(NUM_TRIALS):
            kwd = args.model_keyword + "_" + args.task_keyword + "_" + str(trial_iter)
            args.devbest_cls_model_save_file_path = "/home/user/fine-tuning/" + args.model_keyword + "/" + kwd + "_devbest.pt"
            dev_accuracy, dev_F1, test_accuracy, test_F1 = TEXT_CLS_train_main(args, kwd)
            args.devbest_cls_output_file = "/home/user/fine-tuning/" + args.model_keyword+"/"+args.model_keyword + "-GLUE-"+str(trial_iter)+"/"+task+".tsv"
            GLUE_eval_main(args)
            accuracy_dev.append(dev_accuracy)
            F1_dev.append(dev_F1)
            accuracy_test.append(test_accuracy)
            F1_test.append(test_F1)
            os.remove(args.devbest_cls_model_save_file_path)

        if args.regression_target:
            print(
                '@END_RESULTS: {} @ {} : \t @DEV-->Pearson-R: AVG {:.2f} STD {:.2f}  @DEV-->Spearman-R: AVG {:.2f} STD {:.2f} \t @TEST-->Pearson-R: AVG {:.2f} STD {:.2f}  @TEST-->Spearman-R: AVG {:.2f} STD {:.2f}'.format(
                    args.task_keyword, args.model_keyword,
                    100.0*np.mean(accuracy_dev), 100.0*np.std(accuracy_dev),
                    100.0*np.mean(F1_dev), 100.0*np.std(F1_dev),
                    100.0*np.mean(accuracy_test), 100.0*np.std(accuracy_test),
                    100.0*np.mean(F1_test), 100.0*np.std(F1_test)))
        else:
            print(
                '@END_RESULTS: {} @ {} : \t @DEV-->ACC: AVG {:.2f} STD {:.2f}  @DEV-->F1: AVG {:.2f} STD {:.2f} \t @TEST-->ACC: AVG {:.2f} STD {:.2f}  @TEST-->F1: AVG {:.2f} STD {:.2f}'.format(
                    args.task_keyword, args.model_keyword,
                    100.0*np.mean(accuracy_dev), 100.0*np.std(accuracy_dev),
                    100.0*np.mean(F1_dev), 100.0*np.std(F1_dev),
                    100.0*np.mean(accuracy_test), 100.0*np.std(accuracy_test),
                    100.0*np.mean(F1_test), 100.0*np.std(F1_test)))


    ###################################################
    # RW_NEWS
    ###################################################
    cls_labels = ['politics', 'health', 'entertainment', 'security', 'economy', 'sports', 'religion', 'development', 'technology', 'culture', 'relationships', 'people']
    args.batch_size = 8
    args.accumulation_steps = 4
    args.embed_dim = 768
    args.peak_lr = 0.00001
    args.wd = 0.1
    args.num_epochs = 15
    args.task_keyword = "NEWS"
    args.cls_labels = ','.join(cls_labels)

    args.regression_target = False

    args.cls_train_input0 = "/home/user/datasets/RW_NEWS/parsed/train_input0_"+args.input_format+".txt"
    args.cls_train_input1 = None
    args.cls_train_label = "/home/user/datasets/RW_NEWS/plain/train.label.txt"

    args.cls_dev_input0 = "/home/user/datasets/RW_NEWS/parsed/dev_input0_"+args.input_format+".txt"
    args.cls_dev_input1 = None
    args.cls_dev_label = "/home/user/datasets/RW_NEWS/plain/dev.label.txt"

    args.cls_test_input0 = "/home/user/datasets/RW_NEWS/parsed/test_input0_"+args.input_format+".txt"
    args.cls_test_input1 = None
    args.cls_test_label = "/home/user/datasets/RW_NEWS/plain/test.label.txt"

    accuracy_dev = []
    F1_dev = []
    accuracy_test = []
    F1_test = []
    for trial_iter in range(NUM_TRIALS):
        kwd = args.model_keyword + "_" + args.task_keyword + "_" + str(trial_iter)
        args.devbest_cls_model_save_file_path = "/home/user/fine-tuning/" + args.model_keyword+"/"+kwd + "_devbest.pt"
        dev_accuracy, dev_F1, test_accuracy, test_F1 = TEXT_CLS_train_main(args, kwd)
        accuracy_dev.append(dev_accuracy)
        F1_dev.append(dev_F1)
        accuracy_test.append(test_accuracy)
        F1_test.append(test_F1)
        os.remove(args.devbest_cls_model_save_file_path)

    print(
        '@END_RESULTS: {} @ {} : \t @DEV-->ACC: AVG {:.2f} STD {:.2f}  @DEV-->F1: AVG {:.2f} STD {:.2f} \t @TEST-->ACC: AVG {:.2f} STD {:.2f}  @TEST-->F1: AVG {:.2f} STD {:.2f}'.format(
            args.task_keyword, args.model_keyword,
            100.0*np.mean(accuracy_dev), 100.0*np.std(accuracy_dev),
            100.0*np.mean(F1_dev), 100.0*np.std(F1_dev),
            100.0*np.mean(accuracy_test), 100.0*np.std(accuracy_test),
            100.0*np.mean(F1_test), 100.0*np.std(F1_test)))

def train_evals_all():
    from morpho_common import setup_common_args

    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-afset34008-32K'
    # args.gpus = 1
    # args.pos = 2
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = True
    # args.afset_dict_size = 34008
    # args.predict_affixes = False
    # args.pretrained_model_file = 'data/backup_10_18_exploratory_kinyabert_model_2021-08-29_pos@2_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True@34008_predaffixes@False.pt'
    # train_evals(args)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-tupe-08-22'
    # args.gpus = 1
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = False
    # args.predict_affixes = True
    # args.pretrained_model_file = 'data/backup_08_22_morpho_attentive_model_base_2021-07-12.pt'
    # train_evals(args)

    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-stem-32K'
    # args.gpus = 1
    # args.pos = 0
    # args.stem = 0
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = False
    # args.predict_affixes = False
    # args.use_morpho_encoder = False
    # args.stem_dim = 768
    # args.pretrained_model_file = 'data/backup_10_23_exploratory_kinyabert_model_2021-09-29_pos@0_stem@0_mbow@False_pawrel@False_tuperel@True_afsets@False@34008_predaffixes@False_morpho@False.pt'
    # train_evals(args)

    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-afset34008-11K'
    # args.gpus = 1
    # args.pos = 2
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = True
    # args.afset_dict_size = 34008
    # args.predict_affixes = False
    # args.pretrained_model_file = 'data/backup_09_29_exploratory_kinyabert_model_2021-08-29_pos@2_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True@34008_predaffixes@False.pt'
    # train_evals(args)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-tupe-07-26'
    # args.gpus = 1
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = False
    # args.predict_affixes = True
    # args.pretrained_model_file = 'data/backup_07_26_morpho_attentive_model_base_2021-07-12.pt'
    # train_evals(args)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-tupe-bowm-08-12'
    # args.gpus = 1
    # args.pos = 2
    # args.stem = 1
    # args.use_affix_bow = True
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = False
    # args.predict_affixes = True
    # args.pretrained_model_file = 'data/backup_08_12_exploratory_kinyabert_model_2021-07-30_pos:2_stem:1_mbow:True_pawrel:False_tuperel:True.pt'
    # train_evals(args)

    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-stem-11K'
    # args.gpus = 1
    # args.pos = 0
    # args.stem = 0
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = False
    # args.predict_affixes = False
    # args.use_morpho_encoder = False
    # args.stem_dim = 768
    # args.pretrained_model_file = 'data/backup_10_07_exploratory_kinyabert_model_2021-09-29_pos@0_stem@0_mbow@False_pawrel@False_tuperel@True_afsets@False@34008_predaffixes@False_morpho@False.pt'
    # train_evals(args)

    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-04-23'
    # args.gpus = 1
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = True
    # args.use_tupe_rel=False
    # args.use_afsets=False
    # args.predict_affixes=True
    # args.pretrained_model_file = 'data/backup_04_23_morpho_attentive_model_base_2021-04-19.pt'
    # train_evals(args)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-05-14'
    # args.gpus = 1
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = True
    # args.use_tupe_rel=False
    # args.use_afsets=False
    # args.predict_affixes=True
    # args.pretrained_model_file = 'data/backup_05_14_morpho_attentive_model_base_2021-04-19.pt'
    # train_evals(args)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-08-27'
    # args.gpus = 1
    # args.pos = 3
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = True
    # args.use_tupe_rel=False
    # args.use_afsets=False
    # args.predict_affixes=True
    # args.pretrained_model_file = 'data/backup_08_27_morpho_attentive_model_base_2021-04-19.pt'
    # train_evals(args)
    #
    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-afset-tupe-08-29'
    # args.gpus = 1
    # args.pos = 2
    # args.stem = 1
    # args.use_affix_bow = False
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = True
    # args.predict_affixes = False
    # args.pretrained_model_file = 'data/backup_08_29_exploratory_kinyabert_model_2021-08-15_pos@2_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True_predaffixes@False.pt'
    # train_evals(args)

    # args = setup_common_args()
    # args.model_keyword = 'kinyabert-bowm-32K'
    # args.gpus = 1
    # args.pos = 2
    # args.stem = 1
    # args.use_affix_bow = True
    # args.use_pos_aware_rel = False
    # args.use_tupe_rel = True
    # args.use_afsets = False
    # args.predict_affixes = True
    # args.pretrained_model_file = 'data/backup_10_30_exploratory_kinyabert_model_2021-07-30_pos:2_stem:1_mbow:True_pawrel:False_tuperel:True.pt'
    # train_evals(args)

    args = setup_common_args()
    args.model_keyword = 'kinyabert-paired-11K'
    args.gpus = 1
    args.pos = 0
    args.stem = 1
    args.morpho_dim = 384
    args.stem_dim = 384
    args.embed_dim = 768
    args.paired_encoder = True
    args.seq_tr_nhead = 8
    args.seq_tr_nlayers = 8
    args.seq_tr_dim_feedforward = 2048
    args.use_afsets = True
    args.afset_dict_size = 34008
    args.predict_affixes = False
    args.use_affix_bow = False
    args.use_pos_aware_rel = False
    args.use_tupe_rel = True
    args.pretrained_model_file = 'data/backup_11_03_exploratory_kinyabert_model_2021-10-22_pos@0_stem@1_mbow@False_pawrel@False_tuperel@True_afsets@True@34008_predaffixes@False_morpho@True_paired@True.pt'
    train_evals(args)

if __name__ == '__main__':
    from morpho_common import setup_common_args
    args = setup_common_args()
    import torch
    # from kinlpmorpho import build_kinlp_morpho_lib
    # build_kinlp_morpho_lib()
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '88599'
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.gpus, rank=0)
    else:
        torch.distributed.init_process_group(backend='gloo', init_method='env://', world_size=1, rank=0)

    train_evals_all()


from __future__ import print_function, division
import torch

# @Workstation-PC

from kinyabert_utils import read_lines, time_now

def cls_model_inference(args, cls_model, device, eval_dataset, cls_dict, outfile, regression_target, regression_scale_factor):
    from morpho_classifier_data_loaders import cls_morpho_model_predict
    out = open(outfile, 'w')
    out.write("index	prediction\n")

    cls_model = cls_model.to(device)
    cls_model.eval()
    with torch.no_grad():
        for idx,data_item in enumerate(eval_dataset.itemized_data):
            output_scores, predicted_label, fake_true_label = cls_morpho_model_predict(args, data_item, cls_model, device)
            if (idx > 0):
                out.write("\n")
            if regression_target:
                out.write(str(idx) + "	" + str(regression_scale_factor * output_scores.item()))
            else:
                out.write(str(idx) + "	" + cls_dict[predicted_label])
    out.close()

def GLUE_eval_main(args):
    from morpho_classifier_data_loaders import KBClsEvalCorpusDataset
    from morpho_data_loaders import KBVocab, AffixSetVocab
    from morpho_model import kinyabert_base_classifier

    kb_vocab = KBVocab()
    kbvocab_state_dict_file_path = (args.home_path+"data/kb_vocab_state_dict_2021-02-07.pt")
    kb_vocab.load_state_dict(torch.load(kbvocab_state_dict_file_path))

    affix_set_vocab = None
    if args.use_afsets:
        affix_set_vocab = AffixSetVocab(reduced_affix_dict_file=args.home_path + "data/reduced_affix_dict_"+str(args.afset_dict_size)+".csv",
                                        reduced_affix_dict_map_file=args.home_path + "data/reduced_affix_dict_map_"+str(args.afset_dict_size)+".csv")

    morpho_rel_pos_dict = None
    morpho_rel_pos_dmax = 5
    if args.use_pos_aware_rel_pos_bias:
        morpho_rel_pos_dict_file_path = (args.home_path+"data/morpho_rel_pos_dict_2021-03-24.pt")
        saved_pos_rel_dict = torch.load(morpho_rel_pos_dict_file_path)
        morpho_rel_pos_dict = saved_pos_rel_dict['morpho_rel_pos_dict']
        morpho_rel_pos_dmax = saved_pos_rel_dict['morpho_rel_pos_dmax']

    labels = args.cls_labels.split(',')
    label_dict = {v:k for k,v in enumerate(labels)}
    cls_dict = {k:v for k,v in enumerate(labels)}
    num_classes = len(label_dict)

    print('Vocab ready!')

    USE_GPU = (args.gpus > 0)

    device = torch.device('cpu')
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')

    print(time_now(), 'Preparing test set ...', flush=True)
    test_lines_input0 = read_lines(args.cls_test_input0)
    test_lines_input1 = read_lines(args.cls_test_input1) if (args.cls_test_input1 is not None) else None
    test_dataset = KBClsEvalCorpusDataset(args, kb_vocab, affix_set_vocab, morpho_rel_pos_dict, morpho_rel_pos_dmax, test_lines_input0, lines_input1=test_lines_input1)

    # print(time_now(), 'Evaluating final model ...', flush=True)
    # cls_model = kinyabert_base_classifier(num_classes, kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
    #                        device, args, saved_model_file=args.final_cls_model_save_file_path)
    # cls_model_inference(args, cls_model, device, test_dataset, cls_dict, args.final_cls_output_file, args.regression_target, args.regression_scale_factor)

    print(time_now(), 'Evaluating devbest model ...', flush=True)
    cls_model = kinyabert_base_classifier(num_classes, kb_vocab, affix_set_vocab, morpho_rel_pos_dict,
                           device, args, saved_model_file=args.devbest_cls_model_save_file_path)
    cls_model_inference(args, cls_model, device, test_dataset, cls_dict, args.devbest_cls_output_file, args.regression_target, args.regression_scale_factor)

    print(time_now(), 'Done!', flush=True)

if __name__ == '__main__':
    from kinlpmorpho import build_kinlp_morpho_lib
    build_kinlp_morpho_lib()
    #from morpho_common import setup_common_args
    #eval_main(setup_common_args())

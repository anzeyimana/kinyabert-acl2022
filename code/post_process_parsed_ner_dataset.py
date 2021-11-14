
def post_process_files(input_file,labels_file,output_labels_file):
    from morpho_data_loaders import ParsedToken
    from kinyabert_utils import read_lines

    input_lines = read_lines(input_file)
    labels_lines = read_lines(labels_file)
    parsed_labels_file = open(output_labels_file, 'w')
    for idx in range(len(input_lines)):
        tokens = [ParsedToken('_', parsed_token=t) for t in input_lines[idx].split('; ')]
        labels = labels_lines[idx].split(' ')
        real_labels = []
        for token,label in zip(tokens,labels):
            new_labels = ([label]+(['I'+label[1:]]*(len(token.stem_idx)-1))) if label.startswith('B') else ([label]*len(token.stem_idx))
            real_labels.extend(new_labels)
        parsed_labels_file.write(' '.join(real_labels) + "\n")
    parsed_labels_file.close()

if __name__ == '__main__':
    from kinlpmorpho import build_kinlp_morpho_lib
    build_kinlp_morpho_lib()
    post_process_files('/home/user/projects/user/kinyabert/datasets/KIN_NER/parsed/train_parsed.txt',
                       '/home/user/projects/user/kinyabert/datasets/KIN_NER/parsed/train_labels.txt',
                       '/home/user/projects/user/kinyabert/datasets/KIN_NER/parsed/train_parsed_labels.txt')

    post_process_files('/home/user/projects/user/kinyabert/datasets/KIN_NER/parsed/dev_parsed.txt',
                       '/home/user/projects/user/kinyabert/datasets/KIN_NER/parsed/dev_labels.txt',
                       '/home/user/projects/user/kinyabert/datasets/KIN_NER/parsed/dev_parsed_labels.txt')

    post_process_files('/home/user/projects/user/kinyabert/datasets/KIN_NER/parsed/test_parsed.txt',
                       '/home/user/projects/user/kinyabert/datasets/KIN_NER/parsed/test_labels.txt',
                       '/home/user/projects/user/kinyabert/datasets/KIN_NER/parsed/test_parsed_labels.txt')

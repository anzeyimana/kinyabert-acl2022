import argparse
from cffi import FFI
import youtokentome as yttm

class SimpleKBVocab:

    def __init__(self):
        self.pos_tag_vocab = dict()
        self.pos_tag_vocab_idx = dict()
        self.pos_tag_vocab_idx_counts = dict()
        self.pos_tag_vocab_idx_subsample_weights = dict()
        self.pos_tag_vocab_idx_subsample_weights_max = 0.1
        self.pos_tag_vocab_idx_subsample_weights_min = 0.1

        self._stem_vocab = dict()
        self._stem_vocab_idx = dict()
        self._stem_vocab_idx_counts = dict()

        self.reduced_stem_vocab = dict()
        self.reduced_stem_vocab_idx = dict()
        self.mapped_stem_vocab_idx = dict()
        self.reduced_stem_vocab_idx_counts = dict()
        self.reduced_stem_vocab_idx_subsample_weights = dict()
        self.reduced_stem_vocab_idx_subsample_weights_max = 0.1
        self.reduced_stem_vocab_idx_subsample_weights_min = 0.1

        self.morpheme_slot_vocab = dict()
        self.morpheme_slot_vocab_idx = dict()
        self.morpheme_slot_vocab_idx_counts = dict()
        self.morpheme_slot_vocab_idx_subsample_weights = dict()
        self.morpheme_slot_vocab_idx_subsample_weights_max = 0.1
        self.morpheme_slot_vocab_idx_subsample_weights_min = 0.1

        self.affix_vocab = dict()
        self.affix_vocab_idx = dict()
        self.affix_vocab_idx_counts = dict()
        self.affix_vocab_idx_subsample_weights = dict()
        self.affix_vocab_idx_subsample_weights_max = 0.1
        self.affix_vocab_idx_subsample_weights_min = 0.1

    def load_state_dict(self, d):
        self.pos_tag_vocab = d['pos_tag_vocab']
        self.pos_tag_vocab_idx = d['pos_tag_vocab_idx']
        self.pos_tag_vocab_idx_counts = d['pos_tag_vocab_idx_counts']
        self._stem_vocab = d['_stem_vocab']
        self._stem_vocab_idx = d['_stem_vocab_idx']
        self._stem_vocab_idx_counts = d['_stem_vocab_idx_counts']
        self.reduced_stem_vocab = d['reduced_stem_vocab']
        self.mapped_stem_vocab_idx = d['mapped_stem_vocab_idx']
        self.reduced_stem_vocab_idx_counts = d['reduced_stem_vocab_idx_counts']
        self.morpheme_slot_vocab = d['morpheme_slot_vocab']
        self.morpheme_slot_vocab_idx = d['morpheme_slot_vocab_idx']
        self.morpheme_slot_vocab_idx_counts = d['morpheme_slot_vocab_idx_counts']
        self.affix_vocab = d['affix_vocab']
        self.affix_vocab_idx = d['affix_vocab_idx']
        self.affix_vocab_idx_counts = d['affix_vocab_idx_counts']

        self.morpheme_slot_vocab_idx_subsample_weights = d['morpheme_slot_vocab_idx_subsample_weights']
        self.morpheme_slot_vocab_idx_subsample_weights_max = d['morpheme_slot_vocab_idx_subsample_weights_max']
        self.morpheme_slot_vocab_idx_subsample_weights_min = d['morpheme_slot_vocab_idx_subsample_weights_min']

        self.pos_tag_vocab_idx_subsample_weights = d['pos_tag_vocab_idx_subsample_weights']
        self.pos_tag_vocab_idx_subsample_weights_max = d['pos_tag_vocab_idx_subsample_weights_max']
        self.pos_tag_vocab_idx_subsample_weights_min = d['pos_tag_vocab_idx_subsample_weights_min']

        self.affix_vocab_idx_subsample_weights = d['affix_vocab_idx_subsample_weights']
        self.affix_vocab_idx_subsample_weights_max = d['affix_vocab_idx_subsample_weights_max']
        self.affix_vocab_idx_subsample_weights_min = d['affix_vocab_idx_subsample_weights_min']

        self.reduced_stem_vocab_idx_subsample_weights = d['reduced_stem_vocab_idx_subsample_weights']
        self.reduced_stem_vocab_idx_subsample_weights_max = d['reduced_stem_vocab_idx_subsample_weights_max']
        self.reduced_stem_vocab_idx_subsample_weights_min = d['reduced_stem_vocab_idx_subsample_weights_min']

        for k in self.reduced_stem_vocab:
            self.reduced_stem_vocab_idx[self.reduced_stem_vocab[k]] = k
        self.reduced_stem_vocab_idx_counts = dict()
        for i in self._stem_vocab_idx_counts:
            self.reduced_stem_vocab_idx_counts[self.mapped_stem_vocab_idx[i]] = self._stem_vocab_idx_counts[i]

class RichParsedToken:
    def __init__(self, surface_form, raw_surface_form, is_apostrophed, parsed_token=None, decode_prob=None, tf_idf=0.0, pos_tag_id=None, stem_ids=None, line_num=0):
        self.surface_form = surface_form
        self.raw_surface_form = raw_surface_form
        self.is_apostrophed = is_apostrophed
        self.tf_idf = tf_idf
        self.morpho_tokens = []
        if parsed_token is not None:
            parts = parsed_token.split('/')
            self.decode_prob = float(parts[0])
            self.tf_idf = float(parts[1])
            morphs = parts[2].split(',')
            pos_stem = morphs[0].split(':')
            stem_parts = pos_stem[1].split('*')
            if(len(stem_parts[0]) < 1):
                # print('\nParsing wrong token: /{}/ at line # {}'.format(parsed_token, line_num))
                self.pos_tag_idx = int(pos_stem[0])
                self.stem_idx = [6]
                self.morpho_slots_idx = []
                self.affixes_idx = []
            else:
                self.pos_tag_idx = int(pos_stem[0])
                self.stem_idx = [int(v) for v in stem_parts]
                self.morpho_slots_idx = [int(morphs[i].split(':')[0]) for i in range(1, len(morphs))]
                self.affixes_idx = [int(morphs[i].split(':')[1]) for i in range(1, len(morphs))]
        else:
            self.decode_prob = decode_prob
            self.pos_tag_idx = pos_tag_id
            self.stem_idx = stem_ids
            self.morpho_slots_idx = []
            self.affixes_idx = []

    def append_morpheme(self, morpho_slot_id, affix_id):
        self.morpho_slots_idx.append(morpho_slot_id)
        self.affixes_idx.append(affix_id)

    def to_parsed_format(self):
        st = ['{}:{}'.format(self.pos_tag_idx,'*'.join([str(i) for i in self.stem_idx]))]
        for i in range(len(self.morpho_slots_idx)):
            st.append('{}:{}'.format(self.morpho_slots_idx[i], self.affixes_idx[i]))
        return '{:.3g}/{:.3g}/{}'.format(self.decode_prob, self.tf_idf, ','.join(st))

    def affix_set_key(self):
        key = '-'.join([str(af) for af in self.affixes_idx]) if (len(self.affixes_idx) > 0) else 'N/A'
        return key

def build_kinlp_morpho_lib():
    ffibuilder = FFI()

    ffibuilder.cdef("""

    typedef struct _snt_morpheme {
        int slot_id;
        int morph_id;
        int morph_token_len;
        char * morph_token;
    } snt_morpheme_t;

    typedef struct _snt_word {
        char * pos_tag;
        char * pos_group;
        char * surface_form;
        char * raw_surface_form;
        char * stem;
        snt_morpheme_t * morphemes;
        double decode_prob;
        double tf_idf;
        int pos_tag_id;
        int word_type;
        int stem_start_index;
        int stem_end_index;
        int stem_start_slot_id;
        int stem_end_slot_id;
        int morphemes_len;
        int apostrophed;
    } snt_word_t;

    typedef struct _snt_sentence {
        snt_word_t * words;
        int words_len;
    } snt_sentence_t;

    void release_sentence(snt_sentence_t * sentences, int num_sent[1]);
    snt_sentence_t * parse_sentences_batch(const char * text, int num_sent[1]);

    void start_kinlp_lib(const char * config_file);
    void stop_kinlp_lib(void);

    int is_word_morphological(const char * word);

    """)

    ffibuilder.set_source("kinlpmorpholib",
                          """
                               #include "/home/user/projects/user/kinlp/kinlp/lib.h"
                               #include "/home/user/projects/user/kinlp/kinlp/snt.h"
                          """,
                          extra_compile_args=['-fopenmp', '-D use_openmp', '-O3', '-march=native', '-ffast-math'],
                          extra_link_args=['-fopenmp'],
                          libraries=['kinlp'])  # library name, for the linker

    ffibuilder.compile(verbose=True)

def parse_raw_text_lines(doc_lines, kb_vocab, bpe):
    from kinlpmorpholib import ffi, lib
    parsed_tokens = []
    grouped_parsed_tokens = []
    num_sent = ffi.new("int[1]")
    sentences = lib.parse_sentences_batch(' '.join(doc_lines).encode('utf-8'), num_sent)
    for i in range(num_sent[0]):
        sent = sentences[i]
        sentence_parsed_tokens = []
        for j in range(sent.words_len):
            w = sent.words[j]
            POS_TAG = ffi.string(w.pos_tag).decode("utf-8")
            WORD_TYPE = ffi.string(w.pos_group).decode("utf-8")
            STEM = ffi.string(w.stem).decode("utf-8")
            SURFACE_FORM = ffi.string(w.surface_form).decode("utf-8")
            RAW_SURFACE_FORM = ffi.string(w.raw_surface_form).decode("utf-8")
            DECODE_PROB = w.decode_prob
            IS_APOSTROPHED = not (w.apostrophed == 0)
            if IS_APOSTROPHED and (RAW_SURFACE_FORM[-1] == 'a'):
                RAW_SURFACE_FORM = RAW_SURFACE_FORM[:-1]+"'"
            RAW_SURFACE_FORM = RAW_SURFACE_FORM.replace('“','"').replace('‘‘', '"').replace('’’', '"').replace('’', '\'').replace('‘','\'')
            TF_IDF = w.tf_idf
            pti = kb_vocab.pos_tag_vocab['<UNK>']
            if POS_TAG in kb_vocab.pos_tag_vocab.keys():
               pti = kb_vocab.pos_tag_vocab[POS_TAG]

            sids = []
            morpho_tokens = []
            stem_tokens = []
            if ((STEM == SURFACE_FORM) and (w.morphemes_len <= 0)):
                list_sub_words = bpe.encode(SURFACE_FORM, output_type=yttm.OutputType.SUBWORD)
                for sub_word in list_sub_words:
                    stem_key = WORD_TYPE + ":" + sub_word
                    morpho_tokens.append(stem_key)
                    stem_tokens.append(stem_key)
                    si = kb_vocab._stem_vocab['<UNK>']
                    if stem_key in kb_vocab._stem_vocab.keys():
                        si = kb_vocab._stem_vocab[stem_key]
                    sids.append(si)
            else:
                stem_key = WORD_TYPE + ":" + STEM
                morpho_tokens.append(stem_key)
                stem_tokens.append(stem_key)
                si = kb_vocab._stem_vocab['<UNK>']
                if stem_key in kb_vocab._stem_vocab.keys():
                    si = kb_vocab._stem_vocab[stem_key]
                sids.append(si)
            ptoken = RichParsedToken(SURFACE_FORM, RAW_SURFACE_FORM, IS_APOSTROPHED, parsed_token=None, decode_prob=DECODE_PROB, tf_idf=TF_IDF, pos_tag_id=pti, stem_ids=sids)
            if (w.morphemes_len > 0):
                for k in range(w.morphemes_len):
                    if ((k != w.stem_start_index) and (k != w.stem_end_index)):
                        MORPHEME_SLOT = WORD_TYPE + ":" + str(w.morphemes[k].slot_id)
                        MORPHEME = MORPHEME_SLOT + ":" + ffi.string(w.morphemes[k].morph_token).decode("utf-8")

                        if (k < w.stem_start_index):
                            morpho_tokens = [MORPHEME] + morpho_tokens
                        elif (k < w.stem_end_index) and (len(stem_tokens) == 1) and (w.stem_start_index != w.stem_end_index):
                            # For reduplication for verbs, a bit rare
                            morpho_tokens = morpho_tokens + [MORPHEME] + stem_tokens
                        else:
                            morpho_tokens = morpho_tokens + [MORPHEME]

                        msi = kb_vocab.morpheme_slot_vocab['<UNK>']
                        if MORPHEME_SLOT in kb_vocab.morpheme_slot_vocab.keys():
                            msi = kb_vocab.morpheme_slot_vocab[MORPHEME_SLOT]

                        mi = kb_vocab.affix_vocab['<UNK>']
                        if MORPHEME in kb_vocab.affix_vocab.keys():
                            mi = kb_vocab.affix_vocab[MORPHEME]

                        ptoken.append_morpheme(msi, mi)
            ptoken.morpho_tokens.extend(morpho_tokens)
            parsed_tokens.append(ptoken)
            sentence_parsed_tokens.append(ptoken)
        grouped_parsed_tokens.append(sentence_parsed_tokens)
    lib.release_sentence(sentences, num_sent)

    return parsed_tokens, grouped_parsed_tokens

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_common_args(list_args=None, silent=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('-p', '--pos', default=1, type=int, help='number of POS embeddings for morphology')
    parser.add_argument('-s', '--stem', default=1, type=int, help='number of Stem embeddings for morphology')
    parser.add_argument('-ii', '--inference-iters', default=1, type=int, help='number of MLM Inference iterations')
    parser.add_argument('-ir', '--inference-runs', default=20, type=int,
                        help='number of MLM Inference runes per iteration')
    parser.add_argument("--use-affix-bow", type=str2bool, default=False,
                        help="Use affix embeddings sum (BOW) for morphology")
    parser.add_argument("--use-pos-aware-rel", type=str2bool, default=True,
                        help="Use POS-aware relative position embedding")
    parser.add_argument("--use-tupe-rel", type=str2bool, default=False, help="Use TUPE relative position embedding")
    parser.add_argument("--resume-from-best-saved", type=str2bool, default=False, help="Resume training from best saved model")

    parser.add_argument("--use-afsets", type=str2bool, default=False)
    parser.add_argument("--predict-affixes", type=str2bool, default=True)

    # KinyaBERT_base architecture hyper-parameters
    parser.add_argument("--seq-tr-dropout", type=float, default=0.1)
    parser.add_argument("--layernorm-epsilon", type=float, default=1e-6)
    parser.add_argument("--morpho-tr-dropout", type=float, default=0.1)
    parser.add_argument("--morpho-tr-nhead", type=int, default=4)
    parser.add_argument("--morpho-tr-nlayers", type=int, default=4)
    parser.add_argument("--morpho-dim", type=int, default=128)
    parser.add_argument("--morpho-tr-dim-feedforward", type=int, default=512)
    parser.add_argument("--stem-dim", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--seq-tr-nhead", type=int, default=12)
    parser.add_argument("--seq-tr-nlayers", type=int, default=12)
    parser.add_argument("--seq-tr-dim-feedforward", type=int, default=3072)

    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--accumulation-steps", type=int, default=128)
    parser.add_argument("--number-of-load-batches", type=int, default=384)
    parser.add_argument("--max-input-lines", type=int, default=999999)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)

    parser.add_argument("--num-iters", type=int, default=200000)
    parser.add_argument("--warmup-iter", type=int, default=2000)

    parser.add_argument("--peak-lr", type=float, default=4e-4)
    parser.add_argument("--wd", type=float, default=0.01)

    parser.add_argument("--cls-labels", type=str, default="0,1")

    parser.add_argument("--cls-train-input0", type=str, default=None)
    parser.add_argument("--cls-train-input1", type=str, default=None)
    parser.add_argument("--cls-train-label", type=str, default=None)

    parser.add_argument("--cls-dev-input0", type=str, default=None)
    parser.add_argument("--cls-dev-input1", type=str, default=None)
    parser.add_argument("--cls-dev-label", type=str, default=None)

    parser.add_argument("--cls-test-input0", type=str, default=None)
    parser.add_argument("--cls-test-input1", type=str, default=None)
    parser.add_argument("--cls-test-label", type=str, default=None)

    parser.add_argument("--pretrained-model-file", type=str, default=None)
    parser.add_argument("--devbest-cls-model-save-file-path", type=str, default=None)
    parser.add_argument("--final-cls-model-save-file-path", type=str, default=None)

    parser.add_argument("--devbest-cls-output-file", type=str, default=None)
    parser.add_argument("--final-cls-output-file", type=str, default=None)

    parser.add_argument("--home-path", type=str, default="/home/user/KINLP/")

    parser.add_argument("--regression-target", type=str2bool, default=False)
    parser.add_argument("--regression-scale-factor", type=float, default=5.0)

    parser.add_argument("--pretrained-roberta-model-dir", type=str, default="/home/user/KINLP/data/")
    parser.add_argument("--pretrained-roberta-checkpoint-file", type=str, default="checkpoint_best.pt")
    parser.add_argument("--xlmr", type=str2bool, default=False)

    parser.add_argument("--pooler-dropout", type=float, default=0.1)

    parser.add_argument("--embed-dim", type=int, default=768)

    parser.add_argument("--inference-model-file", type=str, default=None)

    parser.add_argument("--model-keyword", type=str, default=None)
    parser.add_argument("--task-keyword", type=str, default=None)
    parser.add_argument("--input-format", type=str, default=None)

    parser.add_argument("--afset-dict-size", type=int, default=10000)
    parser.add_argument("--paired-encoder", type=str2bool, default=False)

    parser.add_argument("--debug", type=str2bool, default=False)

    parser.add_argument("--use-morpho-encoder", type=str2bool, default=True)

    parser.add_argument("--exploratory-model-load", type=str, default=None)

    parser.add_argument("-f", type=str, default=None)


    if list_args is not None:
        args = parser.parse_args(list_args)
    else:
        args = parser.parse_args()

    args.world_size = args.gpus
    args.num_pos_m_embeddings = args.pos
    args.num_stem_m_embeddings = args.stem
    args.use_affix_bow_m_embedding = args.use_affix_bow
    args.use_pos_aware_rel_pos_bias = args.use_pos_aware_rel
    args.use_tupe_rel_pos_bias = args.use_tupe_rel
    args.num_inference_iters = args.inference_iters
    args.num_inference_runs = args.inference_runs

    if not silent:
        print('Call arguments:\n', args)

    return args

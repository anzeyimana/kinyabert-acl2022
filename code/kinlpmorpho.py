# Copyright (c) Antoine Nzeyimana.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from cffi import FFI


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

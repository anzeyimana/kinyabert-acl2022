from __future__ import print_function, division
import torch
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from morpho_transformer import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from paired_transformer import PairedTransformerEncoder, PairedTransformerEncoderLayer, PairedMultiheadAttention
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, PairedMultiheadAttention):
        module.in_proj_weight_for_x.data.normal_(mean=0.0, std=0.02)
        module.in_proj_weight_for_y.data.normal_(mean=0.0, std=0.02)

# From: https://github.com/guolinke/TUPE/blob/master/fairseq/modules/transformer_sentence_encoder.py
# this is from T5
def tupe_relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertHeadTransform(nn.Module):
    def __init__(self, tr_d_model, cls_ctxt_size, layernorm_epsilon):
        super(BertHeadTransform, self).__init__()
        self.dense = nn.Linear(tr_d_model, cls_ctxt_size)
        self.layerNorm = BertLayerNorm(cls_ctxt_size, eps=layernorm_epsilon)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        hidden_states = self.layerNorm(hidden_states)
        return hidden_states

class TokenClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.3):
        super(TokenClassificationHead, self).__init__()
        self.input_dim = input_dim

        # Experimental
        #------------------
        self.dense = nn.Linear(input_dim, inner_dim)
        self.layerNorm = torch.nn.LayerNorm(inner_dim)
        self.activation_fn = torch.tanh
        self.in_dropout = nn.Dropout(p=pooler_dropout)
        #------------------

        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, features, input_sequence_lengths):
        # features.shape = S x N x E
        # Remove [CLS]
        # len already includes [CLS] in the sequence length count, so number of normal tokens here is (len-1)
        inputs = [features[1:len, i, :].contiguous().view(-1, self.input_dim) for i,len in enumerate(input_sequence_lengths)]
        x = torch.cat(inputs, 0) #  B x E

        # Experimental
        #------------------
        x = self.in_dropout(x)
        x = self.dense(x)
        x = self.layerNorm(x)
        x = self.activation_fn(x)
        #------------------

        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.0):
        super(ClassificationHead, self).__init__()
        self.input_dim = input_dim

        # Experimental
        #------------------
        self.dense = nn.Linear(input_dim, inner_dim)
        self.layerNorm = torch.nn.LayerNorm(inner_dim)
        self.activation_fn = torch.tanh
        self.in_dropout = nn.Dropout(p=pooler_dropout)
        # ------------------

        self.out_dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        self.apply(init_bert_params)

    def forward(self, features):
        # features.shape = S x N x E
        x = features[0, :, :]  # Take [CLS]

        # Experimental
        # ------------------
        x = self.in_dropout(x)
        x = self.dense(x)
        x = self.layerNorm(x)
        x = self.activation_fn(x)
        # ------------------

        x = self.out_dropout(x)
        x = self.out_proj(x)
        return x

class MorphoHeadPredictor(nn.Module):
    def __init__(self, args, stem_embedding_weights, afset_embedding_weights, affix_embedding_weights, tr_d_model, tr_dropout, layernorm_epsilon):
        super(MorphoHeadPredictor, self).__init__()

        self.stem_transform = BertHeadTransform(tr_d_model, stem_embedding_weights.size(1), layernorm_epsilon)
        self.stem_decoder = nn.Linear(stem_embedding_weights.size(1), stem_embedding_weights.size(0), bias=False)
        self.stem_decoder.weight = stem_embedding_weights
        self.stem_decoder_bias = nn.Parameter(torch.zeros(stem_embedding_weights.size(0)))

        if args.use_afsets:
            self.afset_transform = BertHeadTransform(tr_d_model, afset_embedding_weights.size(1), layernorm_epsilon)
            self.afset_decoder = nn.Linear(afset_embedding_weights.size(1), afset_embedding_weights.size(0), bias=False)
            self.afset_decoder.weight = afset_embedding_weights
            self.afset_decoder_bias = nn.Parameter(torch.zeros(afset_embedding_weights.size(0)))

        if args.predict_affixes:
            self.affix_transform = BertHeadTransform(tr_d_model, affix_embedding_weights.size(1), layernorm_epsilon)
            self.affix_decoder = nn.Linear(affix_embedding_weights.size(1), affix_embedding_weights.size(0), bias=False)
            self.affix_decoder.weight = affix_embedding_weights
            self.affix_decoder_bias = nn.Parameter(torch.zeros(affix_embedding_weights.size(0)))

        self.apply(init_bert_params)

    @custom_fwd
    def forward(self, args, tr_hidden_state,
                predicted_tokens_idx,
                predicted_tokens_affixes_idx,
                predicted_stems,
                predicted_afsets,
                predicted_affixes_prob):

        # print('tr_hidden_state.shape',tr_hidden_state.shape)
        # 1. Crop together predicted tokens
        # tr_hidden_state.shape = S x N x E
        token_hidden_state = tr_hidden_state.permute(1,0,2)
        # N x S x E
        # print('token_hidden_state.shape',token_hidden_state.shape)
        token_hidden_state = token_hidden_state.reshape(-1, token_hidden_state.shape[2])
        # predicted_state.shape: NS x E or B x E
        # print('token_hidden_state.shape',token_hidden_state.shape)
        # print('batch_predicted_token_idx',batch_predicted_token_idx)
        token_hidden_state = torch.index_select(token_hidden_state, 0, index=predicted_tokens_idx)
        # predicted_state.shape: B x E
        stem_predicted_state = self.stem_transform(token_hidden_state)
        # predicted_state.shape: B x F ==> Only containing the 15% tokens to be predicted

        # 3. Propagate to Stem Prediction
        stem_scores = self.stem_decoder(stem_predicted_state) + self.stem_decoder_bias
        stem_scores = F.log_softmax(stem_scores, dim=1)
        stem_loss_avg = F.nll_loss(stem_scores, predicted_stems)

        loss = torch.tensor(0.0).to(stem_loss_avg.device)
        loss += stem_loss_avg

        # 3*. Propagate to AffixSet Prediction
        afset_loss_avg = torch.tensor(0.0).to(stem_loss_avg.device)
        if args.use_afsets and (predicted_afsets is not None):
            afset_predicted_state = self.afset_transform(token_hidden_state)
            afset_scores = self.afset_decoder(afset_predicted_state) + self.afset_decoder_bias
            afset_scores = F.log_softmax(afset_scores, dim=1)
            afset_loss_avg = F.nll_loss(afset_scores, predicted_afsets)
            loss += afset_loss_avg

        # 6. Propagate to Affix prediction
        affix_loss_avg = torch.tensor(0.0).to(stem_loss_avg.device)
        if args.predict_affixes and (predicted_tokens_affixes_idx is not None) and (predicted_affixes_prob is not None):
            if predicted_tokens_affixes_idx.nelement() > 0:
                affix_hidden_state = torch.index_select(token_hidden_state, 0, index=predicted_tokens_affixes_idx)
                affix_predicted_state = self.affix_transform(affix_hidden_state)
                affix_scores = self.affix_decoder(affix_predicted_state) + self.affix_decoder_bias
                affix_scores = F.log_softmax(affix_scores, dim=1)
                affix_loss_avg = F.kl_div(affix_scores, predicted_affixes_prob, reduction='batchmean')

        loss += affix_loss_avg

        return loss, stem_loss_avg, afset_loss_avg, affix_loss_avg

    def predict(self, args, tr_hidden_state,
                seq_predicted_token_idx,
                max_predict_affixes, proposed_stem_ids=None):
        # print('tr_hidden_state.shape',tr_hidden_state.shape)
        # 1. Crop together predicted tokens
        # tr_hidden_state.shape = S x N x E
        token_hidden_state = tr_hidden_state.permute(1,0,2)
        # N x S x E
        # print('token_hidden_state.shape',token_hidden_state.shape)
        token_hidden_state = token_hidden_state.reshape(-1, token_hidden_state.shape[2])
        # predicted_state.shape: NS x E = B x E
        # print('token_hidden_state.shape',token_hidden_state.shape)
        # print('batch_predicted_token_idx',batch_predicted_token_idx)
        token_hidden_state = torch.index_select(token_hidden_state, 0, index=seq_predicted_token_idx)
        # predicted_state.shape: B x E
        stem_predicted_state = self.stem_transform(token_hidden_state)
        # predicted_state.shape: B x F ==> Only containing the 15% tokens to be predicted

        # 3. Propagate to Stem Prediction
        stem_scores = self.stem_decoder(stem_predicted_state) + self.stem_decoder_bias
        stem_scores = F.softmax(stem_scores, dim=1)
        # stem_predictions = stem_scores.argmax(dim=1)
        if proposed_stem_ids is not None:
            selection = stem_scores[:, proposed_stem_ids]
            stem_predictions_prob, pred_ids = selection.max(dim=1)
            pred_list = pred_ids.tolist()
            stem_predictions = torch.tensor([proposed_stem_ids[ii] for ii in pred_list])
        else:
            stem_predictions_prob, stem_predictions = stem_scores.max(dim=1)

        afset_predictions = None
        afset_predictions_prob = None
        if args.use_afsets:
            afset_predicted_state = self.afset_transform(token_hidden_state)
            afset_scores = self.afset_decoder(afset_predicted_state) + self.afset_decoder_bias
            afset_scores = F.softmax(afset_scores, dim=1)
            afset_predictions_prob, afset_predictions = afset_scores.max(dim=1)

        # 5. Predict affix
        affix_predictions = None
        if args.predict_affixes:
            affix_predicted_state = self.affix_transform(token_hidden_state)
            affix_scores = self.affix_decoder(affix_predicted_state) + self.affix_decoder_bias
            affix_scores = F.log_softmax(affix_scores, dim=1)
            _, top_affixes = torch.topk(affix_scores, max_predict_affixes, dim=1)
            affix_predictions = []
            for batch in range(top_affixes.shape[0]):
                affix_predictions.append(top_affixes[batch].tolist())

        return stem_predictions, stem_predictions_prob, afset_predictions, afset_predictions_prob, affix_predictions

class KinyaBERT_MorphoEncoder(nn.Module):
    def __init__(self, args,
                 num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_pos_aware_rel_pos_dict_size,
                 num_pos_m_embeddings,
                 num_stem_m_embeddings,
                 use_affix_bow_m_embedding,
                 use_pos_aware_rel_pos_bias,
                 use_tupe_rel_pos_bias,
                 max_seq_len = 512,
                 morpho_dim = 80, stem_dim = 160,
                 morpho_tr_nhead = 4, morpho_tr_nlayers=4,
                 morpho_tr_dim_feedforward=512, morpho_tr_dropout=0.1, morpho_tr_activation='gelu',
                 seq_tr_nhead=8, seq_tr_nlayers=8,
                 seq_tr_dim_feedforward=2048, seq_tr_dropout=0.1, seq_tr_activation='gelu',
                 tupe_rel_pos_bins: int = 32,
                 tupe_max_rel_pos: int = 128):
        super(KinyaBERT_MorphoEncoder, self).__init__()

        self.seq_tr_nhead = seq_tr_nhead
        self.max_seq_len = max_seq_len
        self.tot_num_affixes = num_affixes
        self.num_pos_m_embeddings = num_pos_m_embeddings
        self.num_stem_m_embeddings = num_stem_m_embeddings
        self.use_affix_bow_m_embedding = use_affix_bow_m_embedding

        self.seq_tr_d_model = stem_dim
        self.morpho_dim = morpho_dim
        self.attn_scale_factor = 2

        if args.paired_encoder:
            assert self.seq_tr_d_model == self.morpho_dim, "Dimensions of morpho encoder and sentence encoder mismatch"
            self.attn_scale_factor = 5 # 5 factors for attn_i_j: stem_i->stem_j, stem_i->afset_j, afset_i->afset_j, afset_i->stem_j, pos_i->pos_j

            self.m_afset_embedding = nn.Embedding(num_afsets, self.morpho_dim, padding_idx=0)
            self.m_stem_embedding = nn.Embedding(num_stems, self.morpho_dim, padding_idx=0)
            self.m_affix_embedding = nn.Embedding(num_affixes, self.morpho_dim, padding_idx=0)
            self.tot_morpho_idx = 2

            morpho_encoder_layers = TransformerEncoderLayer(self.morpho_dim, morpho_tr_nhead, dim_feedforward=morpho_tr_dim_feedforward, dropout=morpho_tr_dropout, activation=morpho_tr_activation)
            self.morpho_transformer_encoder = TransformerEncoder(morpho_encoder_layers, morpho_tr_nlayers)

            sequence_encoder_layers = PairedTransformerEncoderLayer(self.seq_tr_d_model, self.seq_tr_nhead, attn_scale_factor=self.attn_scale_factor, dim_feedforward=seq_tr_dim_feedforward, dropout=seq_tr_dropout, activation=seq_tr_activation)
            self.seq_paired_transformer_encoder = PairedTransformerEncoder(sequence_encoder_layers, seq_tr_nlayers)
        else:
            self.tot_morpho_idx = 0
            if self.num_pos_m_embeddings > 0:
                self.m1_pos_embedding = nn.Embedding(num_pos_tags, self.morpho_dim, padding_idx=0)
                self.tot_morpho_idx += 1

            if self.num_pos_m_embeddings > 1:
                self.m2_pos_embedding = nn.Embedding(num_pos_tags, self.morpho_dim, padding_idx=0)
                self.tot_morpho_idx += 1

            if self.num_pos_m_embeddings > 2:
                self.m3_pos_embedding = nn.Embedding(num_pos_tags, self.morpho_dim, padding_idx=0)
                self.tot_morpho_idx += 1

            if self.num_stem_m_embeddings > 0:
                self.m_stem_embedding = nn.Embedding(num_stems, self.morpho_dim, padding_idx=0)
                self.tot_morpho_idx += 1

            if args.use_afsets and (num_afsets > 0):
                self.m_afset_embedding = nn.Embedding(num_afsets, self.morpho_dim, padding_idx=0)
                self.tot_morpho_idx += 1

            self.seq_tr_d_model += (self.morpho_dim * self.tot_morpho_idx)

            if self.use_affix_bow_m_embedding:
                self.seq_tr_d_model += self.morpho_dim

            self.s_stem_embedding = nn.Embedding(num_stems, stem_dim, padding_idx=0)

            if args.use_morpho_encoder and (self.seq_tr_d_model > stem_dim):
                self.m_affix_embedding = nn.Embedding(num_affixes, self.morpho_dim, padding_idx=0)
                morpho_encoder_layers = TransformerEncoderLayer(self.morpho_dim, morpho_tr_nhead, dim_feedforward=morpho_tr_dim_feedforward, dropout=morpho_tr_dropout, activation=morpho_tr_activation)
                self.morpho_transformer_encoder = TransformerEncoder(morpho_encoder_layers, morpho_tr_nlayers)

            sequence_encoder_layers = TransformerEncoderLayer(self.seq_tr_d_model, self.seq_tr_nhead, dim_feedforward=seq_tr_dim_feedforward, dropout=seq_tr_dropout, activation=seq_tr_activation)
            self.seq_transformer_encoder = TransformerEncoder(sequence_encoder_layers, seq_tr_nlayers)

        self.use_pos_aware_rel_pos_bias = use_pos_aware_rel_pos_bias
        if self.use_pos_aware_rel_pos_bias:
            self.rel_pos_embedding = nn.Embedding(num_pos_aware_rel_pos_dict_size+1, self.seq_tr_nhead, padding_idx=0)

        # This is from TUPE
        self.pos = nn.Embedding(self.max_seq_len + 1, self.seq_tr_d_model)
        self.pos_q_linear = nn.Linear(self.seq_tr_d_model, self.seq_tr_d_model)
        self.pos_k_linear = nn.Linear(self.seq_tr_d_model, self.seq_tr_d_model)
        self.pos_scaling = float(self.seq_tr_d_model / self.seq_tr_nhead * self.attn_scale_factor) ** -0.5
        self.pos_ln = nn.LayerNorm(self.seq_tr_d_model)

        self.use_tupe_rel_pos_bias = use_tupe_rel_pos_bias
        if self.use_tupe_rel_pos_bias:
            assert tupe_rel_pos_bins % 2 == 0
            self.tupe_rel_pos_bins = tupe_rel_pos_bins
            self.tupe_max_rel_pos = tupe_max_rel_pos
            self.relative_attention_bias = nn.Embedding(self.tupe_rel_pos_bins + 1, self.seq_tr_nhead)
            seq_len = self.max_seq_len
            context_position = torch.arange(seq_len, dtype=torch.long)[:, None]
            memory_position = torch.arange(seq_len, dtype=torch.long)[None, :]
            relative_position = memory_position - context_position
            self.rp_bucket = tupe_relative_position_bucket(
                relative_position,
                num_buckets=self.tupe_rel_pos_bins,
                max_distance=self.tupe_max_rel_pos
            )
            # others to [CLS]
            self.rp_bucket[:, 0] = self.tupe_rel_pos_bins
            # [CLS] to others, Note: self.tupe_rel_pos_bins // 2 is not used in relative_position_bucket
            self.rp_bucket[0, :] = self.tupe_rel_pos_bins // 2

        self.apply(init_bert_params)

    def get_tupe_rel_pos_bias(self, seq_len, device):
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        if self.rp_bucket.device != device:
            self.rp_bucket = self.rp_bucket.to(device)
        # Adjusted because final x's shape is L x B X E
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_pos_aware_rel_pos_bias(self, rel_pos_arr, seq_len):
        rel_pos_idx = rel_pos_arr[:, :seq_len, :seq_len]
        rel_pos = self.rel_pos_embedding(rel_pos_idx)
        # N x L x L x h --> N x h x L x L
        rel_pos = rel_pos.permute([0, 3, 1, 2])
        rel_pos = rel_pos.contiguous()
        rel_pos = rel_pos.reshape(-1, seq_len, seq_len)
        return rel_pos.contiguous()

    def get_position_attn_bias(self, rel_pos_arr, seq_len, batch_size, device):
        tupe_rel_pos_bias = self.get_tupe_rel_pos_bias(seq_len, device) if self.use_tupe_rel_pos_bias else None
        pos_aware_rel_pos_bias = self.get_pos_aware_rel_pos_bias(rel_pos_arr, seq_len) if self.use_pos_aware_rel_pos_bias else None

        # This is from TUPE
        # https://github.com/guolinke/TUPE/blob/master/fairseq/modules/transformer_sentence_encoder.py
        # 0 is for other-to-cls 1 is for cls-to-other
        # Assume the input is ordered. If your input token is permuted, you may need to update this accordingly
        weight = self.pos_ln(self.pos.weight[:seq_len + 1, :])
        pos_q =  self.pos_q_linear(weight).view(seq_len + 1, self.seq_tr_nhead, -1).transpose(0, 1) * self.pos_scaling
        pos_k =  self.pos_k_linear(weight).view(seq_len + 1, self.seq_tr_nhead, -1).transpose(0, 1)
        abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
        # p_0 \dot p_0 is cls to others
        cls_2_other = abs_pos_bias[:, 0, 0]
        # p_1 \dot p_1 is others to cls
        other_2_cls = abs_pos_bias[:, 1, 1]
        # offset
        abs_pos_bias = abs_pos_bias[:, 1:, 1:]
        abs_pos_bias[:, :, 0] = other_2_cls.view(-1, 1)
        abs_pos_bias[:, 0, :] = cls_2_other.view(-1, 1)

        if tupe_rel_pos_bias is not None:
            abs_pos_bias += tupe_rel_pos_bias

        abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(batch_size, -1, -1, -1).reshape(-1, seq_len, seq_len)

        if pos_aware_rel_pos_bias is not None:
            abs_pos_bias += pos_aware_rel_pos_bias

        return abs_pos_bias


    @custom_fwd
    def forward(self, args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes):
        # L = len(tokens_lengths)
        # affixes: (N)
        # stems: (L)
        # pos_tags: (L)

        device = stems.device
        if args.paired_encoder:
            xm_stem = self.m_stem_embedding(stems)
            xm_stem = torch.unsqueeze(xm_stem, 0)

            xm_afset = self.m_afset_embedding(afsets)
            xm_afset = torch.unsqueeze(xm_afset, 0)

            x_embed = torch.cat((xm_stem, xm_afset), 0)
            # x_embed: (2,L,E)
            afx = affixes.split(tokens_lengths)
            # [[2,4,5], [6,7]]
            afx_padded = pad_sequence(afx, batch_first=False)
            # afx_padded: (M,L), M: max morphological length
            m_masks_padded = None
            has_morphemes = False
            if afx_padded.nelement() > 0:
                has_morphemes = True
                xm_affix = self.m_affix_embedding(afx_padded)
                # xm_affix: (M,L,E)
                x_embed = torch.cat((x_embed, xm_affix), 0)
                # x_embed: (2+M,L,E)
                m_masks = [torch.zeros((x+(self.tot_morpho_idx)), dtype=torch.bool, device = device) for x in tokens_lengths]
                m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1) # Shape: (L, 2+M)

            morpho_transformer_output = self.morpho_transformer_encoder(x_embed, src_key_padding_mask=m_masks_padded)  # --> Shape: (2+M, L, E)
            stems_morpho_encoded = morpho_transformer_output[0, :, :] # (L,E)
            afsets_morpho_encoded = morpho_transformer_output[1, :, :] # (L,E)

            stem_lists = stems_morpho_encoded.split(input_sequence_lengths, 0)  # len(input_sequence_lengths) = N (i.e. Batch Size, e.g. 32)
            src_stems = pad_sequence(stem_lists, batch_first=False)

            afset_lists = afsets_morpho_encoded.split(input_sequence_lengths, 0)  # len(input_sequence_lengths) = N (i.e. Batch Size, e.g. 32)
            src_afsets = pad_sequence(afset_lists, batch_first=False)

            seq_len = src_stems.size(0)
            batch_size = src_stems.size(1)

            abs_pos_bias = self.get_position_attn_bias(rel_pos_arr, seq_len, batch_size, device)

            masks = [torch.zeros(x, dtype=torch.bool, device=device) for x in input_sequence_lengths]
            masks_padded = pad_sequence(masks, batch_first=True, padding_value=1)  # Shape: N x S

            src_stems, src_afsets  = self.seq_paired_transformer_encoder(src_stems, src_afsets, attn_bias=abs_pos_bias, src_key_padding_mask=masks_padded)  # --> Shape: L x N x E, with L = max sequence length
            # --> (L, N, E) = L: Max Sequence Length, N: Batch Size, E: Embedding dimension
            transformer_output = torch.cat((src_stems, src_afsets), 2) # --> (L, N, 2E)
            return transformer_output

        x_embed = None
        if self.num_pos_m_embeddings > 0:
            xm_pos1 = self.m1_pos_embedding(pos_tags)
            x_embed = torch.unsqueeze(xm_pos1,0)

        if self.num_pos_m_embeddings > 1:
            xm_pos2 = self.m2_pos_embedding(pos_tags)
            xm_pos2 = torch.unsqueeze(xm_pos2,0)
            x_embed = torch.cat((x_embed, xm_pos2), 0)

        if self.num_pos_m_embeddings > 2:
            xm_pos3 = self.m3_pos_embedding(pos_tags)
            xm_pos3 = torch.unsqueeze(xm_pos3,0)
            x_embed = torch.cat((x_embed, xm_pos3), 0)

        if self.num_stem_m_embeddings > 0:
            xm_stem = self.m_stem_embedding(stems)
            xm_stem = torch.unsqueeze(xm_stem,0)
            if x_embed is not None:
                x_embed = torch.cat((x_embed, xm_stem), 0)
            else:
                x_embed = xm_stem

        if (args.use_afsets) and (afsets is not None):
            xm_afset = self.m_afset_embedding(afsets)
            xm_afset = torch.unsqueeze(xm_afset,0)
            if x_embed is not None:
                x_embed = torch.cat((x_embed, xm_afset), 0)
            else:
                x_embed = xm_afset
        # All above: (1,L,E)

        #x_embed = torch.cat((xm_pos1, xm_pos2, xm_pos3, xm_stem, xm_afset), 0)
        # x_embed: (4,L,E)

        if args.use_morpho_encoder:
            afx = affixes.split(tokens_lengths)
            # [[2,4,5], [6,7]]
            afx_padded = pad_sequence(afx, batch_first=False)
            # afx_padded: (M,L), M: max morphological length
            m_masks_padded = None
            has_morphemes = False
            if afx_padded.nelement() > 0:
                has_morphemes = True
                xm_affix = self.m_affix_embedding(afx_padded)
                # xa_embed: (M,L,E)
                if x_embed is not None:
                    x_embed = torch.cat((x_embed, xm_affix), 0)
                else:
                    x_embed = xm_affix

                m_masks = [torch.zeros((x+(self.tot_morpho_idx)), dtype=torch.bool, device = device) for x in tokens_lengths]
                m_masks_padded = pad_sequence(m_masks, batch_first=True, padding_value=1) # Shape: (L, M+4)

        # x_embed: (4+M,L,E)
        morpho_input = None
        if (x_embed is not None) and args.use_morpho_encoder:
            morpho_transformer_output = self.morpho_transformer_encoder(x_embed, src_key_padding_mask=m_masks_padded)  # Shape: (M+4, L, E)
            if (self.tot_morpho_idx > 0) and (self.use_affix_bow_m_embedding):
                heads = morpho_transformer_output[:(self.tot_morpho_idx), :, :]
                affixes_bow = torch.sum(morpho_transformer_output[(self.tot_morpho_idx):, :, :], 0, keepdim=True) if has_morphemes else torch.zeros((1, stems.size(0), self.morpho_dim), device = device)
                morpho_input = torch.cat((heads, affixes_bow), 0)
            elif (self.tot_morpho_idx > 0):
                morpho_input = morpho_transformer_output[:(self.tot_morpho_idx), :, :]
            elif (self.use_affix_bow_m_embedding):
                morpho_input = torch.sum(morpho_transformer_output[(self.tot_morpho_idx):, :, :], 0, keepdim=True) if has_morphemes else torch.zeros((1, stems.size(0), self.morpho_dim), device = device)
        elif (self.use_affix_bow_m_embedding):
            # No POS, no STEM, No afset, No affixes, but allows affixes_bow
            morpho_input = torch.zeros((1, stems.size(0), self.morpho_dim), device = device)

        input_sequences = self.s_stem_embedding(stems) # (L, E')
        if morpho_input is not None:
            # 4 x L x E ==> L x 4 x E
            #i.e. K x L x E ==> L x K x E, K: number of morpho components from the morpho encoder tier
            morpho_input = morpho_input.permute(1, 0, 2)
            L = morpho_input.size(0)
            morpho_input = morpho_input.contiguous().view(L,-1) # (L, 4E), i.e. (L, KE)
            input_sequences = torch.cat((morpho_input, input_sequences), 1)

        lists = input_sequences.split(input_sequence_lengths, 0) # len(input_sequence_lengths) = N (i.e. Batch Size, e.g. 32)
        tr_padded = pad_sequence(lists, batch_first=False)

        seq_len = tr_padded.size(0)
        batch_size = tr_padded.size(1)

        abs_pos_bias = self.get_position_attn_bias(rel_pos_arr, seq_len, batch_size, device)

        masks = [torch.zeros(x, dtype=torch.bool, device = device) for x in input_sequence_lengths]
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=1) # Shape: N x S

        transformer_output = self.seq_transformer_encoder(tr_padded, attn_bias = abs_pos_bias, src_key_padding_mask = masks_padded) # Shape: L x N x E, with L = max sequence length

        return transformer_output


class KinyaBERT(nn.Module):
    def __init__(self, args,
                 num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 num_pos_m_embeddings,
                 num_stem_m_embeddings,
                 use_affix_bow_m_embedding,
                 use_pos_aware_rel_pos_bias,
                 use_tupe_rel_pos_bias,
                 max_seq_len = 512,
                 morpho_dim=80, stem_dim=160,
                 morpho_tr_nhead=4, morpho_tr_nlayers=4,
                 morpho_tr_dim_feedforward=512, morpho_tr_dropout=0.1, morpho_tr_activation='gelu',
                 seq_tr_nhead=8, seq_tr_nlayers=8,
                 seq_tr_dim_feedforward=2048, seq_tr_dropout=0.1, seq_tr_activation='gelu',
                 layernorm_epsilon = 1e-6,
                 tupe_rel_pos_bins: int = 32,
                 tupe_max_rel_pos: int = 128):
        super(KinyaBERT, self).__init__()

        self.encoder = KinyaBERT_MorphoEncoder(args, num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 num_pos_m_embeddings,
                 num_stem_m_embeddings,
                 use_affix_bow_m_embedding,
                 use_pos_aware_rel_pos_bias,
                 use_tupe_rel_pos_bias,
                 max_seq_len = max_seq_len,
                 morpho_dim = morpho_dim, stem_dim = stem_dim,
                 morpho_tr_nhead = morpho_tr_nhead, morpho_tr_nlayers=morpho_tr_nlayers,
                 morpho_tr_dim_feedforward=morpho_tr_dim_feedforward, morpho_tr_dropout=morpho_tr_dropout, morpho_tr_activation=morpho_tr_activation,
                 seq_tr_nhead=seq_tr_nhead, seq_tr_nlayers=seq_tr_nlayers,
                 seq_tr_dim_feedforward=seq_tr_dim_feedforward, seq_tr_dropout=seq_tr_dropout, seq_tr_activation=seq_tr_activation,
                 tupe_rel_pos_bins = tupe_rel_pos_bins,
                 tupe_max_rel_pos = tupe_max_rel_pos)

        if args.paired_encoder:
            self.predictor = MorphoHeadPredictor(args, self.encoder.m_stem_embedding.weight,
                                                 self.encoder.m_afset_embedding.weight if (num_afsets > 0) else None,
                                                 self.encoder.m_affix_embedding.weight if args.predict_affixes else None,
                                                 self.encoder.seq_tr_d_model*2, seq_tr_dropout, layernorm_epsilon)
        else:
            self.predictor = MorphoHeadPredictor(args, self.encoder.s_stem_embedding.weight,
                                                 self.encoder.m_afset_embedding.weight if (num_afsets > 0) else None,
                                                 self.encoder.m_affix_embedding.weight if args.predict_affixes else None,
                                                 self.encoder.seq_tr_d_model, seq_tr_dropout, layernorm_epsilon)

    @custom_fwd
    def forward(self, args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes,
                predicted_tokens_idx,
                predicted_tokens_affixes_idx,
                predicted_stems,
                predicted_afsets,
                predicted_affixes_prob):

        tr_hidden_state = self.encoder(args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes)

        return self.predictor(args, tr_hidden_state,
                              predicted_tokens_idx,
                              predicted_tokens_affixes_idx,
                              predicted_stems,
                              predicted_afsets,
                              predicted_affixes_prob)

    def predict(self, args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes,
                seq_predicted_token_idx,
                max_predict_affixes, proposed_stem_ids=None):

        tr_hidden_state = self.encoder(args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes)

        return self.predictor.predict(args, tr_hidden_state,
                seq_predicted_token_idx,
                max_predict_affixes, proposed_stem_ids=proposed_stem_ids)

class KinyaBERTClassifier(nn.Module):
    def __init__(self, args,
                 num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 num_classes,
                 num_pos_m_embeddings,
                 num_stem_m_embeddings,
                 use_affix_bow_m_embedding,
                 use_pos_aware_rel_pos_bias,
                 use_tupe_rel_pos_bias,
                 max_seq_len = 512,
                 morpho_dim=80, stem_dim=160,
                 morpho_tr_nhead=4, morpho_tr_nlayers=4,
                 morpho_tr_dim_feedforward=512, morpho_tr_dropout=0.1, morpho_tr_activation='gelu',
                 seq_tr_nhead=8, seq_tr_nlayers=8,
                 seq_tr_dim_feedforward=2048, seq_tr_dropout=0.1, pooler_dropout=0.0, seq_tr_activation='gelu',
                 tupe_rel_pos_bins: int = 32,
                 tupe_max_rel_pos: int = 128):
        super(KinyaBERTClassifier, self).__init__()

        self.encoder = KinyaBERT_MorphoEncoder(args, num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 num_pos_m_embeddings,
                 num_stem_m_embeddings,
                 use_affix_bow_m_embedding,
                 use_pos_aware_rel_pos_bias,
                 use_tupe_rel_pos_bias,
                 max_seq_len = max_seq_len,
                 morpho_dim = morpho_dim, stem_dim = stem_dim,
                 morpho_tr_nhead = morpho_tr_nhead, morpho_tr_nlayers=morpho_tr_nlayers,
                 morpho_tr_dim_feedforward=morpho_tr_dim_feedforward, morpho_tr_dropout=morpho_tr_dropout, morpho_tr_activation=morpho_tr_activation,
                 seq_tr_nhead=seq_tr_nhead, seq_tr_nlayers=seq_tr_nlayers,
                 seq_tr_dim_feedforward=seq_tr_dim_feedforward, seq_tr_dropout=seq_tr_dropout, seq_tr_activation=seq_tr_activation,
                 tupe_rel_pos_bins = tupe_rel_pos_bins,
                 tupe_max_rel_pos = tupe_max_rel_pos)

        if args.paired_encoder:
            self.cls_head = ClassificationHead(self.encoder.seq_tr_d_model * 2, num_classes * 32, num_classes, pooler_dropout=pooler_dropout)
        else:
            self.cls_head = ClassificationHead(self.encoder.seq_tr_d_model, num_classes * 32, num_classes, pooler_dropout=pooler_dropout)

    @custom_fwd
    def forward(self, args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes):
        tr_hidden_state = self.encoder(args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes)
        return self.cls_head(tr_hidden_state)

class KinyaBERTSequenceTagger(nn.Module):
    def __init__(self, args,
                 num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 num_classes,
                 num_pos_m_embeddings,
                 num_stem_m_embeddings,
                 use_affix_bow_m_embedding,
                 use_pos_aware_rel_pos_bias,
                 use_tupe_rel_pos_bias,
                 max_seq_len = 512,
                 morpho_dim=80, stem_dim=160,
                 morpho_tr_nhead=4, morpho_tr_nlayers=4,
                 morpho_tr_dim_feedforward=512, morpho_tr_dropout=0.1, morpho_tr_activation='gelu',
                 seq_tr_nhead=8, seq_tr_nlayers=8,
                 seq_tr_dim_feedforward=2048, seq_tr_dropout=0.1, pooler_dropout=0.0, seq_tr_activation='gelu',
                 tupe_rel_pos_bins: int = 32,
                 tupe_max_rel_pos: int = 128):
        super(KinyaBERTSequenceTagger, self).__init__()

        self.encoder = KinyaBERT_MorphoEncoder(args, num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 num_pos_m_embeddings,
                 num_stem_m_embeddings,
                 use_affix_bow_m_embedding,
                 use_pos_aware_rel_pos_bias,
                 use_tupe_rel_pos_bias,
                 max_seq_len = max_seq_len,
                 morpho_dim = morpho_dim, stem_dim = stem_dim,
                 morpho_tr_nhead = morpho_tr_nhead, morpho_tr_nlayers=morpho_tr_nlayers,
                 morpho_tr_dim_feedforward=morpho_tr_dim_feedforward, morpho_tr_dropout=morpho_tr_dropout, morpho_tr_activation=morpho_tr_activation,
                 seq_tr_nhead=seq_tr_nhead, seq_tr_nlayers=seq_tr_nlayers,
                 seq_tr_dim_feedforward=seq_tr_dim_feedforward, seq_tr_dropout=seq_tr_dropout, seq_tr_activation=seq_tr_activation,
                 tupe_rel_pos_bins = tupe_rel_pos_bins,
                 tupe_max_rel_pos = tupe_max_rel_pos)

        if args.paired_encoder:
            self.cls_head = TokenClassificationHead(self.encoder.seq_tr_d_model*2, num_classes * 32, num_classes, pooler_dropout=pooler_dropout)
        else:
            self.cls_head = TokenClassificationHead(self.encoder.seq_tr_d_model, num_classes * 32, num_classes, pooler_dropout=pooler_dropout)

    @custom_fwd
    def forward(self, args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes):
        tr_hidden_state = self.encoder(args, rel_pos_arr, tokens_lengths, input_sequence_lengths, pos_tags, stems, afsets, affixes)
        return self.cls_head(tr_hidden_state, input_sequence_lengths)

from morpho_data_loaders import KBVocab, AffixSetVocab

def kinyabert_base(kb_vocab : KBVocab, affix_set_vocab : AffixSetVocab, rel_pos_dict, device, args, saved_model_file = None) -> KinyaBERT:
    num_pos_tags = len(kb_vocab.pos_tag_vocab) + 1
    num_stems = len(kb_vocab.reduced_stem_vocab) + 1
    num_afsets = (len(affix_set_vocab.affix_set_vocab_idx) + 1) if args.use_afsets else 0
    num_affixes = len(kb_vocab.affix_vocab) + 1

    num_rel_pos_dict_size = (len(rel_pos_dict)+1) if (rel_pos_dict is not None) else 0

    activation_fn = 'gelu'

    kb_model = KinyaBERT(args, num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 args.num_pos_m_embeddings,
                 args.num_stem_m_embeddings,
                 args.use_affix_bow_m_embedding,
                 args.use_pos_aware_rel_pos_bias,
                 args.use_tupe_rel_pos_bias,
                 max_seq_len = args.max_seq_len,
                 morpho_dim = args.morpho_dim, stem_dim = args.stem_dim,
                 morpho_tr_nhead = args.morpho_tr_nhead, morpho_tr_nlayers=args.morpho_tr_nlayers,
                 morpho_tr_dim_feedforward=args.morpho_tr_dim_feedforward, morpho_tr_dropout=args.morpho_tr_dropout, morpho_tr_activation=activation_fn,
                 seq_tr_nhead=args.seq_tr_nhead, seq_tr_nlayers=args.seq_tr_nlayers,
                 seq_tr_dim_feedforward=args.seq_tr_dim_feedforward, seq_tr_dropout=args.seq_tr_dropout, seq_tr_activation=activation_fn,
                 layernorm_epsilon = args.layernorm_epsilon).to(device)

    if saved_model_file is not None:
        kb_state_dict = torch.load(saved_model_file, map_location=device)
        kb_model.load_state_dict(kb_state_dict['model_state_dict'])

    return kb_model

def kinyabert_base_classifier(num_classes, kb_vocab : KBVocab, affix_set_vocab : AffixSetVocab, rel_pos_dict, device, args, saved_model_file = None, pooler_dropout=0.0) -> KinyaBERTClassifier:
    num_pos_tags = len(kb_vocab.pos_tag_vocab) + 1
    num_stems = len(kb_vocab.reduced_stem_vocab) + 1
    num_afsets = (len(affix_set_vocab.affix_set_vocab_idx) + 1) if args.use_afsets else 0
    num_affixes = len(kb_vocab.affix_vocab) + 1

    num_rel_pos_dict_size = (len(rel_pos_dict)+1) if (rel_pos_dict is not None) else 0

    activation_fn = 'gelu'

    kb_model = KinyaBERTClassifier(args, num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 num_classes,
                 args.num_pos_m_embeddings,
                 args.num_stem_m_embeddings,
                 args.use_affix_bow_m_embedding,
                 args.use_pos_aware_rel_pos_bias,
                 args.use_tupe_rel_pos_bias,
                 max_seq_len = args.max_seq_len,
                 morpho_dim = args.morpho_dim, stem_dim = args.stem_dim,
                 morpho_tr_nhead = args.morpho_tr_nhead, morpho_tr_nlayers=args.morpho_tr_nlayers,
                 morpho_tr_dim_feedforward=args.morpho_tr_dim_feedforward, morpho_tr_dropout=args.morpho_tr_dropout, morpho_tr_activation=activation_fn,
                 seq_tr_nhead=args.seq_tr_nhead, seq_tr_nlayers=args.seq_tr_nlayers,
                 seq_tr_dim_feedforward=args.seq_tr_dim_feedforward, seq_tr_dropout=args.seq_tr_dropout, seq_tr_activation=activation_fn,
                 pooler_dropout = pooler_dropout).to(device)

    if saved_model_file is not None:
        kb_state_dict = torch.load(saved_model_file, map_location=device)
        kb_model.load_state_dict(kb_state_dict['model_state_dict'])

    return kb_model

def kinyabert_base_classifier_from_pretrained(num_classes, kb_vocab : KBVocab, affix_set_vocab : AffixSetVocab, rel_pos_dict, device, args, pretrained_model_file, ddp = False, pooler_dropout=0.0)\
        -> KinyaBERTClassifier:
    from torch.nn.parallel import DistributedDataParallel as DDP
    pretrained_model = kinyabert_base(kb_vocab, affix_set_vocab, rel_pos_dict, device, args)
    classifier_model = kinyabert_base_classifier(num_classes, kb_vocab, affix_set_vocab, rel_pos_dict, device, args, pooler_dropout=pooler_dropout)

    kb_state_dict = torch.load(pretrained_model_file, map_location=device)
    if ddp:
        ddp_model = DDP(pretrained_model)
        ddp_model.load_state_dict(kb_state_dict['model_state_dict'])
        pretrained_model = ddp_model.module
    else:
        pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])

    classifier_model.encoder.load_state_dict(pretrained_model.encoder.state_dict())

    return classifier_model

def kinyabert_base_sequence_tagger(num_classes, kb_vocab : KBVocab, affix_set_vocab : AffixSetVocab, rel_pos_dict, device, args, saved_model_file = None, pooler_dropout=0.0)\
        -> KinyaBERTSequenceTagger:
    num_pos_tags = len(kb_vocab.pos_tag_vocab) + 1
    num_stems = len(kb_vocab.reduced_stem_vocab) + 1
    num_afsets = (len(affix_set_vocab.affix_set_vocab_idx) + 1) if args.use_afsets else 0
    num_affixes = len(kb_vocab.affix_vocab) + 1

    num_rel_pos_dict_size = (len(rel_pos_dict)+1) if (rel_pos_dict is not None) else 0

    activation_fn = 'gelu'

    kb_model = KinyaBERTSequenceTagger(args, num_stems, num_afsets, num_pos_tags, num_affixes,
                 num_rel_pos_dict_size,
                 num_classes,
                 args.num_pos_m_embeddings,
                 args.num_stem_m_embeddings,
                 args.use_affix_bow_m_embedding,
                 args.use_pos_aware_rel_pos_bias,
                 args.use_tupe_rel_pos_bias,
                 max_seq_len = args.max_seq_len,
                 morpho_dim = args.morpho_dim, stem_dim = args.stem_dim,
                 morpho_tr_nhead = args.morpho_tr_nhead, morpho_tr_nlayers=args.morpho_tr_nlayers,
                 morpho_tr_dim_feedforward=args.morpho_tr_dim_feedforward, morpho_tr_dropout=args.morpho_tr_dropout, morpho_tr_activation=activation_fn,
                 seq_tr_nhead=args.seq_tr_nhead, seq_tr_nlayers=args.seq_tr_nlayers,
                 seq_tr_dim_feedforward=args.seq_tr_dim_feedforward, seq_tr_dropout=args.seq_tr_dropout, pooler_dropout=pooler_dropout, seq_tr_activation=activation_fn).to(device)

    if saved_model_file is not None:
        kb_state_dict = torch.load(saved_model_file, map_location=device)
        kb_model.load_state_dict(kb_state_dict['model_state_dict'])

    return kb_model

def kinyabert_base_tagger_from_pretrained(num_classes, kb_vocab : KBVocab, affix_set_vocab : AffixSetVocab, rel_pos_dict, device, args, pretrained_model_file, ddp = False, pooler_dropout=0.0)\
        -> KinyaBERTSequenceTagger:
    from torch.nn.parallel import DistributedDataParallel as DDP
    pretrained_model = kinyabert_base(kb_vocab, affix_set_vocab, rel_pos_dict, device, args)
    sequence_tagger_model = kinyabert_base_sequence_tagger(num_classes, kb_vocab, affix_set_vocab, rel_pos_dict, device, args, pooler_dropout=pooler_dropout)

    kb_state_dict = torch.load(pretrained_model_file, map_location=device)
    if ddp:
        ddp_model = DDP(pretrained_model)
        ddp_model.load_state_dict(kb_state_dict['model_state_dict'])
        pretrained_model = ddp_model.module
    else:
        pretrained_model.load_state_dict(kb_state_dict['model_state_dict'])

    sequence_tagger_model.encoder.load_state_dict(pretrained_model.encoder.state_dict())

    return sequence_tagger_model

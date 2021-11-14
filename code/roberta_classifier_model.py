from __future__ import print_function, division
import torch

import torch.nn as nn

from torch.cuda.amp import custom_fwd

import torch

from fairseq.models.roberta import RobertaModel

from fairseq.data.encoders import register_bpe

from morpho_model import init_bert_params

@register_bpe("nonebpe")
class NoneBPE(object):

    @staticmethod
    def add_args(parser):
        pass

    def __init__(self, args):
        pass

    def encode(self, x: str) -> str:
        return x

    def decode(self, x: str) -> str:
        return x

class RobertaTokenClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.3):
        super(RobertaTokenClassificationHead, self).__init__()
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
        # features.shape = B x L x E
        # Remove [CLS], [SEP]
        # len already includes [CLS] and [SEP] in the sequence length count, so number of normal tokens here is (len-2)
        inputs = [features[i, 1:(len-1), :].contiguous().view(-1, self.input_dim) for i,len in enumerate(input_sequence_lengths)]
        x = torch.cat(inputs, 0) #  B' x E

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


class RobertaClassificationHead(nn.Module):
    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout=0.0):
        super(RobertaClassificationHead, self).__init__()
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
        # features.shape = B x L x E
        x = features[:, 0, :]  # Take [CLS]

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


class RobertaForTokenClassification(nn.Module):

    def __init__(self, input_dim, inner_dim, num_classes, args):
        super(RobertaForTokenClassification, self).__init__()

        if args.xlmr:
            from fairseq.models.roberta import XLMRModel
            self.roberta = XLMRModel.from_pretrained(args.pretrained_roberta_model_dir, checkpoint_file=args.pretrained_roberta_checkpoint_file, bpe="nonebpe")
        else:
            self.roberta = RobertaModel.from_pretrained(args.pretrained_roberta_model_dir, checkpoint_file=args.pretrained_roberta_checkpoint_file, bpe="nonebpe")

        self.roberta_model = self.roberta.model

        self.classification_head = RobertaTokenClassificationHead(input_dim, inner_dim, num_classes, pooler_dropout=args.pooler_dropout)

    @custom_fwd
    def forward(self, inputs_ids, input_sequence_lengths):
        '''
        Computes a forward pass through the sequence tagging model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            input_sequence_lengths: list of sequences lengths

        Returns :
            logits: unnormalized model outputs.
        '''

        x, _ = self.roberta_model(inputs_ids, features_only=True) # B x L --> B x L x E
        x = self.classification_head(x, input_sequence_lengths)
        return x

    def predict(self, inputs_ids):
        input_sequence_lengths = [inputs_ids.size(0)]
        inputs_ids = inputs_ids.unsqueeze(0)
        x, _ = self.roberta_model(inputs_ids, features_only=True) # B x L --> B x L x E
        x = self.classification_head(x, input_sequence_lengths) # --> B x num_classes
        predicted_labels = torch.argmax(x, dim=1)
        return x.squeeze(), predicted_labels

    def encode_sentence(self, s):
        """
        takes a string and returns a tensor of token ids
        """
        # Adds: <s> and </s> tokens
        sl = s.split(' ')
        if len(sl) > 510:
            s = ' '.join(sl[:510])
        inputs_ids = self.roberta.encode(s)
        tlen = inputs_ids.shape[0]
        tlen = min(tlen, 512)
        inputs_ids = inputs_ids[:tlen]
        # Fill <unk>
        mask_seq = self.roberta.encode('<mask>')
        unk_seq = self.roberta.encode('<unk>')
        inputs_ids.masked_fill_(inputs_ids.gt(mask_seq[1]), unk_seq[1])
        return inputs_ids

class RobertaForSentenceClassification(nn.Module):

    def __init__(self, input_dim, inner_dim, num_classes, args):
        super(RobertaForSentenceClassification, self).__init__()

        if args.xlmr:
            from fairseq.models.roberta import XLMRModel
            self.roberta = XLMRModel.from_pretrained(args.pretrained_roberta_model_dir, checkpoint_file=args.pretrained_roberta_checkpoint_file, bpe="nonebpe")
        else:
            self.roberta = RobertaModel.from_pretrained(args.pretrained_roberta_model_dir, checkpoint_file=args.pretrained_roberta_checkpoint_file, bpe="nonebpe")

        self.roberta_model = self.roberta.model

        self.classification_head = RobertaClassificationHead(input_dim, inner_dim, num_classes, pooler_dropout=args.pooler_dropout)

    @custom_fwd
    def forward(self, inputs_ids, input_sequence_lengths):
        '''
        Computes a forward pass through the sequence tagging model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            input_sequence_lengths: list of sequences lengths

        Returns :
            logits: unnormalized model outputs.
        '''

        x, _ = self.roberta_model(inputs_ids, features_only=True) # B x L --> B x L x E
        x = self.classification_head(x)
        return x

    def predict(self, inputs_ids):
        inputs_ids = inputs_ids.unsqueeze(0)
        x, _ = self.roberta_model(inputs_ids, features_only=True) # B x L --> B x L x E
        x = self.classification_head(x) # --> B x num_classes
        predicted_labels = torch.argmax(x, dim=1)
        return x.squeeze(), predicted_labels

    def encode_sentences(self, s0, s1):
        """
        takes a string and returns a tensor of token ids
        """
        # Adds: <s> and </s> tokens
        if s1 is not None:
            sl1 = s1.split(' ')
            if len(sl1) > 255:
                s1 = ' '.join(sl1[:255])
            sl0 = s0.split(' ')
            if len(sl0) > 255:
                s0 = ' '.join(sl0[:255])
        else:
            sl0 = s0.split(' ')
            if len(sl0) > 510:
                s0 = ' '.join(sl0[:510])
        inputs_ids = self.roberta.encode(s0) if (s1 is None) else self.roberta.encode(s0, s1)
        tlen = inputs_ids.shape[0]
        tlen = min(tlen, 512)
        inputs_ids = inputs_ids[:tlen]
        # Fill <unk>
        mask_seq = self.roberta.encode('<mask>')
        unk_seq = self.roberta.encode('<unk>')
        inputs_ids.masked_fill_(inputs_ids.gt(mask_seq[1]), unk_seq[1])
        return inputs_ids

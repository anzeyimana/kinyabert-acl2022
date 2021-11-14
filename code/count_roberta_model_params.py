
from __future__ import print_function, division

from fairseq.models.roberta import RobertaModel

from fairseq.data.encoders import register_bpe

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


if __name__ == '__main__':
    roberta = RobertaModel.from_pretrained("/mnt/NVM/KinyaBERT_Checkpoints/checkpoints-roberta-tupe-bpe-tpu/",
                                           checkpoint_file="checkpoint_94_31920.pt",
                                           bpe="nonebpe")
    total_params = sum(p.numel() for p in roberta.model.parameters())
    print('BERT-BPE', total_params)

    roberta = RobertaModel.from_pretrained("//mnt/NVM/KinyaBERT_Checkpoints/checkpoints-roberta-tupe-morpho-tpu/",
                                           checkpoint_file="checkpoint_44_31920.pt",
                                           bpe="nonebpe")
    total_params = sum(p.numel() for p in roberta.model.parameters())
    print('BERT-MORPHO', total_params)


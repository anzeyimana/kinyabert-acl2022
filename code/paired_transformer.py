r"""Functional interface"""
from typing import Callable, List, Optional, Tuple
import warnings

import torch

from torch import Tensor

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module, Linear, LayerNorm, Dropout, ModuleList
import copy
from typing import Optional, Any

import torch.nn.functional as F
from torch.nn.functional import linear, pad, softmax, dropout

def paired_multi_head_attention_forward(
    query_x: Tensor,
    key_x: Tensor,
    value_x: Tensor,
    query_y: Tensor,
    key_y: Tensor,
    value_y: Tensor,
    scale_factor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight_for_x: Tensor,
    in_proj_bias_for_x: Optional[Tensor],
    in_proj_weight_for_y: Tensor,
    in_proj_bias_for_y: Optional[Tensor],
    bias_k_x: Optional[Tensor],
    bias_v_x: Optional[Tensor],
    bias_k_y: Optional[Tensor],
    bias_v_y: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight_for_x: Tensor,
    out_proj_bias_for_x: Optional[Tensor],
    out_proj_weight_for_y: Tensor,
    out_proj_bias_for_y: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight_for_x: Optional[Tensor] = None,
    k_proj_weight_for_x: Optional[Tensor] = None,
    v_proj_weight_for_x: Optional[Tensor] = None,
    q_proj_weight_for_y: Optional[Tensor] = None,
    k_proj_weight_for_y: Optional[Tensor] = None,
    v_proj_weight_for_y: Optional[Tensor] = None,
    static_k_x: Optional[Tensor] = None,
    static_v_x: Optional[Tensor] = None,
    static_k_y: Optional[Tensor] = None,
    static_v_y: Optional[Tensor] = None,
    attn_bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tgt_len, bsz, embed_dim = query_x.size()
    tgt_len, bsz, embed_dim = query_y.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key_x.size(0) == value_x.size(0) and key_x.size(1) == value_x.size(1)
    assert key_y.size(0) == value_y.size(0) and key_y.size(1) == value_y.size(1)

    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim * scale_factor) ** -0.5

    if not use_separate_proj_weight:
        if (query_x is key_x or torch.equal(query_x, key_x)) and (key_x is value_x or torch.equal(key_x, value_x)) and (query_y is key_y or torch.equal(query_y, key_y)) and (key_y is value_y or torch.equal(key_y, value_y)):
            # self-attention
            q_x, k_x, v_x = linear(query_x, in_proj_weight_for_x, in_proj_bias_for_x).chunk(3, dim=-1)
            q_y, k_y, v_y = linear(query_y, in_proj_weight_for_y, in_proj_bias_for_y).chunk(3, dim=-1)

        elif (key_x is value_x or torch.equal(key_x, value_x)) and (key_y is value_y or torch.equal(key_y, value_y)):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _start = 0
            _end = embed_dim

            _b_x = in_proj_bias_for_x
            _w_x = in_proj_weight_for_x[_start:_end, :]
            if _b_x is not None:
                _b_x = _b_x[_start:_end]
            q_x = linear(query_x, _w_x, _b_x)

            _b_y = in_proj_bias_for_y
            _w_y = in_proj_weight_for_y[_start:_end, :]
            if _b_y is not None:
                _b_y = _b_y[_start:_end]
            q_y = linear(query_y, _w_y, _b_y)

            if (key_x is None) and (key_y is None):
                assert value_x is None
                assert value_y is None
                k_x = None
                v_x = None
                k_y = None
                v_y = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _start = embed_dim
                _end = None

                _b_x = in_proj_bias_for_x
                _w_x = in_proj_weight_for_x[_start:, :]
                if _b_x is not None:
                    _b_x = _b_x[_start:]
                k_x, v_x = linear(key_x, _w_x, _b_x).chunk(2, dim=-1)

                _b_y = in_proj_bias_for_y
                _w_y = in_proj_weight_for_y[_start:, :]
                if _b_y is not None:
                    _b_y = _b_y[_start:]
                k_y, v_y = linear(key_y, _w_y, _b_y).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _start = 0
            _end = embed_dim

            _b = in_proj_bias_for_x
            _w = in_proj_weight_for_x[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q_x = linear(query_x, _w, _b)

            _b = in_proj_bias_for_y
            _w = in_proj_weight_for_y[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q_y = linear(query_y, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2

            _b = in_proj_bias_for_x
            _w = in_proj_weight_for_x[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k_x = linear(key_x, _w, _b)

            _b = in_proj_bias_for_y
            _w = in_proj_weight_for_y[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k_y = linear(key_y, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _end = None
            _start = embed_dim * 2

            _b = in_proj_bias_for_x
            _w = in_proj_weight_for_x[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v_x = linear(value_x, _w, _b)

            _b = in_proj_bias_for_y
            _w = in_proj_weight_for_y[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v_y = linear(value_y, _w, _b)

    else:
        q_proj_weight_non_opt_for_x = torch.jit._unwrap_optional(q_proj_weight_for_x)
        len1, len2 = q_proj_weight_non_opt_for_x.size()
        assert len1 == embed_dim and len2 == query_x.size(-1)

        k_proj_weight_non_opt_for_x = torch.jit._unwrap_optional(k_proj_weight_for_x)
        len1, len2 = k_proj_weight_non_opt_for_x.size()
        assert len1 == embed_dim and len2 == key_x.size(-1)

        v_proj_weight_non_opt_for_x = torch.jit._unwrap_optional(v_proj_weight_for_x)
        len1, len2 = v_proj_weight_non_opt_for_x.size()
        assert len1 == embed_dim and len2 == value_x.size(-1)

        q_proj_weight_non_opt_for_y = torch.jit._unwrap_optional(q_proj_weight_for_y)
        len1, len2 = q_proj_weight_non_opt_for_y.size()
        assert len1 == embed_dim and len2 == query_y.size(-1)

        k_proj_weight_non_opt_for_y = torch.jit._unwrap_optional(k_proj_weight_for_y)
        len1, len2 = k_proj_weight_non_opt_for_y.size()
        assert len1 == embed_dim and len2 == key_y.size(-1)

        v_proj_weight_non_opt_for_y = torch.jit._unwrap_optional(v_proj_weight_for_y)
        len1, len2 = v_proj_weight_non_opt_for_y.size()
        assert len1 == embed_dim and len2 == value_y.size(-1)

        if in_proj_bias_for_x is not None:
            q_x = linear(query_x, q_proj_weight_non_opt_for_x, in_proj_bias_for_x[0:embed_dim])
            k_x = linear(key_x, k_proj_weight_non_opt_for_x, in_proj_bias_for_x[embed_dim : (embed_dim * 2)])
            v_x = linear(value_x, v_proj_weight_non_opt_for_x, in_proj_bias_for_x[(embed_dim * 2) :])
        else:
            q_x = linear(query_x, q_proj_weight_non_opt_for_x, in_proj_bias_for_x)
            k_x = linear(key_x, k_proj_weight_non_opt_for_x, in_proj_bias_for_x)
            v_x = linear(value_x, v_proj_weight_non_opt_for_x, in_proj_bias_for_x)

        if in_proj_bias_for_y is not None:
            q_y = linear(query_y, q_proj_weight_non_opt_for_y, in_proj_bias_for_y[0:embed_dim])
            k_y = linear(key_y, k_proj_weight_non_opt_for_y, in_proj_bias_for_y[embed_dim : (embed_dim * 2)])
            v_y = linear(value_y, v_proj_weight_non_opt_for_y, in_proj_bias_for_y[(embed_dim * 2) :])
        else:
            q_y = linear(query_y, q_proj_weight_non_opt_for_y, in_proj_bias_for_y)
            k_y = linear(key_y, k_proj_weight_non_opt_for_y, in_proj_bias_for_y)
            v_y = linear(value_y, v_proj_weight_non_opt_for_y, in_proj_bias_for_y)

    q_x = q_x * scaling
    q_y = q_y * scaling

    # Validation
    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.PairedMultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query_x.size(0), key_x.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query_x.size(0), key_x.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.PairedMultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k_x is not None and bias_v_x is not None:
        if static_k_x is None and static_v_x is None:
            k_x = torch.cat([k_x, bias_k_x.repeat(1, bsz, 1)])
            v_x = torch.cat([v_x, bias_v_x.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k_x is None, "bias cannot be added to static key."
            assert static_v_x is None, "bias cannot be added to static value."
    else:
        assert bias_k_x is None
        assert bias_v_x is None

    if bias_k_y is not None and bias_v_y is not None:
        if static_k_y is None and static_v_y is None:
            k_y = torch.cat([k_y, bias_k_y.repeat(1, bsz, 1)])
            v_y = torch.cat([v_y, bias_v_y.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k_y is None, "bias cannot be added to static key."
            assert static_v_y is None, "bias cannot be added to static value."
    else:
        assert bias_k_y is None
        assert bias_v_y is None

    q_x = q_x.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k_x is not None:
        k_x = k_x.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v_x is not None:
        v_x = v_x.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k_x is not None:
        assert static_k_x.size(0) == bsz * num_heads
        assert static_k_x.size(2) == head_dim
        k_x = static_k_x

    if static_v_x is not None:
        assert static_v_x.size(0) == bsz * num_heads
        assert static_v_x.size(2) == head_dim
        v_x = static_v_x

    q_y = q_y.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k_y is not None:
        k_y = k_y.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v_y is not None:
        v_y = v_y.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k_y is not None:
        assert static_k_y.size(0) == bsz * num_heads
        assert static_k_y.size(2) == head_dim
        k_y = static_k_y

    if static_v_y is not None:
        assert static_v_y.size(0) == bsz * num_heads
        assert static_v_y.size(2) == head_dim
        v_y = static_v_y

    src_len = k_x.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k_x = torch.cat([k_x, torch.zeros((k_x.size(0), 1) + k_x.size()[2:], dtype=k_x.dtype, device=k_x.device)], dim=1)
        v_x = torch.cat([v_x, torch.zeros((v_x.size(0), 1) + v_x.size()[2:], dtype=v_x.dtype, device=v_x.device)], dim=1)

        k_y = torch.cat([k_y, torch.zeros((k_y.size(0), 1) + k_y.size()[2:], dtype=k_y.dtype, device=k_y.device)], dim=1)
        v_y = torch.cat([v_y, torch.zeros((v_y.size(0), 1) + v_y.size()[2:], dtype=v_y.dtype, device=v_y.device)], dim=1)

        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q_x, k_x.transpose(1, 2))
    attn_output_weights += torch.bmm(q_x, k_y.transpose(1, 2))
    attn_output_weights += torch.bmm(q_y, k_x.transpose(1, 2))
    attn_output_weights += torch.bmm(q_y, k_y.transpose(1, 2))

    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_bias is not None:
        assert list(attn_bias.size()) == [bsz * num_heads, tgt_len, src_len]
        attn_output_weights += attn_bias

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output_for_x = torch.bmm(attn_output_weights, v_x)
    assert list(attn_output_for_x.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output_for_x = attn_output_for_x.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_for_x = linear(attn_output_for_x, out_proj_weight_for_x, out_proj_bias_for_x)

    attn_output_for_y = torch.bmm(attn_output_weights, v_y)
    assert list(attn_output_for_y.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output_for_y = attn_output_for_y.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_for_y = linear(attn_output_for_y, out_proj_weight_for_y, out_proj_bias_for_y)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output_for_x, attn_output_for_y, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output_for_x, attn_output_for_y, None


class PairedMultiheadAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> multihead_attn = nn.PairedMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, scale_factor=1, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(PairedMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.scale_factor=scale_factor

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight_for_x = Parameter(torch.empty(embed_dim, embed_dim))
            self.k_proj_weight_for_x = Parameter(torch.empty(embed_dim, self.kdim))
            self.v_proj_weight_for_x = Parameter(torch.empty(embed_dim, self.vdim))

            self.q_proj_weight_for_y = Parameter(torch.empty(embed_dim, embed_dim))
            self.k_proj_weight_for_y = Parameter(torch.empty(embed_dim, self.kdim))
            self.v_proj_weight_for_y = Parameter(torch.empty(embed_dim, self.vdim))

            self.register_parameter('in_proj_weight_for_x', None)
            self.register_parameter('in_proj_weight_for_y', None)
        else:
            self.in_proj_weight_for_x = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.in_proj_weight_for_y = Parameter(torch.empty(3 * embed_dim, embed_dim))

            self.register_parameter('q_proj_weight_for_x', None)
            self.register_parameter('k_proj_weight_for_x', None)
            self.register_parameter('v_proj_weight_for_x', None)

            self.register_parameter('q_proj_weight_for_y', None)
            self.register_parameter('k_proj_weight_for_y', None)
            self.register_parameter('v_proj_weight_for_y', None)

        if bias:
            self.in_proj_bias_for_x = Parameter(torch.empty(3 * embed_dim))
            self.in_proj_bias_for_y = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias_for_x', None)
            self.register_parameter('in_proj_bias_for_y', None)

        self.out_proj_for_x = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj_for_y = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k_for_x = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v_for_x = Parameter(torch.empty(1, 1, embed_dim))

            self.bias_k_for_y = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v_for_y = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k_for_x = self.bias_v_for_x = None
            self.bias_k_for_y = self.bias_v_for_y = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight_for_x)
            xavier_uniform_(self.in_proj_weight_for_y)
        else:
            xavier_uniform_(self.q_proj_weight_for_x)
            xavier_uniform_(self.k_proj_weight_for_x)
            xavier_uniform_(self.v_proj_weight_for_x)

            xavier_uniform_(self.q_proj_weight_for_y)
            xavier_uniform_(self.k_proj_weight_for_y)
            xavier_uniform_(self.v_proj_weight_for_y)

        if (self.in_proj_bias_for_x is not None) and (self.in_proj_bias_for_y is not None):
            constant_(self.in_proj_bias_for_x, 0.)
            constant_(self.out_proj_for_x.bias, 0.)

            constant_(self.in_proj_bias_for_y, 0.)
            constant_(self.out_proj_for_y.bias, 0.)

        if (self.bias_k_for_x is not None) and (self.bias_k_for_y is not None):
            xavier_normal_(self.bias_k_for_x)
            xavier_normal_(self.bias_k_for_y)
        if (self.bias_v_for_x is not None) and (self.bias_v_for_y is not None):
            xavier_normal_(self.bias_v_for_x)
            xavier_normal_(self.bias_v_for_y)

    def __setstate__(self, state):
        # Support loading old PairedMultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(PairedMultiheadAttention, self).__setstate__(state)

    def forward(self, query_x: Tensor, key_x: Tensor, value_x: Tensor,
                query_y: Tensor, key_y: Tensor, value_y: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return paired_multi_head_attention_forward(
                query_x, key_x, value_x,
                query_y, key_y, value_y,
                self.scale_factor, self.embed_dim, self.num_heads,
                self.in_proj_weight_for_x, self.in_proj_bias_for_x,
                self.in_proj_weight_for_y, self.in_proj_bias_for_y,
                self.bias_k_for_x, self.bias_v_for_x,
                self.bias_k_for_y, self.bias_v_for_y,
                self.add_zero_attn,
                self.dropout,
                self.out_proj_for_x.weight, self.out_proj_for_x.bias,
                self.out_proj_for_y.weight, self.out_proj_for_y.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, attn_bias=attn_bias,
                use_separate_proj_weight=True,
                q_proj_weight_for_x=self.q_proj_weight_for_x,
                k_proj_weight_for_x=self.k_proj_weight_for_x,
                v_proj_weight_for_x=self.v_proj_weight_for_x,
                q_proj_weight_for_y=self.q_proj_weight_for_y,
                k_proj_weight_for_y=self.k_proj_weight_for_y,
                v_proj_weight_for_y=self.v_proj_weight_for_y)
        else:
            return paired_multi_head_attention_forward(
                query_x, key_x, value_x,
                query_y, key_y, value_y,
                self.scale_factor, self.embed_dim, self.num_heads,
                self.in_proj_weight_for_x, self.in_proj_bias_for_x,
                self.in_proj_weight_for_y, self.in_proj_bias_for_y,
                self.bias_k_for_x, self.bias_v_for_x,
                self.bias_k_for_y, self.bias_v_for_y,
                self.add_zero_attn,
                self.dropout,
                self.out_proj_for_x.weight, self.out_proj_for_x.bias,
                self.out_proj_for_y.weight, self.out_proj_for_y.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, attn_bias=attn_bias)

class PairedTransformerEncoderLayer(Module):
    r"""PairedTransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = PairedTransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, attn_scale_factor=1, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(PairedTransformerEncoderLayer, self).__init__()
        self.self_attn = PairedMultiheadAttention(d_model, nhead, scale_factor=attn_scale_factor, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1_x = Linear(d_model, dim_feedforward)
        self.dropout_x = Dropout(dropout)
        self.linear2_x = Linear(dim_feedforward, d_model)

        self.norm1_x = LayerNorm(d_model)
        self.norm2_x = LayerNorm(d_model)
        self.dropout1_x = Dropout(dropout)
        self.dropout2_x = Dropout(dropout)

        self.activation_x = _get_activation_fn(activation)

        # Implementation of Feedforward model
        self.linear1_y = Linear(d_model, dim_feedforward)
        self.dropout_y = Dropout(dropout)
        self.linear2_y = Linear(dim_feedforward, d_model)

        self.norm1_y = LayerNorm(d_model)
        self.norm2_y = LayerNorm(d_model)
        self.dropout1_y = Dropout(dropout)
        self.dropout2_y = Dropout(dropout)

        self.activation_y = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation_x' not in state:
            state['activation_x'] = F.relu
        if 'activation_y' not in state:
            state['activation_y'] = F.relu
        super(PairedTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src_x: Tensor, src_y: Tensor, src_mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2_x, src2_y, _ = self.self_attn(src_x, src_x, src_x, src_y, src_y, src_y, attn_mask=src_mask, attn_bias=attn_bias,
                              key_padding_mask=src_key_padding_mask)

        src_x = src_x + self.dropout1_x(src2_x)
        src_x = self.norm1_x(src_x)
        src2_x = self.linear2_x(self.dropout_x(self.activation_x(self.linear1_x(src_x))))
        src_x = src_x + self.dropout2_x(src2_x)
        src_x = self.norm2_x(src_x)

        src_y = src_y + self.dropout1_y(src2_y)
        src_y = self.norm1_y(src_y)
        src2_y = self.linear2_y(self.dropout_y(self.activation_y(self.linear1_y(src_y))))
        src_y = src_y + self.dropout2_y(src2_y)
        src_y = self.norm2_y(src_y)

        return src_x, src_y

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PairedTransformerEncoder(Module):
    r"""PairedTransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the PairedTransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = PairedTransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = PairedTransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm_x=None, norm_y=None):
        super(PairedTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm_x = norm_x
        self.norm_y = norm_y

    def forward(self, src_x: Tensor, src_y: Tensor, mask: Optional[Tensor] = None, attn_bias: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output_x = src_x
        output_y = src_y

        for mod in self.layers:
            output_x, output_y = mod(output_x, output_y, src_mask=mask, attn_bias=attn_bias, src_key_padding_mask=src_key_padding_mask)

        if self.norm_x is not None:
            output_x = self.norm(output_x)

        if self.norm_y is not None:
            output_y = self.norm(output_y)

        return output_x, output_y

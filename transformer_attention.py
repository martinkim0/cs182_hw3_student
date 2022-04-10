from typing import Optional, Callable, Tuple

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from transformer_utils import ApplyAttentionMask

class AttentionQKV(nn.Module):
    """
    Computes attention based on provided similarity metric.
    """

    def __init__(self):
        super().__init__()
        self.apply_mask = ApplyAttentionMask()

    def forward(self, queries, keys, values, mask=None):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
            :param mask:    Tensor with shape [batch_size, n_queries, n_queries]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        # weights
        if len(queries.size()) > 3:
            # (N, h, q, d) (N, h, d, k) -> (N, h, q, k)
            sum_exp = "nhqd,nhdk->nhqk"
            keys_ = keys.permute(0, 1, 3, 2)
        else:
            # (N, q, d) (N, d, k) -> (N, q, k)
            sum_exp = "nqd,ndk->nqk"
            keys_ = keys.permute(0, 2, 1)
        sim = torch.einsum(sum_exp, queries, keys_)
        
        # mask and normalize
        masked_sim = self.apply_mask(sim, mask=mask)
        weights = F.softmax(masked_sim / math.sqrt(keys.shape[-1]), dim=-1)

        # attention output
        if len(masked_sim.size()) > 3:
            # (N, h, q, k) (N, h, k, v) -> (N, h, q, v)
            sum_exp = "nhqk,nhkv->nhqv"
        else:
            # (N, q, k) (N, k, v) -> (N, q, v)
            sum_exp = "nqk,nkv->nqv"
        output = torch.einsum(sum_exp, weights, values)

        return output, weights


class MultiHeadProjection(nn.Module):

    def __init__(self, n_heads, feature_sizes):
        """Map the multi-headed attention across the map

        Arguments:
            n_heads {int} -- The number of heads in the attention map
            feature_sizes {int} -- The size of the feature dimensions for key, query, and value

        """

        super().__init__()
        self.attention_map = AttentionQKV()
        self.n_heads = n_heads

        for size in feature_sizes:
            assert size % self.n_heads == 0, 'Shape of feature input must be divisible by n_heads'

    def forward(self, inputs, mask=None):
        """Fast multi-head attention.

        :param queries: Tensor with shape [batch_size, n_queries, depth_k]
        :param keys:    Tensor with shape [batch_size, n_keyval, depth_k]
        :param values:  Tensor with shape [batch_size, n_keyval, depth_v]

        :return: output: Tensor with shape [batch_size, n_queries, depth_v]
        """
        queries, keys, values = inputs

        # Split each of the projection into its heads, by adding a new dimension
        # You must implement _split_heads, and _combine_heads
        queries_split = self._split_heads(queries)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        # Apply the attention map
        attention_output_split, _ = self.attention_map(queries_split, keys_split, values_split, mask=mask)

        # Re-combine the heads together, and return the output.
        output = self._combine_heads(attention_output_split)
        return output

    def _split_heads(self, tensor):
        assert len(tensor.shape) == 3

        n, l, d = tensor.size()
        # (N, l, d) -> (N, l, h, d/h)
        tensor = torch.reshape(tensor, (n, l, self.n_heads, int(d/self.n_heads)))
        # (N, l, h, d/h) -> (N, h, l, d/h)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor
        ##########################################################################################

    def _combine_heads(self, tensor):
        assert len(tensor.shape) == 4

        n, h, l, d = tensor.size()
        # (N, h, l, d) -> (N, l, h, d)
        tensor = tensor.permute(0, 2, 1, 3)
        # (N, l, h, d) -> (N, l, d*h)
        tensor = torch.reshape(tensor, (n, l, d*h))
        return tensor
        ##########################################################################################

class MultiHeadAttention(nn.Module):
    """
    Fast multi-head attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, n_heads, input_shapes):
        super().__init__()

        self.qa_channels, self.ma_channels = input_shapes

        self.n_heads = n_heads
        self.attention_layer = MultiHeadProjection(n_heads, (self.qa_channels,self.ma_channels))

        assert self.qa_channels % self.n_heads == 0 and self.ma_channels % self.n_heads == 0 and \
                                                        'Feature size must be divisible by n_heads'
        assert self.qa_channels == self.ma_channels and 'Cannot combine tensors with different shapes'

        self.query_layer = weight_norm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))
        self.key_layer = weight_norm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))
        self.value_layer = weight_norm(nn.Linear(self.ma_channels, self.ma_channels, bias=False))

        self.output_layer = weight_norm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))

        def weights_init(m):
            # if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
        self.query_layer.apply(weights_init)
        self.key_layer.apply(weights_init)
        self.value_layer.apply(weights_init)
        self.output_layer.apply(weights_init)


    def forward(self, inputs, mask=None):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        assert (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 2 and \
                                                        'Must pass query and memory'
        query_antecedent, memory_antecedent = inputs
        q = self.query_layer(query_antecedent)
        k = self.key_layer(memory_antecedent)
        v = self.value_layer(memory_antecedent)

        attention_output = self.attention_layer((q, k, v), mask=mask)
        output = self.output_layer(attention_output)
        return output

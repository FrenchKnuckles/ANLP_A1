import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional import apply_rope


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, v)
    return output, attn_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, rope_cos=None, rope_sin=None):
        batch_size = query.size(0)

        q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        attn_output, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attn_output)


class GroupedQueryAttention(nn.Module):

    def __init__(self, d_model, num_query_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_query_heads == 0
        assert num_query_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.groups = num_query_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, num_query_heads * self.head_dim)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, rope_cos=None, rope_sin=None):
        batch_size = query.size(0)

        q = self.W_q(query).view(batch_size, -1, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(key).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(value).view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            q, k = apply_rope(q, k, rope_cos, rope_sin)

        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)

        attn_output, _ = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(attn_output)

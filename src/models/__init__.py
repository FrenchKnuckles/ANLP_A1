import math
import torch
import torch.nn as nn
from .attention import MultiHeadAttention, GroupedQueryAttention
from .norm import LayerNorm, RMSNorm
from .positional import SinusoidalPositionalEncoding, build_rope_cache


def _get_norm(norm_type, d_model):
    if norm_type == 'layernorm':
        return LayerNorm(d_model)
    elif norm_type == 'rmsnorm':
        return RMSNorm(d_model)
    else:
        raise ValueError(f'Unknown norm_type: {norm_type}')


def _get_attention(attention_type, d_model, num_heads, num_kv_heads, dropout):
    if attention_type == 'mha':
        return MultiHeadAttention(d_model, num_heads, dropout)
    elif attention_type == 'gqa':
        return GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
    else:
        raise ValueError(f'Unknown attention_type: {attention_type}')


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, attention_type='mha', norm_type='layernorm', num_kv_heads=2):
        super().__init__()
        self.self_attn = _get_attention(attention_type, d_model, num_heads, num_kv_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = _get_norm(norm_type, d_model)
        self.norm2 = _get_norm(norm_type, d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, src_mask=None, rope_cos=None, rope_sin=None):
        x_norm = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x_norm, x_norm, x_norm, mask=src_mask, rope_cos=rope_cos, rope_sin=rope_sin))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, attention_type='mha', norm_type='layernorm', num_kv_heads=2):
        super().__init__()
        self.self_attn = _get_attention(attention_type, d_model, num_heads, num_kv_heads, dropout)
        self.cross_attn = _get_attention(attention_type, d_model, num_heads, num_kv_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = _get_norm(norm_type, d_model)
        self.norm2 = _get_norm(norm_type, d_model)
        self.norm3 = _get_norm(norm_type, d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None, rope_cos=None, rope_sin=None):
        x_norm = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x_norm, x_norm, x_norm, mask=tgt_mask, rope_cos=rope_cos, rope_sin=rope_sin))

        x_norm = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x_norm, enc_output, enc_output, mask=memory_mask))

        x = x + self.dropout3(self.ffn(self.norm3(x)))
        return x


class Seq2SeqTransformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8,
                 num_encoder_layers=4, num_decoder_layers=4, d_ff=1024, dropout=0.1,
                 max_seq_len=512, pad_idx=0, attention_type='mha', norm_type='layernorm',
                 positional_encoding='sinusoidal', num_kv_heads=2):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.positional_encoding_type = positional_encoding
        self.max_seq_len = max_seq_len

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        if positional_encoding == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            self.rope_cos = None
            self.rope_sin = None
        elif positional_encoding == 'rope':
            self.pos_encoder = None
            head_dim = d_model // num_heads
            cos_cache, sin_cache = build_rope_cache(max_seq_len, head_dim)
            self.register_buffer('rope_cos', cos_cache)
            self.register_buffer('rope_sin', sin_cache)
        else:
            raise ValueError(f'Unknown positional_encoding: {positional_encoding}')

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, attention_type, norm_type, num_kv_heads)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, attention_type, norm_type, num_kv_heads)
            for _ in range(num_decoder_layers)
        ])

        self.encoder_norm = _get_norm(norm_type, d_model)
        self.decoder_norm = _get_norm(norm_type, d_model)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        self.output_projection.weight = self.tgt_embedding.weight
        self.embed_dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_src_mask(self, src):
        src_pad_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_pad_mask.float() * -1e9

    def _make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.shape
        tgt_pad_mask = (tgt == self.pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        combined = (tgt_pad_mask | causal_mask).float() * -1e9
        return combined

    def _make_memory_mask(self, src):
        return self._make_src_mask(src)

    def encode(self, src, src_mask=None):
        x = self.src_embedding(src) * math.sqrt(self.d_model)

        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        else:
            x = self.embed_dropout(x)

        rope_cos = self.rope_cos if self.positional_encoding_type == 'rope' else None
        rope_sin = self.rope_sin if self.positional_encoding_type == 'rope' else None

        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask, rope_cos=rope_cos, rope_sin=rope_sin)

        return self.encoder_norm(x)

    def decode(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        else:
            x = self.embed_dropout(x)

        rope_cos = self.rope_cos if self.positional_encoding_type == 'rope' else None
        rope_sin = self.rope_sin if self.positional_encoding_type == 'rope' else None

        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask, rope_cos=rope_cos, rope_sin=rope_sin)

        return self.decoder_norm(x)

    def forward(self, src, tgt):
        src_mask = self._make_src_mask(src)
        tgt_mask = self._make_tgt_mask(tgt)
        memory_mask = self._make_memory_mask(src)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, memory_mask)
        return self.output_projection(dec_output)

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx, eos_idx, max_len=512):
        batch_size = src.size(0)
        device = src.device

        src_mask = self._make_src_mask(src)
        enc_output = self.encode(src, src_mask)
        memory_mask = self._make_memory_mask(src)

        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask = self._make_tgt_mask(ys)
            dec_output = self.decode(ys, enc_output, tgt_mask, memory_mask)
            logits = self.output_projection(dec_output[:, -1, :])

            next_token = logits.argmax(dim=-1)
            next_token = next_token.masked_fill(finished, eos_idx)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)

            if finished.all():
                break

        return ys[:, 1:]

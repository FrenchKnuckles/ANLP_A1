"""
Transformer assembly — FeedForward, EncoderLayer, DecoderLayer, Seq2SeqTransformer.

All components use a Pre-LN residual pattern: x = x + Sublayer(Norm(x)),
plus a final norm after the last block before the output projection.

Seq2SeqTransformer serves C1–C4 via constructor flags:
    attention_type: 'mha' or 'gqa'
    norm_type: 'layernorm' or 'rmsnorm'
    positional_encoding: 'sinusoidal' or 'rope'
    num_kv_heads: used only when attention_type='gqa'
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GroupedQueryAttention
from .norm import LayerNorm, RMSNorm
from .positional import SinusoidalPositionalEncoding, build_rope_cache


def _get_norm(norm_type: str, d_model: int) -> nn.Module:
    """Factory for normalization layer."""
    if norm_type == "layernorm":
        return LayerNorm(d_model)
    elif norm_type == "rmsnorm":
        return RMSNorm(d_model)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def _get_attention(
    attention_type: str, d_model: int, num_heads: int, num_kv_heads: int, dropout: float
) -> nn.Module:
    """Factory for attention layer."""
    if attention_type == "mha":
        return MultiHeadAttention(d_model, num_heads, dropout)
    elif attention_type == "gqa":
        return GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout)
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network.

    Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer with Pre-LN residual connections.

    Architecture: x = x + SelfAttn(Norm(x))
                  x = x + FFN(Norm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: str = "mha",
        norm_type: str = "layernorm",
        num_kv_heads: int = 2,
    ):
        super().__init__()
        self.self_attn = _get_attention(attention_type, d_model, num_heads, num_kv_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = _get_norm(norm_type, d_model)
        self.norm2 = _get_norm(norm_type, d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            src_mask: additive mask for padding, broadcastable to (batch, heads, seq, seq)
            rope_cos, rope_sin: precomputed RoPE tables.
        """
        # Pre-LN self-attention
        x_norm = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x_norm, x_norm, x_norm, mask=src_mask,
                                              rope_cos=rope_cos, rope_sin=rope_sin))
        # Pre-LN FFN
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer with Pre-LN residual connections.

    Architecture: x = x + CausalSelfAttn(Norm(x))
                  x = x + CrossAttn(Norm(x), encoder_output)
                  x = x + FFN(Norm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: str = "mha",
        norm_type: str = "layernorm",
        num_kv_heads: int = 2,
    ):
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

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_seq_len, d_model)
            enc_output: (batch, src_seq_len, d_model)
            tgt_mask: causal + padding mask for decoder self-attention.
            memory_mask: padding mask for cross-attention (src padding).
            rope_cos, rope_sin: precomputed RoPE tables.
        """
        # Pre-LN causal self-attention
        x_norm = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x_norm, x_norm, x_norm, mask=tgt_mask,
                                              rope_cos=rope_cos, rope_sin=rope_sin))
        # Pre-LN cross-attention (no RoPE — cross-attention has no meaningful positional relationship)
        x_norm = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x_norm, enc_output, enc_output, mask=memory_mask))

        # Pre-LN FFN
        x = x + self.dropout3(self.ffn(self.norm3(x)))
        return x


class Seq2SeqTransformer(nn.Module):
    """Full Encoder-Decoder Transformer for C1–C4 configs.

    Embedding (scaled by sqrt(d_model)) + Positional Encoding + N EncoderLayers
    + N DecoderLayers + final norm + output projection.

    Constructor flags control the ablation axis:
        positional_encoding: 'sinusoidal' (C1/C3/C4) or 'rope' (C2)
        attention_type: 'mha' (C1/C2/C4) or 'gqa' (C3)
        norm_type: 'layernorm' (C1/C2/C3) or 'rmsnorm' (C4)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_idx: int = 0,
        attention_type: str = "mha",
        norm_type: str = "layernorm",
        positional_encoding: str = "sinusoidal",
        num_kv_heads: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.positional_encoding_type = positional_encoding
        self.max_seq_len = max_seq_len

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # Positional encoding (mutually exclusive)
        if positional_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            self.rope_cos = None
            self.rope_sin = None
        elif positional_encoding == "rope":
            self.pos_encoder = None
            head_dim = d_model // num_heads
            cos_cache, sin_cache = build_rope_cache(max_seq_len, head_dim)
            self.register_buffer("rope_cos", cos_cache)
            self.register_buffer("rope_sin", sin_cache)
        else:
            raise ValueError(f"Unknown positional_encoding: {positional_encoding}")

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, attention_type, norm_type, num_kv_heads)
            for _ in range(num_encoder_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, attention_type, norm_type, num_kv_heads)
            for _ in range(num_decoder_layers)
        ])

        # Final norms (after last block, before output projection)
        self.encoder_norm = _get_norm(norm_type, d_model)
        self.decoder_norm = _get_norm(norm_type, d_model)

        # Output projection to target vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Weight tying
        self.output_projection.weight = self.tgt_embedding.weight

        # Dropout for RoPE path (sinusoidal has its own dropout)
        self.embed_dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for embeddings and linear layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask for source sequence.

        Returns additive mask: 0 for valid positions, -inf for padding.
        Shape: (batch, 1, 1, src_len) — broadcastable over heads and query positions.
        """
        # src: (batch, src_len)
        src_pad_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
        return src_pad_mask.float() * -1e9

    def _make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create combined causal + padding mask for target sequence.

        Returns additive mask: 0 for valid positions, -inf for masked.
        Shape: (batch, 1, tgt_len, tgt_len).
        """
        batch_size, tgt_len = tgt.shape

        # Padding mask: (batch, 1, 1, tgt_len)
        tgt_pad_mask = (tgt == self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Causal mask: upper triangle = True (future positions masked)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, tgt_len)

        # Combine: mask if either is masked
        combined = (tgt_pad_mask | causal_mask).float() * -1e9
        return combined

    def _make_memory_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask for cross-attention (decoder attending to encoder output).

        Same as src_mask: mask padding in encoder output.
        Shape: (batch, 1, 1, src_len).
        """
        return self._make_src_mask(src)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run encoder.

        Args:
            src: (batch, src_len) — token IDs.
            src_mask: optional precomputed mask.

        Returns:
            (batch, src_len, d_model) — encoder output.
        """
        # Embed and scale
        x = self.src_embedding(src) * math.sqrt(self.d_model)

        # Positional encoding
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        else:
            x = self.embed_dropout(x)

        # RoPE cos/sin for attention layers
        rope_cos = self.rope_cos if self.positional_encoding_type == "rope" else None
        rope_sin = self.rope_sin if self.positional_encoding_type == "rope" else None

        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask, rope_cos=rope_cos, rope_sin=rope_sin)

        return self.encoder_norm(x)

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run decoder.

        Args:
            tgt: (batch, tgt_len) — target token IDs.
            enc_output: (batch, src_len, d_model).
            tgt_mask: causal + padding mask.
            memory_mask: src padding mask for cross-attention.

        Returns:
            (batch, tgt_len, d_model)
        """
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        else:
            x = self.embed_dropout(x)

        rope_cos = self.rope_cos if self.positional_encoding_type == "rope" else None
        rope_sin = self.rope_sin if self.positional_encoding_type == "rope" else None

        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask,
                      rope_cos=rope_cos, rope_sin=rope_sin)

        return self.decoder_norm(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass for teacher-forced training.

        Args:
            src: (batch, src_len) — source token IDs.
            tgt: (batch, tgt_len) — target token IDs (shifted right, with BOS).

        Returns:
            (batch, tgt_len, tgt_vocab_size) — logits.
        """
        src_mask = self._make_src_mask(src)
        tgt_mask = self._make_tgt_mask(tgt)
        memory_mask = self._make_memory_mask(src)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, tgt_mask, memory_mask)

        return self.output_projection(dec_output)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 512,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding for evaluation.

        Args:
            src: (batch, src_len) — source token IDs.
            bos_idx: beginning-of-sequence token ID.
            eos_idx: end-of-sequence token ID.
            max_len: maximum target length.

        Returns:
            (batch, decoded_len) — predicted token IDs (without BOS, with EOS if generated).
        """
        batch_size = src.size(0)
        device = src.device

        src_mask = self._make_src_mask(src)
        enc_output = self.encode(src, src_mask)
        memory_mask = self._make_memory_mask(src)

        # Start with BOS token
        ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask = self._make_tgt_mask(ys)
            dec_output = self.decode(ys, enc_output, tgt_mask, memory_mask)
            logits = self.output_projection(dec_output[:, -1, :])  # (batch, vocab)
            next_token = logits.argmax(dim=-1)  # (batch,)

            # Don't update finished sequences
            next_token = next_token.masked_fill(finished, eos_idx)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            # Update finished status
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break

        return ys[:, 1:]  # Remove BOS

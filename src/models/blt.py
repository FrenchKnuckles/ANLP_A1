"""
Simplified Byte Latent Transformer (BLT) — Token-Free Seq2Seq (Config C5).

Reduced from the full BLT (Meta AI, Dec 2024):
- Fixed-size patches instead of entropy-based dynamic boundaries.
- LocalEncoder: byte embedding -> small self-attention -> cross-attention pooling per patch.
- Global transformer: reuses EncoderLayer/DecoderLayer from models/__init__.py, configured
  like C1 (sinusoidal absolute PE, MHA, LayerNorm).
- LocalDecoder: per-patch autoregressive byte decoder conditioned on global decoder output.

Design note: cross-attention pooling is used (one learned query per patch attending over
that patch's byte representations), which is closer to the real BLT architecture.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .norm import LayerNorm
from .positional import SinusoidalPositionalEncoding


# ── Special token indices for byte-level vocab ──────────────────────────
# Byte values: 0–255. Sentinel values past byte range:
BYTE_PAD = 256
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_VOCAB_SIZE = 259  # 256 bytes + 3 sentinels


class LocalEncoder(nn.Module):
    """Byte-level encoder that produces one patch embedding per fixed-size window.

    Pipeline per patch:
        1. Byte embedding + byte-level sinusoidal positional encoding (within each patch).
        2. 1-layer self-attention over the bytes in the patch.
        3. Cross-attention pooling: one learned query per patch attends over the
           patch's byte representations to produce a single patch embedding.

    Args:
        d_model: global model dimension (patch embedding dim).
        patch_size: number of bytes per patch.
        d_local: local encoder hidden dimension (may differ from d_model).
        num_heads: attention heads in the local self-attention.
        dropout: dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        patch_size: int = 4,
        d_local: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_local = d_local

        # Byte embedding (vocab = 256 byte values + PAD/BOS/EOS)
        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, d_local, padding_idx=BYTE_PAD)

        # Byte-level positional encoding (within each patch — separate from patch-level PE)
        # Using context=2, max_len is patch_size + 2 * context + 2
        self.byte_pos_enc = SinusoidalPositionalEncoding(d_local, max_len=patch_size + 6, dropout=dropout)

        # 2-layer self-attention over bytes within a patch
        self.self_attn1 = MultiHeadAttention(d_local, num_heads, dropout)
        self.self_attn_norm1 = LayerNorm(d_local)
        self.self_attn_dropout1 = nn.Dropout(dropout)

        self.self_attn2 = MultiHeadAttention(d_local, num_heads, dropout)
        self.self_attn_norm2 = LayerNorm(d_local)
        self.self_attn_dropout2 = nn.Dropout(dropout)

        # Cross-attention pooling: learned query for each patch
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_local))
        self.pool_attn = MultiHeadAttention(d_local, num_heads, dropout)
        self.pool_norm = LayerNorm(d_local)

        # Project from local dim to global d_model
        self.project = nn.Linear(d_local, d_model)

    def forward(self, byte_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            byte_ids: (batch, seq_len) — byte-level token IDs (0–258).

        Returns:
            patch_embeddings: (batch, num_patches, d_model)
            patch_pad_mask: (batch, num_patches) — True where patch is all-padding.
        """
        batch_size, seq_len = byte_ids.shape

        # Pad sequence to be divisible by patch_size
        remainder = seq_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            byte_ids = torch.nn.functional.pad(byte_ids, (0, pad_len), value=BYTE_PAD)
            seq_len = byte_ids.size(1)

        num_patches = seq_len // self.patch_size

        # --- Context sliding window ---
        context = 2
        window_size = self.patch_size + 2 * context
        
        # Pad with context on both sides
        padded_ids = torch.nn.functional.pad(byte_ids, (context, context), value=BYTE_PAD)
        
        # Unfold to get windows: (batch, num_patches, window_size)
        patches = padded_ids.unfold(1, window_size, self.patch_size)

        # Track which patches are all-padding in their core region
        core_patches = byte_ids.view(batch_size, num_patches, self.patch_size)
        patch_pad_mask = (core_patches == BYTE_PAD).all(dim=-1)

        # Flatten for embedding: (batch * num_patches, window_size)
        patches_flat = patches.contiguous().view(batch_size * num_patches, window_size)

        # Additive padding mask
        pad_mask = (patches_flat == BYTE_PAD) # (B*P, window_size)
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2).float() * -1e9 # (B*P, 1, 1, window_size)

        # Embed bytes + positional encoding
        x = self.byte_embedding(patches_flat)  # (B*P, window_size, d_local)
        x = self.byte_pos_enc(x)

        # 2-layer Self-attention (Pre-LN)
        x_norm = self.self_attn_norm1(x)
        x = x + self.self_attn_dropout1(self.self_attn1(x_norm, x_norm, x_norm, mask=attn_mask))

        x_norm = self.self_attn_norm2(x)
        x = x + self.self_attn_dropout2(self.self_attn2(x_norm, x_norm, x_norm, mask=attn_mask))

        # Cross-attention pooling: one query per patch -> one embedding per patch
        query = self.pool_query.expand(batch_size * num_patches, -1, -1)  # (B*P, 1, d_local)
        x_norm = self.pool_norm(x)
        pooled = self.pool_attn(query, x_norm, x_norm, mask=attn_mask)  # (B*P, 1, d_local)
        pooled = pooled.squeeze(1)  # (B*P, d_local)

        # Project to global dimension and reshape
        patch_emb = self.project(pooled)  # (B*P, d_model)
        patch_emb = patch_emb.view(batch_size, num_patches, self.d_model)

        return patch_emb, patch_pad_mask


class LocalDecoder(nn.Module):
    """Per-patch autoregressive byte decoder.

    For each patch, autoregressively predicts the patch's bytes:
    - Causal self-attention among already-generated bytes within the current patch.
    - Cross-attention to the corresponding global-decoder patch output.

    Args:
        d_model: global model dimension.
        patch_size: bytes per patch.
        d_local: local decoder hidden dimension.
        num_heads: attention heads.
        dropout: dropout rate.
    """

    def __init__(
        self,
        d_model: int = 256,
        patch_size: int = 4,
        d_local: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_local = d_local

        # Byte embedding for target bytes
        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, d_local, padding_idx=BYTE_PAD)
        self.byte_pos_enc = SinusoidalPositionalEncoding(d_local, max_len=patch_size + 2, dropout=dropout)

        # Project global decoder output to local dimension
        self.project_in = nn.Linear(d_model, d_local)

        # 2-layer Causal self-attention within patch
        self.self_attn1 = MultiHeadAttention(d_local, num_heads, dropout)
        self.self_attn_norm1 = LayerNorm(d_local)
        self.self_attn_dropout1 = nn.Dropout(dropout)
        
        self.self_attn2 = MultiHeadAttention(d_local, num_heads, dropout)
        self.self_attn_norm2 = LayerNorm(d_local)
        self.self_attn_dropout2 = nn.Dropout(dropout)

        # 2-layer Cross-attention to global decoder output
        self.cross_attn1 = MultiHeadAttention(d_local, num_heads, dropout)
        self.cross_attn_norm1 = LayerNorm(d_local)
        self.cross_attn_dropout1 = nn.Dropout(dropout)
        
        self.cross_attn2 = MultiHeadAttention(d_local, num_heads, dropout)
        self.cross_attn_norm2 = LayerNorm(d_local)
        self.cross_attn_dropout2 = nn.Dropout(dropout)

        # Output projection to byte vocabulary
        self.output_norm = LayerNorm(d_local)
        self.output_proj = nn.Linear(d_local, BYTE_VOCAB_SIZE)
        
        # Weight tying
        self.output_proj.weight = self.byte_embedding.weight

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Causal mask for within-patch autoregressive decoding."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.float() * -1e9

    def step(self, x: torch.Tensor, global_proj: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Run the 2-layer local decoder step."""
        # Layer 1
        x_norm = self.self_attn_norm1(x)
        x = x + self.self_attn_dropout1(self.self_attn1(x_norm, x_norm, x_norm, mask=causal_mask))
        x_norm = self.cross_attn_norm1(x)
        x = x + self.cross_attn_dropout1(self.cross_attn1(x_norm, global_proj, global_proj))

        # Layer 2
        x_norm = self.self_attn_norm2(x)
        x = x + self.self_attn_dropout2(self.self_attn2(x_norm, x_norm, x_norm, mask=causal_mask))
        x_norm = self.cross_attn_norm2(x)
        x = x + self.cross_attn_dropout2(self.cross_attn2(x_norm, global_proj, global_proj))
        return x

    def forward(
        self,
        tgt_bytes: torch.Tensor,
        global_dec_output: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            tgt_bytes: (batch, tgt_byte_len) — target byte IDs.
            global_dec_output: (batch, num_tgt_patches, d_model) — from global decoder.

        Returns:
            logits: (batch, tgt_byte_len, BYTE_VOCAB_SIZE)
        """
        batch_size, tgt_byte_len = tgt_bytes.shape
        num_patches = global_dec_output.size(1)

        # Pad target to be divisible by patch_size
        remainder = tgt_byte_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            tgt_bytes = torch.nn.functional.pad(tgt_bytes, (0, pad_len), value=BYTE_PAD)
            tgt_byte_len = tgt_bytes.size(1)

        # Ensure we have the right number of patches
        actual_patches = tgt_byte_len // self.patch_size
        if actual_patches > num_patches:
            # Truncate target to match global decoder output
            tgt_bytes = tgt_bytes[:, : num_patches * self.patch_size]
            tgt_byte_len = tgt_bytes.size(1)
            actual_patches = num_patches
        elif actual_patches < num_patches:
            # Pad global output (shouldn't normally happen)
            global_dec_output = global_dec_output[:, :actual_patches, :]

        # Reshape target into patches: (batch * num_patches, patch_size)
        tgt_patches = tgt_bytes.view(batch_size, actual_patches, self.patch_size)
        tgt_flat = tgt_patches.reshape(batch_size * actual_patches, self.patch_size)

        # Embed target bytes
        x = self.byte_embedding(tgt_flat)  # (B*P, patch_size, d_local)
        x = self.byte_pos_enc(x)

        causal_mask = self._make_causal_mask(self.patch_size, x.device)
        global_proj = self.project_in(global_dec_output)
        global_flat = global_proj.reshape(batch_size * actual_patches, 1, self.d_local)

        x = self.step(x, global_flat, causal_mask)

        # Output projection
        logits = self.output_proj(self.output_norm(x))  # (B*P, patch_size, vocab)
        logits = logits.view(batch_size, actual_patches * self.patch_size, BYTE_VOCAB_SIZE)

        # Trim back to original target length if we padded
        return logits[:, :tgt_byte_len, :]


class BLTSeq2SeqModel(nn.Module):
    """Byte Latent Transformer: LocalEncoder -> Global Enc/Dec -> LocalDecoder.

    The global transformer uses the same architecture as C1:
    sinusoidal absolute PE at the patch level, MHA, LayerNorm.
    Only the tokenization changes in C5 — the backbone must match the base config.

    Args:
        d_model: global model dimension.
        num_heads: global attention heads.
        num_encoder_layers: global encoder depth.
        num_decoder_layers: global decoder depth.
        d_ff: FFN intermediate dimension.
        dropout: dropout rate.
        max_seq_len: max patch sequence length.
        patch_size: bytes per patch.
        d_local: local encoder/decoder hidden dim.
        local_heads: local attention heads.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        patch_size: int = 4,
        d_local: int = 128,
        local_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        # Initial patch representation to condition the first patch's local decoder
        self.init_patch_repr = nn.Parameter(torch.randn(1, 1, d_model))

        # Local encoder: bytes -> patches
        self.local_encoder = LocalEncoder(d_model, patch_size, d_local, local_heads, dropout)

        # Global patch-level positional encoding (sinusoidal — matches C1)
        self.patch_pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Global encoder layers (same config as C1: MHA + LayerNorm)
        from . import EncoderLayer, DecoderLayer
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, "mha", "layernorm", num_heads)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, "mha", "layernorm", num_heads)
            for _ in range(num_decoder_layers)
        ])

        self.encoder_norm = LayerNorm(d_model)
        self.decoder_norm = LayerNorm(d_model)

        # Local decoder: patches -> bytes
        self.local_decoder = LocalDecoder(d_model, patch_size, d_local, local_heads, dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_patch_pad_mask(self, patch_pad_mask: torch.Tensor) -> torch.Tensor:
        """Convert boolean patch padding mask to additive attention mask.

        Args:
            patch_pad_mask: (batch, num_patches) — True where padded.
        Returns:
            (batch, 1, 1, num_patches) additive mask.
        """
        return patch_pad_mask.unsqueeze(1).unsqueeze(2).float() * -1e9

    def _make_causal_patch_mask(self, num_patches: int, device: torch.device) -> torch.Tensor:
        """Causal mask for patch-level decoder self-attention."""
        mask = torch.triu(torch.ones(num_patches, num_patches, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0).float() * -1e9

    def encode(
        self,
        src_bytes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode source bytes to patch-level representations.

        Args:
            src_bytes: (batch, src_byte_len) — source byte IDs.

        Returns:
            enc_output: (batch, num_src_patches, d_model)
            src_patch_pad_mask: (batch, num_src_patches) — True where padded.
        """
        # Local encoder: bytes -> patches
        patch_emb, patch_pad_mask = self.local_encoder(src_bytes)

        # Add patch-level positional encoding
        patch_emb = self.patch_pos_enc(patch_emb)

        # Make attention mask from patch padding
        src_mask = self._make_patch_pad_mask(patch_pad_mask)

        # Global encoder
        x = patch_emb
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)

        return self.encoder_norm(x), patch_pad_mask

    def decode_patches(
        self,
        tgt_bytes: torch.Tensor,
        enc_output: torch.Tensor,
        src_patch_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run global decoder on target patches.

        Args:
            tgt_bytes: (batch, tgt_byte_len) — target byte IDs.
            enc_output: (batch, num_src_patches, d_model).
            src_patch_pad_mask: (batch, num_src_patches).

        Returns:
            global_dec_output: (batch, num_tgt_patches, d_model)
        """
        # Encode target bytes into patches (reuse local encoder for consistency)
        tgt_patch_emb, tgt_patch_pad_mask = self.local_encoder(tgt_bytes)
        tgt_patch_emb = self.patch_pos_enc(tgt_patch_emb)

        num_tgt_patches = tgt_patch_emb.size(1)

        # Masks
        tgt_mask_pad = self._make_patch_pad_mask(tgt_patch_pad_mask)
        tgt_mask_causal = self._make_causal_patch_mask(num_tgt_patches, tgt_patch_emb.device)
        tgt_mask = tgt_mask_pad.expand(-1, -1, num_tgt_patches, -1) + tgt_mask_causal

        memory_mask = self._make_patch_pad_mask(src_patch_pad_mask)

        # Global decoder
        x = tgt_patch_emb
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return self.decoder_norm(x)

    def forward(
        self,
        src_bytes: torch.Tensor,
        tgt_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass for teacher-forced training.

        Args:
            src_bytes: (batch, src_byte_len) — source byte IDs.
            tgt_bytes: (batch, tgt_byte_len) — target byte IDs (with BOS prepended).

        Returns:
            logits: (batch, tgt_byte_len, BYTE_VOCAB_SIZE)
        """
        enc_output, src_pad_mask = self.encode(src_bytes)
        global_dec_output = self.decode_patches(tgt_bytes, enc_output, src_pad_mask)
        
        # Shift global_dec_output right by 1 patch to prevent data leak
        batch_size = global_dec_output.size(0)
        init_repr = self.init_patch_repr.expand(batch_size, 1, -1)
        shifted_global_dec_output = torch.cat([init_repr, global_dec_output[:, :-1, :]], dim=1)
        
        logits = self.local_decoder(tgt_bytes, shifted_global_dec_output)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src_bytes: torch.Tensor,
        max_len: int = 512,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding at the byte level.

        Decodes patch-by-patch: for each target patch, use the global decoder
        to get the patch representation, then autoregressively generate bytes
        within that patch using the local decoder.

        Args:
            src_bytes: (batch, src_byte_len) — source byte IDs.
            max_len: maximum number of target bytes to generate.

        Returns:
            (batch, decoded_len) — predicted byte IDs.
        """
        batch_size = src_bytes.size(0)
        device = src_bytes.device

        enc_output, src_pad_mask = self.encode(src_bytes)

        # Start with BOS
        generated = torch.full((batch_size, 1), BYTE_BOS, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        max_patches = max_len // self.patch_size + 1

        for patch_idx in range(max_patches):
            if finished.all():
                break

            if patch_idx == 0:
                last_patch_repr = self.init_patch_repr.expand(batch_size, 1, -1)
            else:
                # Run global decoder on all completed patches
                completed_bytes = generated[:, :patch_idx * self.patch_size]
                global_dec_output = self.decode_patches(completed_bytes, enc_output, src_pad_mask)
                last_patch_repr = global_dec_output[:, -1:, :]

            # The first byte of this patch is the LAST byte generated so far
            patch_bytes = generated[:, -1:]  # (batch, 1)

            for byte_idx in range(self.patch_size):
                # Embed and decode within patch
                x = self.local_decoder.byte_embedding(patch_bytes)
                x = self.local_decoder.byte_pos_enc(x)

                # Causal self-attention and cross-attention via step()
                causal_mask = self.local_decoder._make_causal_mask(x.size(1), device)
                global_proj = self.local_decoder.project_in(last_patch_repr)
                x = self.local_decoder.step(x, global_proj, causal_mask)

                # Predict next byte
                logits = self.local_decoder.output_proj(self.local_decoder.output_norm(x))
                next_byte = logits[:, -1, :].argmax(dim=-1)  # (batch,)

                # Mark finished if EOS
                finished = finished | (next_byte == BYTE_EOS)
                next_byte = next_byte.masked_fill(finished, BYTE_PAD)

                patch_bytes = torch.cat([patch_bytes, next_byte.unsqueeze(1)], dim=1)

            # Append generated patch bytes (skip the first byte which was already in 'generated')
            new_bytes = patch_bytes[:, 1:]  # (batch, patch_size)
            generated = torch.cat([generated, new_bytes], dim=1)

        # Remove initial BOS
        result = generated[:, 1:]
        return result

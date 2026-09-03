import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .norm import LayerNorm
from .positional import SinusoidalPositionalEncoding
from . import EncoderLayer, DecoderLayer

BYTE_PAD = 256
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_VOCAB_SIZE = 259


class LocalEncoder(nn.Module):

    def __init__(self, d_model=256, patch_size=4, d_local=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_local = d_local

        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, d_local, padding_idx=BYTE_PAD)
        self.byte_pos_enc = SinusoidalPositionalEncoding(d_local, max_len=512, dropout=dropout)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MultiHeadAttention(d_local, num_heads, dropout),
                'norm': LayerNorm(d_local),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_local))
        self.pool_attn = MultiHeadAttention(d_local, num_heads, dropout)
        self.pool_norm = LayerNorm(d_local)

        self.project = nn.Linear(d_local, d_model)

    def forward(self, byte_ids, boundaries=None):
        batch_size, seq_len = byte_ids.shape
        device = byte_ids.device

        if boundaries is None:
            # Fallback for decoding: fixed patch size
            boundaries = torch.zeros_like(byte_ids, dtype=torch.bool)
            boundaries[:, ::self.patch_size] = True
            boundaries[:, 0] = True 

        patch_ids = boundaries.cumsum(dim=1) - 1  # 0-indexed patches
        num_patches_per_seq = patch_ids[:, -1] + 1
        max_patches = num_patches_per_seq.max().item()

        # Local Attention Mask
        same_patch_mask = patch_ids.unsqueeze(2) == patch_ids.unsqueeze(1) # [B, L, L]
        pad_mask = byte_ids == BYTE_PAD
        
        attn_mask = (~same_patch_mask).float() * -1e9
        attn_mask = attn_mask.unsqueeze(1) # [B, 1, L, L]
        attn_mask = attn_mask.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), -1e9)

        x = self.byte_embedding(byte_ids)
        
        # Local Positional Encoding via cummax
        indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        start_indices = torch.cummax(indices * boundaries.long(), dim=1)[0]
        local_pos = indices - start_indices # [B, L]
        
        # apply positional encoding
        pe = self.byte_pos_enc.pe.squeeze(0).to(device)
        local_pos_clamped = torch.clamp(local_pos, max=pe.size(0)-1)
        pos_emb = F.embedding(local_pos_clamped, pe)
        x = self.byte_pos_enc.dropout(x + pos_emb)

        for layer in self.layers:
            x_norm = layer['norm'](x)
            x = x + layer['dropout'](layer['attn'](x_norm, x_norm, x_norm, mask=attn_mask))

        # Cross attention pooling
        query = self.pool_query.expand(batch_size, max_patches, -1)
        x_norm = self.pool_norm(x)
        
        k_idx = torch.arange(max_patches, device=device).view(1, -1, 1) # [1, max_patches, 1]
        cross_mask = patch_ids.unsqueeze(1) == k_idx # [B, max_patches, L]
        cross_attn_mask = (~cross_mask).float() * -1e9
        cross_attn_mask = cross_attn_mask.unsqueeze(1) # [B, 1, max_patches, L]
        cross_attn_mask = cross_attn_mask.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        pooled = self.pool_attn(query, x_norm, x_norm, mask=cross_attn_mask)
        patch_emb = self.project(pooled.squeeze(1)) # [B, max_patches, d_model]
        
        patch_idx = torch.arange(max_patches, device=device).unsqueeze(0)
        patch_pad_mask = patch_idx >= num_patches_per_seq.unsqueeze(1) # [B, max_patches]
        
        return patch_emb, patch_pad_mask


class BLTSeq2SeqModel(nn.Module):

    def __init__(self, d_model=256, num_heads=8, num_encoder_layers=4, num_decoder_layers=4,
                 d_ff=1024, dropout=0.1, max_seq_len=512, patch_size=4, d_local=128, local_heads=4,
                 num_local_layers=4):
        super().__init__()
        self.d_model = d_model
        self.src_patch_size = patch_size
        self.max_seq_len = max_seq_len

        self.local_encoder = LocalEncoder(d_model, self.src_patch_size, d_local, local_heads, num_local_layers, dropout)
        self.tgt_embedding = nn.Embedding(BYTE_VOCAB_SIZE, d_model, padding_idx=BYTE_PAD)

        self.patch_pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.tgt_pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, 'mha', 'layernorm', num_heads)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, 'mha', 'layernorm', num_heads)
            for _ in range(num_decoder_layers)
        ])

        self.encoder_norm = LayerNorm(d_model)
        self.decoder_norm = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, BYTE_VOCAB_SIZE)
        self.embed_dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # xavier overwrites padding rows, so re-zero them
        with torch.no_grad():
            self.tgt_embedding.weight[BYTE_PAD].zero_()
            self.local_encoder.byte_embedding.weight[BYTE_PAD].zero_()

    def _make_pad_mask(self, pad_mask):
        return pad_mask.unsqueeze(1).unsqueeze(2).float() * -1e9

    def _make_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0).float() * -1e9

    def encode(self, src_bytes, src_boundaries=None):
        patch_emb, patch_pad_mask = self.local_encoder(src_bytes, src_boundaries)
        patch_emb = self.patch_pos_enc(patch_emb)

        src_mask = self._make_pad_mask(patch_pad_mask)
        x = patch_emb

        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)

        return self.encoder_norm(x), patch_pad_mask

    def decode(self, tgt_bytes, enc_output, src_patch_pad_mask):
        x = self.tgt_embedding(tgt_bytes) * math.sqrt(self.d_model)
        x = self.tgt_pos_enc(x)

        tgt_len = tgt_bytes.size(1)
        tgt_pad_mask = tgt_bytes == BYTE_PAD
        tgt_mask_pad = self._make_pad_mask(tgt_pad_mask)
        tgt_mask_causal = self._make_causal_mask(tgt_len, tgt_bytes.device)
        tgt_mask = tgt_mask_pad.expand(-1, -1, tgt_len, -1) + tgt_mask_causal

        memory_mask = self._make_pad_mask(src_patch_pad_mask)

        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return self.decoder_norm(x)

    def forward(self, src_bytes, tgt_bytes, src_boundaries=None):
        enc_output, src_pad_mask = self.encode(src_bytes, src_boundaries)
        dec_output = self.decode(tgt_bytes, enc_output, src_pad_mask)
        return self.output_projection(dec_output)

    @torch.no_grad()
    def greedy_decode(self, src_bytes, src_boundaries=None, max_len=512):
        batch_size = src_bytes.size(0)
        device = src_bytes.device

        enc_output, src_pad_mask = self.encode(src_bytes, src_boundaries)
        ys = torch.full((batch_size, 1), BYTE_BOS, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            dec_output = self.decode(ys, enc_output, src_pad_mask)
            logits = self.output_projection(dec_output[:, -1, :])

            next_byte = logits.argmax(dim=-1)
            next_byte = next_byte.masked_fill(finished, BYTE_EOS)
            ys = torch.cat([ys, next_byte.unsqueeze(1)], dim=1)
            finished = finished | (next_byte == BYTE_EOS)

            if finished.all():
                break

        return ys[:, 1:]

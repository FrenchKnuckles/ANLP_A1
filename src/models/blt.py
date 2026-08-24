import math
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .norm import LayerNorm
from .positional import SinusoidalPositionalEncoding
from . import EncoderLayer, DecoderLayer

BYTE_PAD = 256
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_VOCAB_SIZE = 259


class LocalEncoder(nn.Module):

    def __init__(self, d_model=256, patch_size=4, d_local=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_local = d_local

        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, d_local, padding_idx=BYTE_PAD)
        self.byte_pos_enc = SinusoidalPositionalEncoding(d_local, max_len=patch_size, dropout=dropout)

        self.self_attn1 = MultiHeadAttention(d_local, num_heads, dropout)
        self.self_attn_norm1 = LayerNorm(d_local)
        self.self_attn_dropout1 = nn.Dropout(dropout)

        self.self_attn2 = MultiHeadAttention(d_local, num_heads, dropout)
        self.self_attn_norm2 = LayerNorm(d_local)
        self.self_attn_dropout2 = nn.Dropout(dropout)

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_local))
        self.pool_attn = MultiHeadAttention(d_local, num_heads, dropout)
        self.pool_norm = LayerNorm(d_local)

        self.project = nn.Linear(d_local, d_model)

    def forward(self, byte_ids):
        batch_size, seq_len = byte_ids.shape

        # strip BOS so the cipher content aligns cleanly to 9-byte patches
        bos_tok = byte_ids[:, :1]
        content = byte_ids[:, 1:]
        content_len = content.size(1)

        # pad content to a multiple of patch_size
        remainder = content_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            content = torch.nn.functional.pad(content, (0, pad_len), value=BYTE_PAD)
            content_len = content.size(1)

        num_patches = content_len // self.patch_size
        patches = content.view(batch_size, num_patches, self.patch_size)
        patch_pad_mask = (patches == BYTE_PAD).all(dim=-1)

        # flatten patches for parallel self-attention
        patches_flat = patches.view(batch_size * num_patches, self.patch_size)
        pad_mask = patches_flat == BYTE_PAD
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2).float() * -1e9

        x = self.byte_embedding(patches_flat)
        x = self.byte_pos_enc(x)

        # two layers of pre-norm self attention within each patch
        x_norm = self.self_attn_norm1(x)
        x = x + self.self_attn_dropout1(self.self_attn1(x_norm, x_norm, x_norm, mask=attn_mask))

        x_norm = self.self_attn_norm2(x)
        x = x + self.self_attn_dropout2(self.self_attn2(x_norm, x_norm, x_norm, mask=attn_mask))

        # cross-attention pooling: learnable query compresses each patch to one vector
        query = self.pool_query.expand(batch_size * num_patches, -1, -1)
        x_norm = self.pool_norm(x)
        pooled = self.pool_attn(query, x_norm, x_norm, mask=attn_mask)

        patch_emb = self.project(pooled.squeeze(1))
        patch_emb = patch_emb.view(batch_size, num_patches, self.d_model)

        # embed BOS separately and prepend it
        bos_emb = self.project(self.byte_embedding(bos_tok).squeeze(1)).unsqueeze(1)
        patch_emb = torch.cat([bos_emb, patch_emb], dim=1)

        bos_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=byte_ids.device)
        patch_pad_mask = torch.cat([bos_pad, patch_pad_mask], dim=1)

        return patch_emb, patch_pad_mask


class BLTSeq2SeqModel(nn.Module):

    def __init__(self, d_model=256, num_heads=8, num_encoder_layers=4, num_decoder_layers=4,
                 d_ff=1024, dropout=0.1, max_seq_len=512, patch_size=4, d_local=128, local_heads=4):
        super().__init__()
        self.d_model = d_model
        self.src_patch_size = patch_size
        self.max_seq_len = max_seq_len

        self.local_encoder = LocalEncoder(d_model, self.src_patch_size, d_local, local_heads, dropout)
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

    def encode(self, src_bytes):
        patch_emb, patch_pad_mask = self.local_encoder(src_bytes)
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

    def forward(self, src_bytes, tgt_bytes):
        enc_output, src_pad_mask = self.encode(src_bytes)
        dec_output = self.decode(tgt_bytes, enc_output, src_pad_mask)
        return self.output_projection(dec_output)

    @torch.no_grad()
    def greedy_decode(self, src_bytes, max_len=512):
        batch_size = src_bytes.size(0)
        device = src_bytes.device

        enc_output, src_pad_mask = self.encode(src_bytes)
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

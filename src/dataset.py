"""
Dataset module — handles the provided cipher-plaintext dataset for both
tokenized (C1–C4) and token-free/BLT (C5) modes.

Dataset source: provided `Dataset_A1/brown_cipher.txt` (binary 0/1 strings) and
`Dataset_A1/brown_plain.txt` (English text).

The cipher is the 8-bit binary representation of each plaintext character.

Preprocessing:
    - We chunk the plain text into 128-character blocks and the cipher into
      matching 1024-bit blocks. This produces ~25k examples and ensures
      sensible sequence lengths.
    - Caches chunked dataset to `.cache/chunked_dataset.json`.

Tokenized mode (C1–C4):
    - Trains two separate byte-level BPE tokenizers (src and tgt) on the training split.
    - Returns (src_ids, tgt_ids) with PAD/BOS/EOS special tokens.

Token-free mode (C5):
    - Raw byte tensors: each character of '0'/'1' cipher string is a byte.
    - Vocab size = 259 (256 bytes + PAD/BOS/EOS sentinels past byte range).
"""

import os
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


# ── Constants ───────────────────────────────────────────────────────────
# dataset.py is at src/dataset.py, so dirname(dirname(__file__)) = project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_PROJECT_ROOT, "Dataset_A1")
CACHE_DIR = os.path.join(_PROJECT_ROOT, ".cache")

PLAIN_CHUNK_SIZE = 128
MIN_CHUNK_CHARS = 32

# BPE special tokens
BPE_PAD = "<pad>"
BPE_BOS = "<bos>"
BPE_EOS = "<eos>"
BPE_UNK = "<unk>"

# Byte-level (BLT) special tokens — sentinel values past byte range
BYTE_PAD = 256
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_VOCAB_SIZE = 259


# ── Data Loading & Chunking ───────────────────────────────────────────

def chunk_dataset(cipher_lines: list[str], plain_lines: list[str], plain_chunk_size: int = PLAIN_CHUNK_SIZE) -> tuple[list[str], list[str]]:
    """Chunk dataset aligned by character."""
    assert len(cipher_lines) == len(plain_lines), \
        f"Cipher ({len(cipher_lines)}) and plain ({len(plain_lines)}) line counts don't match"

    chunked_cipher = []
    chunked_plain = []
    
    for c, p in zip(cipher_lines, plain_lines):
        # 1 character = 8 bits
        for i in range(0, len(p), plain_chunk_size):
            p_chunk = p[i:i+plain_chunk_size]
            c_chunk_raw = c[i*8:(i+plain_chunk_size)*8]
            
            if len(p_chunk) < MIN_CHUNK_CHARS:
                continue
            
            # insert '|' every 8 bits (after every byte)
            c_chunk = ""
            for j in range(0, len(c_chunk_raw), 8):
                c_chunk += c_chunk_raw[j:j+8] + "|"
                
            if len(p_chunk) > 0:
                chunked_plain.append(p_chunk)
                chunked_cipher.append(c_chunk)
                
    return chunked_cipher, chunked_plain


def load_raw_lines(data_dir: str) -> tuple[list[str], list[str]]:
    cipher_path = os.path.join(data_dir, "brown_cipher.txt")
    plain_path = os.path.join(data_dir, "brown_plain.txt")

    with open(cipher_path, "r", encoding="utf-8") as f:
        cipher_lines = [line.strip() for line in f if line.strip()]
    with open(plain_path, "r", encoding="utf-8") as f:
        plain_lines = [line.strip() for line in f if line.strip()]

    return cipher_lines, plain_lines


def split_data(
    cipher_lines: list[str],
    plain_lines: list[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict:
    """Split data into train/val/test with fixed seed. 80/10/10."""
    import random
    n = len(cipher_lines)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {}
    for name, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        c_lines = [cipher_lines[i] for i in idx_list]
        p_lines = [plain_lines[i] for i in idx_list]
        
        c_chunked, p_chunked = chunk_dataset(c_lines, p_lines, plain_chunk_size=PLAIN_CHUNK_SIZE)
        
        splits[name] = {
            "cipher": c_chunked,
            "plain": p_chunked,
        }

    return splits


def get_split_data_cached(data_dir: str, cache_dir: str, seed: int = 42) -> dict:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"splits_v3_{seed}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    cipher_lines, plain_lines = load_raw_lines(data_dir)
    splits = split_data(cipher_lines, plain_lines, seed=seed)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)
        
    return splits


# ── BPE Tokenizer (C1–C4) ─────────────────────────────────────────────

def train_single_tokenizer(texts: list[str], vocab_size: int, save_path: str) -> Tokenizer:
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)

    tokenizer = Tokenizer(BPE(unk_token=BPE_UNK))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[BPE_PAD, BPE_BOS, BPE_EOS, BPE_UNK],
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    bos_id = tokenizer.token_to_id(BPE_BOS)
    eos_id = tokenizer.token_to_id(BPE_EOS)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{BPE_BOS}:0 $A:0 {BPE_EOS}:0",
        special_tokens=[(BPE_BOS, bos_id), (BPE_EOS, eos_id)],
    )

    pad_id = tokenizer.token_to_id(BPE_PAD)
    tokenizer.enable_padding(pad_id=pad_id, pad_token=BPE_PAD)

    tokenizer.save(save_path)
    return tokenizer

def train_bpe_tokenizers(
    train_cipher: list[str],
    train_plain: list[str],
    vocab_size: int = 8000,
    cache_dir: str = CACHE_DIR,
) -> tuple[Tokenizer, Tokenizer]:
    """Train separate BPE tokenizers for source and target."""
    os.makedirs(cache_dir, exist_ok=True)
    src_path = os.path.join(cache_dir, "bpe_tokenizer_src_v3.json")
    tgt_path = os.path.join(cache_dir, "bpe_tokenizer_tgt_v3.json")
    
    src_tokenizer = train_single_tokenizer(train_cipher, vocab_size, src_path)
    tgt_tokenizer = train_single_tokenizer(train_plain, vocab_size, tgt_path)
    
    return src_tokenizer, tgt_tokenizer


def get_bpe_special_ids(tokenizer: Tokenizer) -> dict:
    """Get special token IDs from trained BPE tokenizer."""
    return {
        "pad": tokenizer.token_to_id(BPE_PAD),
        "bos": tokenizer.token_to_id(BPE_BOS),
        "eos": tokenizer.token_to_id(BPE_EOS),
    }


# ── Tokenized Dataset (C1–C4) ─────────────────────────────────────────

class CipherDatasetTokenized(Dataset):
    """Dataset for tokenized mode (C1–C4).

    Source: BPE-tokenized cipher binary strings.
    Target: BPE-tokenized plaintext.
    Both include BOS/EOS tokens (added by tokenizer post-processing).
    """

    def __init__(
        self,
        cipher_lines: list[str],
        plain_lines: list[str],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_seq_len: int = 512,
    ):
        self.cipher_lines = cipher_lines
        self.plain_lines = plain_lines
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = src_tokenizer.token_to_id(BPE_PAD)

    def __len__(self) -> int:
        return len(self.cipher_lines)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cipher = self.cipher_lines[idx]
        plain = self.plain_lines[idx]

        # Tokenize (BOS/EOS are added by post-processor)
        src_enc = self.src_tokenizer.encode(cipher)
        tgt_enc = self.tgt_tokenizer.encode(plain)

        src_ids = src_enc.ids[:self.max_seq_len]
        tgt_ids = tgt_enc.ids[:self.max_seq_len]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


# ── Token-Free Dataset (C5 — BLT) ────────────────────────────────────

class CipherDatasetTokenFree(Dataset):
    """Dataset for token-free mode (C5 — BLT).

    Source: raw byte tensors of cipher binary string characters ('0' and '1').
    Target: raw byte tensors of plaintext characters.
    BOS/EOS sentinels are added past the 0–255 byte range (vocab size 259).
    """

    def __init__(
        self,
        cipher_lines: list[str],
        plain_lines: list[str],
        max_byte_len: int = 2048,
    ):
        self.cipher_lines = cipher_lines
        self.plain_lines = plain_lines
        self.max_byte_len = max_byte_len

    def __len__(self) -> int:
        return len(self.cipher_lines)

    def _str_to_bytes(self, s: str, max_len: int) -> torch.Tensor:
        """Convert string to byte tensor with BOS/EOS sentinels."""
        byte_vals = list(s.encode("utf-8"))[:max_len - 2]  # Leave room for BOS/EOS
        return torch.tensor([BYTE_BOS] + byte_vals + [BYTE_EOS], dtype=torch.long)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cipher = self.cipher_lines[idx]
        plain = self.plain_lines[idx]

        src = self._str_to_bytes(cipher, self.max_byte_len)
        tgt = self._str_to_bytes(plain, self.max_byte_len)

        return src, tgt


# ── Collate Functions ─────────────────────────────────────────────────

def collate_tokenized(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int = 0):
    """Pad sequences in a batch to the same length."""
    src_list, tgt_list = zip(*batch)

    src_max = max(s.size(0) for s in src_list)
    tgt_max = max(t.size(0) for t in tgt_list)

    src_batch = torch.full((len(batch), src_max), pad_id, dtype=torch.long)
    tgt_batch = torch.full((len(batch), tgt_max), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_batch[i, :s.size(0)] = s
        tgt_batch[i, :t.size(0)] = t

    return src_batch, tgt_batch


def collate_token_free(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Pad byte sequences using BYTE_PAD sentinel."""
    return collate_tokenized(batch, pad_id=BYTE_PAD)


# ── High-Level API ────────────────────────────────────────────────────

def build_dataloaders(
    tokenization: str = "subword",
    batch_size: int = 64,
    max_seq_len: int = 512,
    vocab_size: int = 8000,
    seed: int = 42,
    num_workers: int = 0,
    data_dir: str = DATASET_DIR,
) -> dict:
    """Build train/val/test dataloaders and tokenizer info."""
    splits = get_split_data_cached(data_dir, CACHE_DIR, seed=seed)

    if tokenization == "subword":
        src_tokenizer, tgt_tokenizer = train_bpe_tokenizers(
            splits["train"]["cipher"], splits["train"]["plain"], vocab_size=vocab_size
        )
        special_ids = get_bpe_special_ids(src_tokenizer)

        datasets = {}
        for split_name in ["train", "val", "test"]:
            datasets[split_name] = CipherDatasetTokenized(
                splits[split_name]["cipher"],
                splits[split_name]["plain"],
                src_tokenizer,
                tgt_tokenizer,
                max_seq_len,
            )

        collate_fn = lambda batch: collate_tokenized(batch, pad_id=special_ids["pad"])

        info = {
            "src_vocab_size": src_tokenizer.get_vocab_size(),
            "tgt_vocab_size": tgt_tokenizer.get_vocab_size(),
            "pad_idx": special_ids["pad"],
            "bos_idx": special_ids["bos"],
            "eos_idx": special_ids["eos"],
            "tokenizer_src": src_tokenizer,
            "tokenizer_tgt": tgt_tokenizer,
        }

    elif tokenization == "blt":
        # For token-free byte-level, the provided max_seq_len applies to bytes
        max_byte_len = max_seq_len

        datasets = {}
        for split_name in ["train", "val", "test"]:
            datasets[split_name] = CipherDatasetTokenFree(
                splits[split_name]["cipher"],
                splits[split_name]["plain"],
                max_byte_len,
            )

        collate_fn = collate_token_free

        info = {
            "src_vocab_size": BYTE_VOCAB_SIZE,
            "tgt_vocab_size": BYTE_VOCAB_SIZE,
            "pad_idx": BYTE_PAD,
            "bos_idx": BYTE_BOS,
            "eos_idx": BYTE_EOS,
            "tokenizer_src": None,
            "tokenizer_tgt": None,
        }

    else:
        raise ValueError(f"Unknown tokenization mode: {tokenization}")

    # Build dataloaders
    train_loader = DataLoader(
        datasets["train"], batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "splits": splits,
        **info,
    }

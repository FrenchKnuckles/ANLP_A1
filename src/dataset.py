import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
import random

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_PROJECT_ROOT, 'Dataset_A1')
CACHE_DIR = os.path.join(_PROJECT_ROOT, '.cache')

PLAIN_CHUNK_SIZE = 128
MIN_CHUNK_CHARS = 32

BPE_PAD = '<pad>'
BPE_BOS = '<bos>'
BPE_EOS = '<eos>'
BPE_UNK = '<unk>'

BYTE_PAD = 256
BYTE_BOS = 257
BYTE_EOS = 258
BYTE_VOCAB_SIZE = 259


def chunk_dataset(cipher_lines, plain_lines, plain_chunk_size=PLAIN_CHUNK_SIZE):
    assert len(cipher_lines) == len(plain_lines)

    chunked_cipher = []
    chunked_plain = []

    for c, p in zip(cipher_lines, plain_lines):
        for i in range(0, len(p), plain_chunk_size):
            p_chunk = p[i:i + plain_chunk_size]
            c_chunk_raw = c[i * 8:(i + plain_chunk_size) * 8]

            if len(p_chunk) < MIN_CHUNK_CHARS:
                continue

            # insert '|' separator every 8 bits to form 9-byte groups
            c_chunk = ''
            for j in range(0, len(c_chunk_raw), 8):
                c_chunk += c_chunk_raw[j:j + 8] + '|'

            if len(p_chunk) > 0:
                chunked_plain.append(p_chunk)
                chunked_cipher.append(c_chunk)

    return chunked_cipher, chunked_plain


def load_raw_lines(data_dir):
    cipher_path = os.path.join(data_dir, 'brown_cipher.txt')
    plain_path = os.path.join(data_dir, 'brown_plain.txt')

    with open(cipher_path, 'r', encoding='utf-8') as f:
        cipher_lines = [line.strip() for line in f if line.strip()]

    with open(plain_path, 'r', encoding='utf-8') as f:
        plain_lines = [line.strip() for line in f if line.strip()]

    return cipher_lines, plain_lines


def split_data(cipher_lines, plain_lines, seed=42, train_ratio=0.8, val_ratio=0.1):
    n = len(cipher_lines)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {}
    for name, idx_list in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        c_lines = [cipher_lines[i] for i in idx_list]
        p_lines = [plain_lines[i] for i in idx_list]
        c_chunked, p_chunked = chunk_dataset(c_lines, p_lines, plain_chunk_size=PLAIN_CHUNK_SIZE)
        splits[name] = {'cipher': c_chunked, 'plain': p_chunked}

    return splits


def get_split_data_cached(data_dir, cache_dir, seed=42):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'splits_v3_{seed}.json')

    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    cipher_lines, plain_lines = load_raw_lines(data_dir)
    splits = split_data(cipher_lines, plain_lines, seed=seed)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f)

    return splits


def train_single_tokenizer(texts, vocab_size, save_path):
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)

    tokenizer = Tokenizer(BPE(unk_token=BPE_UNK))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[BPE_PAD, BPE_BOS, BPE_EOS, BPE_UNK],
        show_progress=True
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    bos_id = tokenizer.token_to_id(BPE_BOS)
    eos_id = tokenizer.token_to_id(BPE_EOS)
    tokenizer.post_processor = TemplateProcessing(
        single=f'{BPE_BOS}:0 $A:0 {BPE_EOS}:0',
        special_tokens=[(BPE_BOS, bos_id), (BPE_EOS, eos_id)]
    )

    pad_id = tokenizer.token_to_id(BPE_PAD)
    tokenizer.enable_padding(pad_id=pad_id, pad_token=BPE_PAD)
    tokenizer.save(save_path)

    return tokenizer


def train_bpe_tokenizers(train_cipher, train_plain, vocab_size=8000, cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    src_path = os.path.join(cache_dir, 'bpe_tokenizer_src_v3.json')
    tgt_path = os.path.join(cache_dir, 'bpe_tokenizer_tgt_v3.json')

    src_tokenizer = train_single_tokenizer(train_cipher, vocab_size, src_path)
    tgt_tokenizer = train_single_tokenizer(train_plain, vocab_size, tgt_path)

    return src_tokenizer, tgt_tokenizer


def get_bpe_special_ids(tokenizer):
    return {
        'pad': tokenizer.token_to_id(BPE_PAD),
        'bos': tokenizer.token_to_id(BPE_BOS),
        'eos': tokenizer.token_to_id(BPE_EOS),
    }


class CipherDatasetTokenized(Dataset):

    def __init__(self, cipher_lines, plain_lines, src_tokenizer, tgt_tokenizer, max_seq_len=512):
        self.cipher_lines = cipher_lines
        self.plain_lines = plain_lines
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len
        self.pad_id = src_tokenizer.token_to_id(BPE_PAD)

    def __len__(self):
        return len(self.cipher_lines)

    def __getitem__(self, idx):
        cipher = self.cipher_lines[idx]
        plain = self.plain_lines[idx]

        src_enc = self.src_tokenizer.encode(cipher)
        tgt_enc = self.tgt_tokenizer.encode(plain)

        src_ids = src_enc.ids[:self.max_seq_len]
        tgt_ids = tgt_enc.ids[:self.max_seq_len]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


class CipherDatasetTokenFree(Dataset):

    def __init__(self, cipher_lines, plain_lines, max_byte_len=2048):
        self.cipher_lines = cipher_lines
        self.plain_lines = plain_lines
        self.max_byte_len = max_byte_len

    def __len__(self):
        return len(self.cipher_lines)

    def _str_to_bytes(self, s, max_len):
        byte_vals = list(s.encode('utf-8'))[:max_len - 2]
        return torch.tensor([BYTE_BOS] + byte_vals + [BYTE_EOS], dtype=torch.long)

    def __getitem__(self, idx):
        cipher = self.cipher_lines[idx]
        plain = self.plain_lines[idx]
        src = self._str_to_bytes(cipher, self.max_byte_len)
        tgt = self._str_to_bytes(plain, self.max_byte_len)
        return src, tgt


def collate_tokenized(batch, pad_id=0):
    src_list, tgt_list = zip(*batch)
    src_max = max(s.size(0) for s in src_list)
    tgt_max = max(t.size(0) for t in tgt_list)

    src_batch = torch.full((len(batch), src_max), pad_id, dtype=torch.long)
    tgt_batch = torch.full((len(batch), tgt_max), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_batch[i, :s.size(0)] = s
        tgt_batch[i, :t.size(0)] = t

    return src_batch, tgt_batch


def collate_token_free(batch):
    return collate_tokenized(batch, pad_id=BYTE_PAD)


def build_dataloaders(tokenization='subword', batch_size=64, max_seq_len=512,
                      vocab_size=8000, seed=42, num_workers=0, data_dir=DATASET_DIR):

    splits = get_split_data_cached(data_dir, CACHE_DIR, seed=seed)

    if tokenization == 'subword':
        src_tokenizer, tgt_tokenizer = train_bpe_tokenizers(
            splits['train']['cipher'], splits['train']['plain'], vocab_size=vocab_size
        )
        special_ids = get_bpe_special_ids(src_tokenizer)
        datasets = {}
        for split_name in ['train', 'val', 'test']:
            datasets[split_name] = CipherDatasetTokenized(
                splits[split_name]['cipher'], splits[split_name]['plain'],
                src_tokenizer, tgt_tokenizer, max_seq_len
            )
        collate_fn = lambda batch: collate_tokenized(batch, pad_id=special_ids['pad'])
        info = {
            'src_vocab_size': src_tokenizer.get_vocab_size(),
            'tgt_vocab_size': tgt_tokenizer.get_vocab_size(),
            'pad_idx': special_ids['pad'],
            'bos_idx': special_ids['bos'],
            'eos_idx': special_ids['eos'],
            'tokenizer_src': src_tokenizer,
            'tokenizer_tgt': tgt_tokenizer,
        }

    elif tokenization == 'blt':
        max_byte_len = max_seq_len
        datasets = {}
        for split_name in ['train', 'val', 'test']:
            datasets[split_name] = CipherDatasetTokenFree(
                splits[split_name]['cipher'], splits[split_name]['plain'], max_byte_len
            )
        collate_fn = collate_token_free
        info = {
            'src_vocab_size': BYTE_VOCAB_SIZE,
            'tgt_vocab_size': BYTE_VOCAB_SIZE,
            'pad_idx': BYTE_PAD,
            'bos_idx': BYTE_BOS,
            'eos_idx': BYTE_EOS,
            'tokenizer_src': None,
            'tokenizer_tgt': None,
        }

    else:
        raise ValueError(f'Unknown tokenization mode: {tokenization}')

    train_loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'splits': splits,
        **info,
    }

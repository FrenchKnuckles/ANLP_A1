import os
import json
import re
import heapq
from collections import Counter, defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
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


class Encoded:
    __slots__ = ('ids',)

    def __init__(self, ids):
        self.ids = ids


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

            if len(p_chunk) > 0:
                chunked_plain.append(p_chunk)
                chunked_cipher.append(c_chunk_raw)

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
    cache_path = os.path.join(cache_dir, f'splits_v6_{seed}.json')

    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    cipher_lines, plain_lines = load_raw_lines(data_dir)
    splits = split_data(cipher_lines, plain_lines, seed=seed)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(splits, f)

    return splits


class BPETokenizer:
    def __init__(self, pre_tokenize_mode='none'):
        # pre_tokenize_mode: 'none', 'whitespace', or 'bytes'
        #   'none'       - no pre-tokenization, entire text is one word
        #   'whitespace' - split on word/punctuation boundaries (for English plaintext)
        #   'bytes'      - split binary string into 8-char chunks (for cipher)
        self.pre_tokenize_mode = pre_tokenize_mode
        self.vocab = {}
        self.merges = {}
        self.special_tokens = {
            BPE_PAD: 0,
            BPE_BOS: 1,
            BPE_EOS: 2,
            BPE_UNK: 3
        }
        self.id_to_token = {}
        
        # Initialize with base byte vocabulary (0-255)
        for name, id_ in self.special_tokens.items():
            self.vocab[id_] = name.encode('utf-8') if isinstance(name, str) else name
            self.id_to_token[id_] = self.vocab[id_]
            
        for i in range(256):
            id_ = i + len(self.special_tokens)
            b = bytes([i])
            self.vocab[id_] = b
            self.id_to_token[id_] = b
            
        self._next_id = 256 + len(self.special_tokens)

    def _pre_tokenize(self, text):
        if self.pre_tokenize_mode == 'whitespace':
            return re.findall(r'\s*\w+|\s*[^\w\s]|\s+', text)
        elif self.pre_tokenize_mode == 'bytes':
            # Split binary string into 8-bit "byte words"
            # BPE can still merge within or across these boundaries
            # but the training deduplication works on 8-char chunks
            return [text[i:i+8] for i in range(0, len(text), 8)]
        else:
            return [text]

    def train_from_iterator(self, texts, vocab_size):
        # Dedupe identical chunks up front and carry a count for each unique
        # one. The cipher side in particular repeats a lot (repeating-key XOR
        # means the same byte pattern recurs every period), so this alone can
        # collapse a big chunk of the corpus before any counting starts.
        word_counts = Counter()
        for text in texts:
            chunks = self._pre_tokenize(text)
                
            for chunk in chunks:
                byte_seq = chunk.encode('utf-8')
                ids = tuple(b + len(self.special_tokens) for b in byte_seq)
                word_counts[ids] += 1

        words = [list(ids) for ids in word_counts]
        counts = list(word_counts.values())

        # pair_freq: pair -> total weighted occurrence count across the corpus.
        # pair_to_words: pair -> set of word indices that currently contain it.
        # Together these let a merge update touch only the words it actually
        # changed, instead of re-scanning every word on every iteration.
        pair_freq = defaultdict(int)
        pair_to_words = defaultdict(set)
        for wi, word in enumerate(words):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += counts[wi]
                pair_to_words[pair].add(wi)

        while self._next_id < vocab_size:
            if not pair_freq:
                break

            # Tie-break on the pair itself so the result is reproducible
            # regardless of dict iteration order (the original used plain
            # max() over a dict, which lets ties resolve arbitrarily).
            best_pair = max(pair_freq, key=lambda p: (pair_freq[p], p))

            new_id = self._next_id
            self._next_id += 1
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.id_to_token[new_id] = self.vocab[new_id]

            affected = set(pair_to_words.get(best_pair, ()))

            for wi in affected:
                word = words[wi]
                c = counts[wi]

                # Remove this word's old pair counts before merging it.
                old_pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
                for p in old_pairs:
                    pair_freq[p] -= c
                    if pair_freq[p] <= 0:
                        del pair_freq[p]
                for p in set(old_pairs):
                    pair_to_words[p].discard(wi)

                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_id)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                words[wi] = new_word

                # Add back this word's new pair counts after merging.
                new_pairs = [(new_word[i], new_word[i + 1]) for i in range(len(new_word) - 1)]
                for p in new_pairs:
                    pair_freq[p] += c
                for p in set(new_pairs):
                    pair_to_words[p].add(wi)

            if self._next_id % 100 == 0:
                print(f"Vocab size: {self._next_id}/{vocab_size}")

    def _encode_chunk(self, ids):
        n = len(ids)
        if n < 2 or not self.merges:
            return ids

        # Walk the sequence as a doubly linked list (nxt/prv over positions)
        # so applying a merge is an O(1) splice instead of rebuilding the
        # whole list, and use a min-heap keyed by merge rank instead of
        # rescanning every remaining pair to find the next one to apply.
        # merges[pair] == the id that pair produces, and ids were assigned
        # in creation order, so "smallest id" is exactly "earliest merge" --
        # same tie-breaking the original min_rank scan used, just found
        # via a heap instead of a full linear scan each iteration.
        nxt = list(range(1, n)) + [-1]
        prv = [-1] + list(range(0, n - 1))
        alive = [True] * n
        heap = []

        def push_pair(i):
            j = nxt[i]
            if j == -1:
                return
            rank = self.merges.get((ids[i], ids[j]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        for i in range(n):
            push_pair(i)

        while heap:
            rank, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            # Re-check against the current pair at this position: earlier
            # merges elsewhere may have changed what's adjacent to i since
            # this heap entry was pushed, so a stale entry just gets skipped.
            if self.merges.get((ids[i], ids[j])) != rank:
                continue

            ids[i] = rank
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prv[k] = i

            if prv[i] != -1:
                push_pair(prv[i])
            push_pair(i)

        out = []
        i = 0
        while i != -1:
            out.append(ids[i])
            i = nxt[i]

        return out

    def encode(self, text):
        chunks = self._pre_tokenize(text)
            
        all_out = []
        for chunk in chunks:
            byte_seq = chunk.encode('utf-8')
            ids = [b + len(self.special_tokens) for b in byte_seq]
            all_out.extend(self._encode_chunk(ids))
            
        return Encoded([self.special_tokens[BPE_BOS]] + all_out + [self.special_tokens[BPE_EOS]])
        
    def decode(self, ids):
        b = bytearray()
        for i in ids:
            if i in self.special_tokens.values():
                continue
            if i in self.vocab:
                b.extend(self.vocab[i])
        return b.decode('utf-8', errors='replace')
        
    def save(self, path):
        with open(path, 'w') as f:
            merges_str = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
            json.dump({'merges': merges_str, 'next_id': self._next_id, 'pre_tokenize_mode': self.pre_tokenize_mode}, f)
            
    def load(self, path):
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            data = json.load(f)
            self._next_id = data['next_id']
            self.pre_tokenize_mode = data.get('pre_tokenize_mode', 'none')
            self.merges = {}
            for k, v in data['merges'].items():
                p1, p2 = map(int, k.split(','))
                self.merges[(p1, p2)] = v
                
            for pair, new_id in sorted(self.merges.items(), key=lambda x: x[1]):
                self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
                self.id_to_token[new_id] = self.vocab[new_id]
                
    @classmethod
    def from_file(cls, path):
        tok = cls()
        tok.load(path)
        return tok
        
    def get_vocab_size(self):
        return self._next_id
        
    def token_to_id(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        return None


def train_single_tokenizer(texts, vocab_size, save_path, pre_tokenize_mode='none'):
    if os.path.exists(save_path):
        return BPETokenizer.from_file(save_path)

    tokenizer = BPETokenizer(pre_tokenize_mode=pre_tokenize_mode)
    tokenizer.train_from_iterator(texts, vocab_size)
    tokenizer.save(save_path)

    return tokenizer


def train_bpe_tokenizers(train_cipher, train_plain, vocab_size=8000, cache_dir=CACHE_DIR):
    os.makedirs(cache_dir, exist_ok=True)
    src_path = os.path.join(cache_dir, 'custom_bpe_tokenizer_src_v4.json')
    tgt_path = os.path.join(cache_dir, 'custom_bpe_tokenizer_tgt_v4.json')

    # Cipher (binary 0/1 stream): no pre-tokenization, smaller vocab to avoid
    # over-compression. With vocab_size=1000 on a 2-char alphabet, BPE creates
    # tokens of ~5-10 bits, giving ~100-200 src tokens per 1024-bit chunk.
    # This produces a much better src:tgt ratio for the Transformer.
    src_tokenizer = train_single_tokenizer(train_cipher, min(vocab_size, 1000), src_path, pre_tokenize_mode='none')
    
    # Plaintext (English): word-boundary pre-tokenization.
    # We also cap this at 1000. In decipherment, large vocabs (8000) force the
    # model to memorize bit-patterns for entire words, which fails on rare words.
    # A vocab of 1000 forces smaller 2-3 character chunks, helping the model
    # learn the actual compositional 8-bit to 1-char mapping rules.
    tgt_tokenizer = train_single_tokenizer(train_plain, min(vocab_size, 1000), tgt_path, pre_tokenize_mode='whitespace')

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
        
        # BLT requires 9-byte patches (8 bits + '|' separator)
        patched_cipher = ''
        for j in range(0, len(cipher), 8):
            patched_cipher += cipher[j:j + 8] + '|'
            
        src = self._str_to_bytes(patched_cipher, self.max_byte_len)
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

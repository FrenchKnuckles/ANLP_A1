import os
import json
import statistics
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

DATASET_DIR = "Dataset_A1"

def analyze_dataset():
    cipher_path = os.path.join(DATASET_DIR, "brown_cipher.txt")
    plain_path = os.path.join(DATASET_DIR, "brown_plain.txt")
    
    with open(cipher_path, "r", encoding="utf-8") as f:
        cipher_lines = [l.strip() for l in f if l.strip()]
    with open(plain_path, "r", encoding="utf-8") as f:
        plain_lines = [l.strip() for l in f if l.strip()]
        
    # Chunking
    chunk_size = 128
    chunked_cipher = []
    chunked_plain = []
    
    for c, p in zip(cipher_lines, plain_lines):
        # c is 8 times length of p
        for i in range(0, len(p), chunk_size):
            p_chunk = p[i:i+chunk_size]
            c_chunk = c[i*8:(i+chunk_size)*8]
            if len(p_chunk) > 0:
                chunked_plain.append(p_chunk)
                chunked_cipher.append(c_chunk)
                
    print(f"Original pairs: {len(plain_lines)}")
    print(f"Chunked pairs (len={chunk_size}): {len(chunked_plain)}")
    
    # Train separate tokenizers
    def train_tok(texts, name, vocab=4000):
        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(vocab_size=vocab, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"], show_progress=False)
        tok.train_from_iterator(texts, trainer=trainer)
        return tok
        
    print("Training source (cipher) tokenizer...")
    src_tok = train_tok(chunked_cipher, "src", vocab=4000)
    print("Training target (plain) tokenizer...")
    tgt_tok = train_tok(chunked_plain, "tgt", vocab=4000)
    
    src_lens = [len(src_tok.encode(t).ids) for t in chunked_cipher]
    tgt_lens = [len(tgt_tok.encode(t).ids) for t in chunked_plain]
    
    def print_stats(name, lens):
        lens.sort()
        mean = statistics.mean(lens)
        median = statistics.median(lens)
        p95 = lens[int(len(lens) * 0.95)]
        max_len = max(lens)
        print(f"[{name}] Token Counts: Min: {min(lens)}, Mean: {mean:.1f}, Median: {median}, 95th: {p95}, Max: {max_len}")
        
    print_stats("Cipher (BPE)", src_lens)
    print_stats("Plain (BPE)", tgt_lens)
    
    # For BLT (bytes)
    byte_lens = [len(c.encode("utf-8")) for c in chunked_cipher]
    print_stats("Cipher (Bytes)", byte_lens)

if __name__ == "__main__":
    analyze_dataset()

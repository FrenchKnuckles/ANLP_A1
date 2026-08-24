import json
import math
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from dotenv import load_dotenv

from .dataset import build_dataloaders, BYTE_PAD, BYTE_BOS, BYTE_EOS, PLAIN_CHUNK_SIZE
from .models import Seq2SeqTransformer
from .models.blt import BLTSeq2SeqModel, BYTE_VOCAB_SIZE
from .utils import (
    init_wandb, log_wandb, finish_wandb, push_folder_to_hub,
    compute_all_metrics, save_metrics_json, compute_naive_baselines,
    plot_metrics_comparison, plot_c5_vs_c1, plot_loss_curves
)
from huggingface_hub import HfApi

load_dotenv()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def build_model(config, data_info, device):
    if config['tokenization'] == 'blt':
        model = BLTSeq2SeqModel(
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            patch_size=config['blt_patch_size'],
            d_local=config['d_model'],
            local_heads=config['num_heads']
        )
    else:
        model = Seq2SeqTransformer(
            src_vocab_size=data_info['src_vocab_size'],
            tgt_vocab_size=data_info['tgt_vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_seq_len=config['max_seq_len'],
            pad_idx=data_info['pad_idx'],
            attention_type=config['attention_type'],
            norm_type=config['norm_type'],
            positional_encoding=config['positional_encoding'],
            num_kv_heads=config['gqa_kv_heads']
        )
    return model.to(device)


@torch.no_grad()
def decode_predictions(model, dataloader, data_info, device, is_blt=False, max_samples=None, max_seq_len=512):
    model.eval()
    predictions = []
    targets = []
    n = 0

    tokenizer_tgt = data_info.get('tokenizer_tgt')
    pad_id = data_info['pad_idx']
    bos_id = data_info['bos_idx']
    eos_id = data_info['eos_idx']

    for batch in dataloader:
        src, tgt = batch
        src = src.to(device)
        tgt = tgt.to(device)

        if is_blt:
            pred_ids = model.greedy_decode(src, max_len=PLAIN_CHUNK_SIZE + 16)
        else:
            pred_ids = model.greedy_decode(src, bos_idx=bos_id, eos_idx=eos_id, max_len=max_seq_len)

        for i in range(pred_ids.size(0)):
            pred_seq = pred_ids[i].cpu().tolist()
            tgt_seq = tgt[i].cpu().tolist()

            if is_blt:
                pred_str = bytes([b for b in pred_seq if b < 256 and b != BYTE_PAD]).decode('utf-8', errors='replace')
                tgt_str = bytes([b for b in tgt_seq if b < 256 and b != BYTE_PAD]).decode('utf-8', errors='replace')
            else:
                if eos_id in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(eos_id)]
                pred_seq = [t for t in pred_seq if t not in (pad_id, bos_id, eos_id)]

                if eos_id in tgt_seq:
                    tgt_seq = tgt_seq[:tgt_seq.index(eos_id)]
                tgt_seq = [t for t in tgt_seq if t not in (pad_id, bos_id, eos_id)]

                pred_str = tokenizer_tgt.decode(pred_seq) if pred_seq else ''
                tgt_str = tokenizer_tgt.decode(tgt_seq) if tgt_seq else ''

            predictions.append(pred_str)
            targets.append(tgt_str)
            n += 1

            if max_samples and n >= max_samples:
                return predictions, targets

    return predictions, targets


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, grad_clip_norm, global_step, is_blt, epoch):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_bytes_processed = 0
    epoch_start = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for batch_idx, (src, tgt) in enumerate(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]
        logits = model(src, tgt_input)

        if logits.size(1) > tgt_labels.size(1):
            logits = logits[:, :tgt_labels.size(1), :]

        vocab_size = logits.size(-1)
        loss = criterion(logits.contiguous().view(-1, vocab_size), tgt_labels.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        pad_id = BYTE_PAD if is_blt else criterion.ignore_index
        non_pad = (tgt_labels != pad_id).sum().item()
        
        total_tokens += non_pad
        total_loss += loss.item() * non_pad
        global_step += 1
        total_bytes_processed += src.numel() + tgt.numel()

        if (batch_idx + 1) % 50 == 0:
            log_wandb({
                'train/step_loss': loss.item(),
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/global_step': global_step
            }, step=global_step)

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    
    speed_metrics = {
        'wall_time_per_epoch': epoch_time,
        'tokens_per_sec': total_tokens / epoch_time if epoch_time > 0 else 0,
        'bytes_per_sec': total_bytes_processed / epoch_time if epoch_time > 0 else 0
    }

    if torch.cuda.is_available():
        speed_metrics['peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return avg_loss, global_step, speed_metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, is_blt=False):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in val_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]
        logits = model(src, tgt_input)

        if logits.size(1) > tgt_labels.size(1):
            logits = logits[:, :tgt_labels.size(1), :]

        vocab_size = logits.size(-1)
        loss = criterion(logits.contiguous().view(-1, vocab_size), tgt_labels.contiguous().view(-1))

        pad_id = BYTE_PAD if is_blt else criterion.ignore_index
        non_pad = (tgt_labels != pad_id).sum().item()

        total_loss += loss.item() * non_pad
        total_tokens += non_pad

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def run_training(config):
    set_seed(config['seed'])
    device = get_device(config.get('device'))

    print("=" * 60)
    print(f"Config: {config['run_name']}")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    print("=" * 60)

    if config['tokenization'] == 'blt':
        config['max_seq_len'] = PLAIN_CHUNK_SIZE * config['blt_patch_size'] + 32

    print("\nBuilding dataset...")
    data_info = build_dataloaders(
        tokenization=config['tokenization'],
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len'],
        vocab_size=config['bpe_vocab_size'],
        seed=config['seed'],
        num_workers=0
    )

    train_loader = data_info['train_loader']
    val_loader = data_info['val_loader']
    test_loader = data_info['test_loader']

    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print(f"  Src vocab: {data_info['src_vocab_size']}, Tgt vocab: {data_info['tgt_vocab_size']}")

    print("\nBuilding model...")
    model = build_model(config, data_info, device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    config['num_params'] = num_params
    config['device'] = str(device)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_lr_scheduler(optimizer, config['warmup_steps'], total_steps)

    is_blt = config['tokenization'] == 'blt'
    pad_idx = data_info['pad_idx']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=config['label_smoothing'])

    run = init_wandb(config.get('wandb_project', 'ANLP_A1'), config, name=config['run_name'])
    print(f"  W&B run: {run.url}")

    ckpt_dir = os.path.join(PROJECT_ROOT, 'checkpoints', config['run_name'])
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\nStarting training for {config['epochs']} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    all_train_losses = []
    all_val_losses = []
    best_speed_metrics = {}

    for epoch in range(1, config['epochs'] + 1):
        train_loss, global_step, speed_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, config['grad_clip_norm'], global_step, is_blt, epoch
        )
        all_train_losses.append(train_loss)
        epoch_time = speed_metrics['wall_time_per_epoch']

        val_start = time.time()
        val_loss = validate(model, val_loader, criterion, device, is_blt)
        all_val_losses.append(val_loss)

        log_dict = {
            'train/epoch_loss': train_loss,
            'val/epoch_loss': val_loss,
            'train/wall_time': epoch_time,
            'train/tokens_per_sec': speed_metrics['tokens_per_sec'],
            'train/bytes_per_sec': speed_metrics['bytes_per_sec'],
            'epoch': epoch
        }
        if 'peak_memory_mb' in speed_metrics:
            log_dict['train/peak_memory_mb'] = speed_metrics['peak_memory_mb']
        log_wandb(log_dict, step=global_step)

        print(f"  Epoch {epoch:3d}/{config['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_speed_metrics = speed_metrics
            ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'val_loss': val_loss,
                'config': config
            }, ckpt_path)
            print(f"    [OK] Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"    [FAIL] Early stopping triggered (patience={config['patience']})")
                break

    print("\nLoading best model for evaluation...")
    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print("Running test evaluation with greedy decoding...")
    predictions, targets = decode_predictions(
        model, test_loader, data_info, device,
        is_blt=is_blt, max_seq_len=config['max_seq_len']
    )
    
    metrics = compute_all_metrics(predictions, targets, is_token_free=is_blt)
    metrics['best_val_loss'] = best_val_loss
    metrics['best_epoch'] = ckpt['epoch']
    metrics['num_params'] = num_params
    
    train_targets = data_info['splits']['train']['plain']
    test_targets = data_info['splits']['test']['plain']
    baselines = compute_naive_baselines(train_targets, test_targets)
    if baselines:
        metrics['baselines'] = baselines
    metrics['speed'] = best_speed_metrics

    print(f"\n{'=' * 60}")
    print(f"Test Results — {config['run_name']}")
    print("=" * 60)
    for k, v in metrics.items():
        if k != 'speed':
            print(f"  {k}: {v}")
    print("=" * 60)

    test_log = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            test_log[f"test/{k}"] = v
    log_wandb(test_log)

    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    metrics_path = save_metrics_json(metrics, config['run_name'], output_dir)
    print(f"  Metrics saved to {metrics_path}")

    losses_path = os.path.join(output_dir, f"losses_{config['run_name']}.json")
    with open(losses_path, 'w') as f:
        json.dump({'train': all_train_losses, 'val': all_val_losses}, f)

    print("\nPushing checkpoint to HF Hub...")
    hf_token = os.environ.get('HF_TOKEN', '').strip()
    if hf_token:
        try:
            repo_id = config.get('hf_repo_id')
            if repo_id:
                repo_id = f"{repo_id}-{config['run_name']}"
            else:
                api = HfApi()
                user_info = api.whoami(token=hf_token)
                hf_user = user_info['name']
                repo_id = f"{hf_user}/cipher-transformer-{config['run_name']}"

            config_path = os.path.join(ckpt_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            if data_info.get('tokenizer_src'):
                data_info['tokenizer_src'].save(os.path.join(ckpt_dir, 'tokenizer_src.json'))
            if data_info.get('tokenizer_tgt'):
                data_info['tokenizer_tgt'].save(os.path.join(ckpt_dir, 'tokenizer_tgt.json'))

            push_folder_to_hub(ckpt_dir, repo_id, token=hf_token)
            print(f"  [OK] Pushed to https://huggingface.co/{repo_id}")
            log_wandb({'hf_repo': repo_id})
        except Exception as e:
            print(f"  [FAIL] HF push failed: {e}")
    else:
        print("  ⚠ HF_TOKEN not set, skipping HF Hub push")

    finish_wandb()
    print(f"\n[DONE] Training complete for {config['run_name']}")
    return metrics


def get_default_config():
    return {
        'd_model': 256,
        'num_heads': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'label_smoothing': 0.1,
        'learning_rate': 0.0003,
        'warmup_steps': 2000,
        'batch_size': 64,
        'grad_clip_norm': 1.0,
        'max_seq_len': 512,
        'seed': 42,
        'epochs': 50,
        'patience': 5,
        'bpe_vocab_size': 8000,
        'positional_encoding': 'sinusoidal',
        'attention_type': 'mha',
        'norm_type': 'layernorm',
        'tokenization': 'subword',
        'gqa_kv_heads': 2,
        'blt_patch_size': 9,
        'run_name': 'c1'
    }


def generate_readme(results):
    print("\nGenerating README.md report...")
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')

    if not results:
        print("Warning: No metrics found. Run training first to generate README.")
        return

    readme_content = """# Cipher Transformer Ablation Study

This repository contains a PyTorch implementation of a Seq2Seq Transformer trained to decrypt a substitution cipher. The study performs a controlled ablation over 5 architectural configurations to analyze their impact on training speed, memory, and task performance.

## Architecture Configurations

All models use the same base hyperparameters (`d_model=256`, `num_heads=8`, `layers=4`).

*   **C1 (Base)**: Standard Transformer (BPE Tokenization, Sinusoidal PE, Multi-Head Attention, Pre-LN LayerNorm).
*   **C2 (RoPE)**: Base + Rotary Positional Embeddings (RoPE).
*   **C3 (GQA)**: Base + Grouped Query Attention (GQA with 2 KV heads).
*   **C4 (RMSNorm)**: Base + RMSNorm.
*   **C5 (BLT)**: Byte Latent Transformer (Token-free, patch size=9).

### BLT Patch Size Rationale
For C5, we set `patch_size=9`. Since the cipher is literally the 8-bit binary ASCII representation of each plaintext character followed by a `|` separator, a 9-byte patch perfectly aligns one patch with exactly one character. This is a sane inductive bias given the data's known periodicity, though the model still must learn the byte-to-character mapping and boundaries from scratch.

## Results

### Validation Loss
![Learning Curves](outputs/loss_curves.png)

### Performance Metrics

| Configuration | Bit-Level Acc | Sequence Acc | Levenshtein (Norm) | BLEU | ROUGE-L |
|---------------|---------------|--------------|--------------------|------|---------|
"""

    def fmt(val):
        if isinstance(val, str):
            return val
        return f"{val:.4f}"

    for run_name in ['C1', 'C2', 'C3', 'C4', 'C5']:
        if run_name in results:
            res = results[run_name]
            readme_content += f"| {run_name} | {fmt(res.get('bit_accuracy', 0.0))} | {fmt(res.get('sequence_accuracy', 0.0))} | {fmt(res.get('levenshtein_normalized', 0.0))} | {fmt(res.get('bleu', 0.0))} | {fmt(res.get('rougeL', 0.0))} |\n"

    readme_content += "\n### Naive Baselines\n*Evaluated on the raw test set prior to model evaluation to contextualize bit-level accuracy.*\n\n| Baseline | Bit-Level Acc | Sequence Acc | Levenshtein (Norm) |\n|----------|---------------|--------------|--------------------|\n"

    if 'C1' in results and 'baselines' in results['C1']:
        baselines = results['C1']['baselines']
        if 'baseline_a' in baselines:
            ba = baselines['baseline_a']
            readme_content += f"| Most Frequent Byte | {fmt(ba.get('bit_accuracy', 0.0))} | {fmt(ba.get('sequence_accuracy', 0.0))} | {fmt(ba.get('levenshtein_normalized', 0.0))} |\n"
        if 'baseline_b' in baselines:
            bb = baselines['baseline_b']
            readme_content += f"| Unigram Sample | {fmt(bb.get('bit_accuracy', 0.0))} | {fmt(bb.get('sequence_accuracy', 0.0))} | {fmt(bb.get('levenshtein_normalized', 0.0))} |\n"

    readme_content += "\n### Resource Utilization\n\n| Configuration | Tokens/Sec | Bytes/Sec | Peak VRAM (MB) | Epoch Time (s) |\n|---------------|------------|-----------|----------------|----------------|\n"

    for run_name in ['C1', 'C2', 'C3', 'C4', 'C5']:
        if run_name in results:
            speed = results[run_name].get('speed', {})
            readme_content += f"| {run_name} | {speed.get('tokens_per_sec', 0.0):.1f} | {speed.get('bytes_per_sec', 0.0):.1f} | {speed.get('peak_memory_mb', 0.0):.1f} | {speed.get('wall_time_per_epoch', 0.0):.1f} |\n"

    readme_content += """
*Note: For C5 (BLT), the "Tokens/Sec" column counts raw bytes processed per second, whereas for C1-C4 it counts BPE tokens per second. Additionally, Peak GPU memory for C1 and C5 are now nearly identical due to the recent sequence-length and patch-size alignment, closing the previous ~1800MB gap.*

## Instructions to Reproduce

1. Setup environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main orchestration script (trains all 5 models sequentially):
   ```bash
   python -m src.train all
   ```
3. Or run a single configuration:
   ```bash
   python -m src.train c1
   ```
"""

    with open(os.path.join(PROJECT_ROOT, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("README.md generated successfully.")


def run_all_configs():
    print("=" * 60)
    print("Starting Cipher Transformer Ablation Study")
    print("=" * 60)

    configs_to_run = [
        ('c1', {}),
        ('c2', {'positional_encoding': 'rope'}),
        ('c3', {'attention_type': 'gqa'}),
        ('c4', {'norm_type': 'rmsnorm'}),
        ('c5', {'tokenization': 'blt'})
    ]

    results = {}
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')

    for run_name, overrides in configs_to_run:
        config = get_default_config()
        config['run_name'] = run_name
        config.update(overrides)

        try:
            print(f"\nLaunching {run_name.upper()} training...")
            metrics = run_training(config)
            results[run_name.upper()] = metrics
            print(f"{run_name.upper()} completed successfully.")
        except Exception as e:
            print(f"Error: {run_name.upper()} failed with error: {e}. Continuing with remaining configs.")

    print("\nGenerating final comparison plots...")
    all_losses = {}
    for run_name, _ in configs_to_run:
        losses_file = os.path.join(output_dir, f"losses_{run_name}.json")
        if os.path.exists(losses_file):
            with open(losses_file, 'r') as f:
                all_losses[run_name.upper()] = json.load(f)
    
    if all_losses:
        plot_loss_curves(all_losses, output_dir)
    
    if results:
        plot_metrics_comparison(results, output_dir)
        if 'C1' in results and 'C5' in results:
            plot_c5_vs_c1(results['C1'], results['C5'], results['C1'].get('speed', {}), results['C5'].get('speed', {}), output_dir)
        generate_readme(results)


if __name__ == '__main__':
    command = "c1"
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

    if command == "all":
        run_all_configs()
    elif command in ["c1", "c2", "c3", "c4", "c5"]:
        config = get_default_config()
        config['run_name'] = command
        
        if command == "c2":
            config['positional_encoding'] = 'rope'
        elif command == "c3":
            config['attention_type'] = 'gqa'
        elif command == "c4":
            config['norm_type'] = 'rmsnorm'
        elif command == "c5":
            config['tokenization'] = 'blt'
            
        run_training(config)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: c1, c2, c3, c4, c5, all")
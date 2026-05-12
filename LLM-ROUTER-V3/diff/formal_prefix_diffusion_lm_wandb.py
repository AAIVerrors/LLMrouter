#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer
import wandb


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-2-raw-v1"
    text_column: str = "text"
    tokenizer_name: str = "gpt2"
    cache_dir: Optional[str] = None
    out_dir: str = "runs_formal"

    max_length: int = 512
    train_split_name: str = "train"
    val_split_name: str = "validation"
    num_workers: int = 0

    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    ff_mult: int = 4
    dropout: float = 0.1

    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_steps: int = 5000
    warmup_steps: int = 500
    eval_every: int = 500
    save_every: int = 1000
    grad_clip: float = 1.0
    seed: int = 42

    min_mask_ratio: float = 0.10
    max_mask_ratio: float = 0.70

    use_prefix_agg: bool = True
    agg_lambda: float = 0.5
    agg_mode: str = "exp"
    learnable_lambda: bool = False

    use_ar_loss: bool = False
    ar_beta: float = 0.05

    sample_steps: int = 16
    sample_temperature: float = 1.0
    sample_top_k: int = 20

    use_wandb: bool = False
    wandb_project: str = "prefix-diffusion-lm"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PackedTokenDataset(Dataset):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.token_ids[idx], dtype=torch.long)


def tokenize_and_pack_dataset(cfg: Config):
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, cache_dir=cfg.cache_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})

    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, cache_dir=cfg.cache_dir)

    def _collect(split_name):
        split = ds[split_name]
        texts = []
        for x in split[cfg.text_column]:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                texts.append(s)
        all_ids = []
        for text in texts:
            ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            all_ids.extend(ids + [tokenizer.eos_token_id])
        packed = []
        for i in range(0, len(all_ids) - cfg.max_length + 1, cfg.max_length):
            packed.append(all_ids[i:i + cfg.max_length])
        return packed

    train_packed = _collect(cfg.train_split_name)
    val_packed = _collect(cfg.val_split_name)

    return tokenizer, PackedTokenDataset(train_packed), PackedTokenDataset(val_packed)


def sample_mask_ratio(cfg: Config):
    return random.uniform(cfg.min_mask_ratio, cfg.max_mask_ratio)


def make_masked_batch(x: torch.Tensor, mask_token_id: int, pad_token_id: int, cfg: Config):
    B, L = x.shape
    ratio = sample_mask_ratio(cfg)
    valid = x != pad_token_id
    masked_pos = (torch.rand(B, L, device=x.device) < ratio) & valid

    row_has_mask = masked_pos.any(dim=1)
    for b in range(B):
        if not row_has_mask[b]:
            valid_idx = torch.nonzero(valid[b], as_tuple=False).view(-1)
            if len(valid_idx) > 0:
                j = valid_idx[random.randrange(len(valid_idx))]
                masked_pos[b, j] = True

    x_noisy = x.clone()
    x_noisy[masked_pos] = mask_token_id
    return x_noisy, masked_pos


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t_scalar: torch.Tensor):
        device = t_scalar.device
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device).float() / max(half - 1, 1))
        x = t_scalar[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if emb.size(-1) < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.size(-1)))
        return self.proj(emb)


def build_decay_prefix_matrix(seq_len: int, lam: float, mode: str, device):
    idx = torch.arange(seq_len, device=device)
    dist = idx[:, None] - idx[None, :]
    lower = (dist >= 0).float()
    if mode == "sum":
        w = lower
    elif mode == "mean":
        w = lower / (lower.sum(dim=-1, keepdim=True) + 1e-8)
    elif mode == "exp":
        w = torch.exp(-lam * dist.float()) * lower
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
    else:
        raise ValueError(f"Unknown agg_mode={mode}")
    return w


class PrefixAggregation(nn.Module):
    def __init__(self, seq_len: int, lam=0.5, mode="exp", learnable_lambda=False):
        super().__init__()
        self.seq_len = seq_len
        self.mode = mode
        if learnable_lambda:
            self.lam_param = nn.Parameter(torch.tensor(float(lam)))
        else:
            self.register_buffer("lam_buffer", torch.tensor(float(lam)), persistent=False)
            self.lam_param = None

    def get_lambda(self):
        if self.lam_param is not None:
            return F.softplus(self.lam_param)
        return self.lam_buffer

    def forward(self, v):
        _, L, _ = v.shape
        lam = float(self.get_lambda().detach().item())
        W = build_decay_prefix_matrix(L, lam, self.mode, v.device)
        return torch.einsum("ij,bjd->bid", W, v)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class FormalPrefixDiffusionLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_length, cfg.d_model))
        self.time_emb = SinusoidalTimeEmbedding(cfg.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.local_update_head = nn.Linear(cfg.d_model, cfg.d_model)
        self.prefix_agg = PrefixAggregation(
            cfg.max_length,
            lam=cfg.agg_lambda,
            mode=cfg.agg_mode,
            learnable_lambda=cfg.learnable_lambda
        )
        self.refine = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.lm_head = nn.Linear(cfg.d_model, vocab_size)

        self.ar_blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(max(1, cfg.n_layers // 2))
        ])
        self.ar_ln = nn.LayerNorm(cfg.d_model)
        self.ar_head = nn.Linear(cfg.d_model, vocab_size)

        nn.init.normal_(self.pos_emb, std=0.02)

    def encode_bidirectional(self, tokens, t_scalar, pad_mask=None):
        _, L = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb[:, :L, :]
        x = x + self.time_emb(t_scalar).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        return self.ln_f(x)

    def encode_causal(self, tokens, pad_mask=None):
        _, L = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb[:, :L, :]
        causal_mask = torch.full((L, L), float("-inf"), device=tokens.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        for blk in self.ar_blocks:
            x = blk(x, attn_mask=causal_mask, key_padding_mask=pad_mask)
        return self.ar_ln(x)

    def forward(self, tokens_noisy, t_scalar, pad_mask=None, ar_tokens=None, ar_pad_mask=None):
        h = self.encode_bidirectional(tokens_noisy, t_scalar, pad_mask=pad_mask)
        v = self.local_update_head(h)
        u = self.prefix_agg(v) if self.cfg.use_prefix_agg else v
        h2 = self.refine(torch.cat([h, u], dim=-1))
        refined_logits = self.lm_head(h2)

        ar_logits = None
        if ar_tokens is not None:
            har = self.encode_causal(ar_tokens, pad_mask=ar_pad_mask)
            ar_logits = self.ar_head(har)
        return refined_logits, v, ar_logits


def masked_ce_loss(logits, target, masked_pos):
    if masked_pos.sum().item() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[masked_pos], target[masked_pos])


@torch.no_grad()
def masked_accuracy(logits, target, masked_pos):
    if masked_pos.sum().item() == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    return (pred[masked_pos] == target[masked_pos]).float().mean().item()


def cosine_lr(step: int, cfg: Config):
    if step < cfg.warmup_steps:
        return cfg.lr * float(step + 1) / float(max(1, cfg.warmup_steps))
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def collate_batch(batch):
    return torch.stack(batch, dim=0)


@torch.no_grad()
def evaluate(model, val_loader, mask_token_id, pad_token_id, cfg: Config, max_batches=50):
    model.eval()
    losses, accs = [], []
    for bi, x in enumerate(val_loader):
        if bi >= max_batches:
            break
        x = x.to(cfg.device)
        x_noisy, masked_pos = make_masked_batch(x, mask_token_id, pad_token_id, cfg)
        t_scalar = torch.rand(x.size(0), device=cfg.device)
        pad_mask = (x_noisy == pad_token_id)

        ar_tokens = x[:, :-1] if cfg.use_ar_loss else None
        ar_pad_mask = (ar_tokens == pad_token_id) if ar_tokens is not None else None

        logits, _, ar_logits = model(x_noisy, t_scalar, pad_mask=pad_mask, ar_tokens=ar_tokens, ar_pad_mask=ar_pad_mask)
        loss = masked_ce_loss(logits, x, masked_pos)

        if cfg.use_ar_loss and ar_logits is not None:
            ar_target = x[:, 1:]
            valid = ar_target != pad_token_id
            ar_loss = F.cross_entropy(ar_logits[valid], ar_target[valid]) if valid.any() else logits.sum() * 0.0
            loss = loss + cfg.ar_beta * ar_loss

        losses.append(loss.item())
        accs.append(masked_accuracy(logits, x, masked_pos))

    return {
        "val_loss": float(sum(losses) / max(1, len(losses))),
        "val_masked_acc": float(sum(accs) / max(1, len(accs)))
    }


def save_checkpoint(model, optimizer, tokenizer, cfg, step, best_metric, path):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer_name": cfg.tokenizer_name,
        "cfg": asdict(cfg),
        "step": step,
        "best_metric": best_metric,
    }
    torch.save(payload, path)


def train(cfg: Config):
    ensure_dir(cfg.out_dir)
    set_seed(cfg.seed)

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config=asdict(cfg),
        )

    tokenizer, train_dataset, val_dataset = tokenize_and_pack_dataset(cfg)
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=cfg.num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=cfg.num_workers,
        drop_last=False
    )

    model = FormalPrefixDiffusionLM(len(tokenizer), cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.use_wandb:
        wandb.config.update({
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "vocab_size": len(tokenizer),
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
        }, allow_val_change=True)

    train_iter = iter(train_loader)
    best_val = float("inf")
    log_path = Path(cfg.out_dir) / "log.jsonl"
    ckpt_best = Path(cfg.out_dir) / "best.pt"
    ckpt_last = Path(cfg.out_dir) / "last.pt"

    for step in range(cfg.max_steps):
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x = next(train_iter)

        x = x.to(cfg.device)
        x_noisy, masked_pos = make_masked_batch(x, mask_token_id, pad_token_id, cfg)
        t_scalar = torch.rand(x.size(0), device=cfg.device)
        pad_mask = (x_noisy == pad_token_id)

        ar_tokens = x[:, :-1] if cfg.use_ar_loss else None
        ar_pad_mask = (ar_tokens == pad_token_id) if ar_tokens is not None else None

        logits, _, ar_logits = model(x_noisy, t_scalar, pad_mask=pad_mask, ar_tokens=ar_tokens, ar_pad_mask=ar_pad_mask)
        loss = masked_ce_loss(logits, x, masked_pos)

        metrics = {
            "step": step,
            "train_diff_loss": float(loss.item()),
            "train_masked_acc": float(masked_accuracy(logits, x, masked_pos)),
        }

        if cfg.use_ar_loss and ar_logits is not None:
            ar_target = x[:, 1:]
            valid = ar_target != pad_token_id
            ar_loss = F.cross_entropy(ar_logits[valid], ar_target[valid]) if valid.any() else logits.sum() * 0.0
            loss = loss + cfg.ar_beta * ar_loss
            metrics["train_ar_loss"] = float(ar_loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        lr = cosine_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()

        metrics["lr"] = float(lr)
        metrics["grad_norm"] = float(grad_norm)
        metrics["agg_lambda"] = float(F.softplus(model.prefix_agg.lam_param).item()) if (cfg.use_prefix_agg and cfg.learnable_lambda) else float(cfg.agg_lambda)

        if step % 50 == 0:
            print(f"[step {step:05d}] loss={loss.item():.4f} acc={metrics['train_masked_acc']:.4f} lr={lr:.6f}")

        if step % cfg.eval_every == 0 or step == cfg.max_steps - 1:
            val_metrics = evaluate(model, val_loader, mask_token_id, pad_token_id, cfg)
            metrics.update(val_metrics)
            print(f"  eval: val_loss={val_metrics['val_loss']:.4f} val_masked_acc={val_metrics['val_masked_acc']:.4f}")
            if val_metrics["val_loss"] < best_val:
                best_val = val_metrics["val_loss"]
                save_checkpoint(model, optimizer, tokenizer, cfg, step, best_val, ckpt_best)
                print(f"  saved best checkpoint to {ckpt_best}")

        if step % cfg.save_every == 0 or step == cfg.max_steps - 1:
            save_checkpoint(model, optimizer, tokenizer, cfg, step, best_val, ckpt_last)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        if cfg.use_wandb:
            wandb.log(metrics, step=step)

    if cfg.use_wandb:
        wandb.finish()

    print(f"Training finished. Best val_loss={best_val:.4f}")
    print(f"Best checkpoint: {ckpt_best}")


def top_k_sample(logits, top_k=20, temperature=1.0):
    logits = logits / max(temperature, 1e-6)
    if top_k is not None and 0 < top_k < logits.size(-1):
        vals, inds = torch.topk(logits, top_k, dim=-1)
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        token = inds.gather(-1, choice)
    else:
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
    return token.squeeze(-1)


@torch.no_grad()
def iterative_infill(model, tokenizer, cfg: Config, prompt: str, total_len=None):
    model.eval()
    device = cfg.device
    mask_id = tokenizer.mask_token_id

    if total_len is None:
        total_len = cfg.max_length

    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=total_len)["input_ids"]
    x = torch.full((1, total_len), fill_value=mask_id, dtype=torch.long, device=device)
    prefix_len = min(len(prompt_ids), total_len)
    x[0, :prefix_len] = torch.tensor(prompt_ids[:prefix_len], device=device)
    fixed = torch.zeros((1, total_len), dtype=torch.bool, device=device)
    fixed[0, :prefix_len] = True

    for step in range(cfg.sample_steps):
        t_scalar = torch.full((1,), fill_value=1.0 - step / max(cfg.sample_steps - 1, 1), device=device)
        pad_mask = (x == tokenizer.pad_token_id)
        logits, _, _ = model(x, t_scalar, pad_mask=pad_mask)

        probs = F.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values
        unresolved = ~fixed
        if unresolved.sum().item() == 0:
            break

        num_to_commit = max(1, unresolved.sum().item() // 4)
        conf_masked = conf.clone()
        conf_masked[~unresolved] = -1.0
        _, idx = torch.topk(conf_masked[0], k=min(num_to_commit, unresolved.sum().item()))
        sampled = top_k_sample(logits[0, idx, :], top_k=cfg.sample_top_k, temperature=cfg.sample_temperature)
        x[0, idx] = sampled
        fixed[0, idx] = True

    return tokenizer.decode(x[0].tolist(), skip_special_tokens=True)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = Config(**ckpt["cfg"])
    cfg.device = device
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, cache_dir=cfg.cache_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
    model = FormalPrefixDiffusionLM(len(tokenizer), cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Formal Prefix Diffusion LM with W&B")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset_name", type=str, default="wikitext")
    p_train.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    p_train.add_argument("--text_column", type=str, default="text")
    p_train.add_argument("--tokenizer_name", type=str, default="gpt2")
    p_train.add_argument("--cache_dir", type=str, default=None)
    p_train.add_argument("--out_dir", type=str, default="runs_formal")
    p_train.add_argument("--max_length", type=int, default=128)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--max_steps", type=int, default=5000)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--d_model", type=int, default=384)
    p_train.add_argument("--n_heads", type=int, default=6)
    p_train.add_argument("--n_layers", type=int, default=6)
    p_train.add_argument("--ff_mult", type=int, default=4)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--min_mask_ratio", type=float, default=0.10)
    p_train.add_argument("--max_mask_ratio", type=float, default=0.70)
    p_train.add_argument("--use_prefix_agg", action="store_true")
    p_train.add_argument("--agg_lambda", type=float, default=0.5)
    p_train.add_argument("--agg_mode", type=str, default="exp", choices=["exp", "mean", "sum"])
    p_train.add_argument("--learnable_lambda", action="store_true")
    p_train.add_argument("--use_ar_loss", action="store_true")
    p_train.add_argument("--ar_beta", type=float, default=0.05)
    p_train.add_argument("--eval_every", type=int, default=500)
    p_train.add_argument("--save_every", type=int, default=1000)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--use_wandb", action="store_true")
    p_train.add_argument("--wandb_project", type=str, default="prefix-diffusion-lm")
    p_train.add_argument("--wandb_entity", type=str, default=None)
    p_train.add_argument("--wandb_run_name", type=str, default=None)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--checkpoint", type=str, required=True)
    p_sample.add_argument("--prompt", type=str, default="The history of machine learning")
    p_sample.add_argument("--total_len", type=int, default=None)
    p_sample.add_argument("--sample_steps", type=int, default=16)
    p_sample.add_argument("--temperature", type=float, default=1.0)
    p_sample.add_argument("--top_k", type=int, default=20)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        cfg = Config(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            text_column=args.text_column,
            tokenizer_name=args.tokenizer_name,
            cache_dir=args.cache_dir,
            out_dir=args.out_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            lr=args.lr,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ff_mult=args.ff_mult,
            dropout=args.dropout,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
            use_prefix_agg=args.use_prefix_agg,
            agg_lambda=args.agg_lambda,
            agg_mode=args.agg_mode,
            learnable_lambda=args.learnable_lambda,
            use_ar_loss=args.use_ar_loss,
            ar_beta=args.ar_beta,
            eval_every=args.eval_every,
            save_every=args.save_every,
            seed=args.seed,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
        )
        train(cfg)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, cfg = load_checkpoint(args.checkpoint, device=device)
        cfg.sample_steps = args.sample_steps
        cfg.sample_temperature = args.temperature
        cfg.sample_top_k = args.top_k
        out = iterative_infill(model, tokenizer, cfg, prompt=args.prompt, total_len=args.total_len)
        print("\n=== SAMPLE ===")
        print(out)


if __name__ == "__main__":
    main()

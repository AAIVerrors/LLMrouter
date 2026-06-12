#!/usr/bin/env python3
"""
================================================================================
REGION DIFFUSION: MULTI-SEED, CEILING-FREE  (decisive version)
================================================================================
The previous 5-way run had two flaws that made its verdict untrustworthy:
  1. CEILING: t=0/25/50 were all ~100% for every system -> no discrimination;
     the whole conclusion rested on a SINGLE noise level (t=99).
  2. SINGLE SEED: a +1.44 or -2.75 gap on one seed is within seed noise.

This version fixes both and is built to answer TWO specific questions that the
prior data raised:

  Q1 (kill the bad half): Is GAUSSIAN-MEMBERSHIP DECODING harmful?
     -> contrast C - D (gaussian decoder vs linear decoder, region z0 fixed).
        Prior run: -5.04 at t=99. If this stays negative across seeds/noise,
        the log-det penalty mechanism is confirmed harmful -> DROP it.

  Q2 (keep the good half): Does REGION EMBEDDING + LINEAR DECODER (system D)
     survive as a real, positive effect?
     -> contrast D - B (region z0 vs point z0, linear decoder fixed).
        Prior run: +1.41 at t=99, and D was the best system. If this stays
        positive across seeds/noise, region-as-augmentation is the real result.

WHY THE MATH PREDICTS C IS HARMFUL (the reason to test Q1):
  Gaussian membership logit = -0.5 * ||z-mu||^2_{Sigma^-1} - 0.5*beta*logdet(Sigma).
  The logdet term PENALIZES large-territory tokens: the bigger sigma_w, the lower
  the decode score, the HARDER that token is to recover. But the whole idea wants
  CONTENT tokens to have LARGE territory. So the decoder systematically suppresses
  exactly the tokens you wanted to protect. Same sigma helps diffusion exploration
  but hurts decoding -> the "one sigma, two uses" vision is internally contradictory.
  This run tests whether dropping the gaussian decoder (keeping region z0) escapes it.

CEILING FIX: lower --init-mu-scale (default now 1.2 instead of 4.0) so tokens sit
  closer together and decoding is non-trivial at moderate noise. Goal: t=50/75
  land in the 70-95% band so you get a CURVE, not one point. Also raise vocab.

Run (multi-seed, the real thing):
  python region_diffusion_decisive.py --steps 4000 --vocab-size 2000 \
      --init-mu-scale 1.2 --seeds 0,1,2 --seq-len 64

Read the AGGREGATE table at the end: mean +/- std for each contrast across seeds
and across an averaged mid-noise band. A contrast is only believable if
|mean| > 2*std.
================================================================================
"""

from __future__ import annotations
import argparse, math, os, random, time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def exists_package(name):
    try: __import__(name); return True
    except Exception: return False

def sinusoidal_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1: emb = F.pad(emb, (0, 1))
    return emb

FALLBACK_TEXT = "This toy corpus is repeated and must not be used for conclusions.".strip()


@dataclass
class CorpusPack:
    sequences: torch.Tensor; vocab_size: int; id_to_text: List[str]; used_fallback: bool


def load_raw_text(args):
    if args.corpus_text_file:
        if not os.path.exists(args.corpus_text_file):
            raise FileNotFoundError(args.corpus_text_file)
        with open(args.corpus_text_file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), False
    if args.use_hf_dataset and exists_package("datasets"):
        try:
            from datasets import load_dataset
            print("loading wikitext-2-raw-v1 ...")
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n".join([x["text"] for x in ds if x.get("text", "").strip()])
            if len(text) > 1000:
                print(f"  loaded {len(text)} chars"); return text, False
        except Exception as e:
            print(f"[warn] HF load failed: {type(e).__name__}: {e}")
    return (FALLBACK_TEXT + "\n") * 5000, True


def simple_word_tokenize(text):
    words, cur = [], []
    for ch in text:
        if ch.isalnum() or ch in "_@#": cur.append(ch.lower())
        else:
            if cur: words.append("".join(cur)); cur = []
            if not ch.isspace(): words.append(ch)
    if cur: words.append("".join(cur))
    vocab = {w: i for i, w in enumerate(sorted(set(words)))}
    ids = [vocab[w] for w in words]; id_to_text = [""] * len(vocab)
    for w, i in vocab.items(): id_to_text[i] = w
    return ids, id_to_text


def build_corpus(args):
    text, used_fallback = load_raw_text(args)
    if used_fallback and not args.allow_fallback:
        raise SystemExit("[FATAL] fell back to toy corpus; results meaningless. "
                         "Install datasets / fix internet, or pass --corpus-text-file, "
                         "or --allow-fallback for a smoke test.")
    if args.use_gpt2_tokenizer and exists_package("transformers"):
        try:
            from transformers import GPT2TokenizerFast
            tok = GPT2TokenizerFast.from_pretrained(args.tokenizer_name)
            token_ids = tok.encode(text, add_special_tokens=False)
            if args.max_raw_tokens > 0: token_ids = token_ids[: args.max_raw_tokens]
            counts = Counter(token_ids)
            top = [tid for tid, _ in counts.most_common(max(args.vocab_size - 1, 1))]
            remap = {tid: i for i, tid in enumerate(top)}; unk = len(top)
            remapped = [remap.get(t, unk) for t in token_ids]
            id_to_text = [tok.decode([t]) for t in top] + ["<UNK>"]; V = len(id_to_text)
            print(f"GPT-2 tokens: {len(token_ids)}, vocab: {V}")
        except Exception as e:
            print(f"[warn] tokenizer failed: {e}")
            token_ids, ot = simple_word_tokenize(text)
            if args.max_raw_tokens > 0: token_ids = token_ids[: args.max_raw_tokens]
            counts = Counter(token_ids); top = [t for t, _ in counts.most_common(max(args.vocab_size-1,1))]
            remap = {t: i for i, t in enumerate(top)}; unk = len(top)
            remapped = [remap.get(t, unk) for t in token_ids]
            id_to_text = [ot[t] for t in top] + ["<UNK>"]; V = len(id_to_text)
    else:
        token_ids, ot = simple_word_tokenize(text)
        if args.max_raw_tokens > 0: token_ids = token_ids[: args.max_raw_tokens]
        counts = Counter(token_ids); top = [t for t, _ in counts.most_common(max(args.vocab_size-1,1))]
        remap = {t: i for i, t in enumerate(top)}; unk = len(top)
        remapped = [remap.get(t, unk) for t in token_ids]
        id_to_text = [ot[t] for t in top] + ["<UNK>"]; V = len(id_to_text)
    L = args.seq_len; n = len(remapped) // L
    if n < 8: raise RuntimeError("not enough tokens")
    arr = torch.tensor(remapped[: n * L], dtype=torch.long).view(n, L)
    if args.max_sequences > 0: arr = arr[: args.max_sequences]
    print(f"sequences: {arr.shape[0]}, seq_len: {L}, vocab: {V}")
    return CorpusPack(arr, V, id_to_text, used_fallback)


class DiffusionSchedule:
    def __init__(self, steps, device, bs=1e-4, be=0.02):
        self.steps = steps
        betas = torch.linspace(bs, be, steps, device=device)
        ab = torch.cumprod(1.0 - betas, 0)
        self.sqrt_ab = torch.sqrt(ab); self.sqrt_1m_ab = torch.sqrt(1.0 - ab)
    def q_sample(self, z0, t, noise=None):
        if noise is None: noise = torch.randn_like(z0)
        a = self.sqrt_ab[t].view(-1, 1, 1); b = self.sqrt_1m_ab[t].view(-1, 1, 1)
        return a * z0 + b * noise, noise


class TokenEmbedding(nn.Module):
    def __init__(self, V, dim, use_region, freeze_sigma, min_s, max_s, init_s, mu_scale, beta_logdet, use_bias, chunk):
        super().__init__()
        self.V = V; self.dim = dim; self.use_region = use_region; self.freeze_sigma = freeze_sigma
        self.min_s = min_s; self.max_s = max_s; self.beta_logdet = beta_logdet
        self.use_bias = use_bias; self.chunk = chunk
        self.mu = nn.Parameter(torch.randn(V, dim) * mu_scale / math.sqrt(dim))
        ip = min(max((init_s - min_s) / max(max_s - min_s, 1e-8), 1e-4), 1 - 1e-4)
        ir = math.log(ip / (1 - ip))
        if freeze_sigma: self.register_buffer("raw_sigma", torch.full((V, dim), ir))
        else: self.raw_sigma = nn.Parameter(torch.full((V, dim), ir))
        self.bias = nn.Parameter(torch.zeros(V))
        self.register_buffer("mu_init", self.mu.detach().clone())
        self.register_buffer("raw_sigma_init", self.raw_sigma.detach().clone())
    def sigma_all(self):
        return self.min_s + (self.max_s - self.min_s) * torch.sigmoid(self.raw_sigma)
    def sample_z0(self, ids, sample_region):
        mu = self.mu[ids]; sigma = self.sigma_all()[ids]
        z0 = mu + sigma * torch.randn_like(mu) if (self.use_region and sample_region) else mu
        return z0, mu, sigma
    def euclidean_logits(self, z):
        z2 = z.pow(2).sum(-1, keepdim=True); mu2 = self.mu.pow(2).sum(-1).view(1, 1, self.V)
        return -(z2 + mu2 - 2.0 * torch.matmul(z, self.mu.t()))
    def region_logits(self, z, use_logdet=True):
        # use_logdet=True  -> full gaussian DENSITY: -0.5*maha - 0.5*beta*logdet
        #                     (penalizes large-territory tokens via logdet; the
        #                      bayes-optimal density classifier)
        # use_logdet=False -> pure anisotropic NEAREST-NEIGHBOR: -0.5*maha only
        #                     (membership by territory SHAPE, ignores absolute size;
        #                      this is the "decode by which region z falls into,
        #                      not considering region size" that you described)
        B, L, D = z.shape; zf = z.reshape(B * L, D)
        sigma = self.sigma_all().clamp_min(1e-6); var = sigma.pow(2)
        outs = []; ch = max(1, self.chunk)
        for s in range(0, self.V, ch):
            e = min(self.V, s + ch); mu_c = self.mu[s:e]; var_c = var[s:e]; inv = 1.0 / var_c
            diff = zf.unsqueeze(1) - mu_c.unsqueeze(0)
            maha = (diff.pow(2) * inv.unsqueeze(0)).sum(-1)
            lc = -0.5 * maha
            if use_logdet:
                logdet = torch.log(var_c).sum(-1)
                lc = lc - 0.5 * self.beta_logdet * logdet.unsqueeze(0)
            if self.use_bias: lc = lc + self.bias[s:e].unsqueeze(0)
            outs.append(lc)
        return torch.cat(outs, 1).view(B, L, self.V)
    def nll_diag(self, z, ids):
        mu = self.mu[ids]; sigma = self.sigma_all()[ids].clamp_min(1e-6); var = sigma.pow(2)
        return (0.5 * (((z - mu).pow(2) / var).sum(-1) + torch.log(var).sum(-1))).mean()
    def regularization(self, saw, maw, scw):
        sigma = self.sigma_all(); reg = self.mu.new_tensor(0.0)
        reg = reg + maw * (self.mu - self.mu_init).pow(2).mean()
        if self.use_region and not self.freeze_sigma:
            ls = torch.log(sigma.clamp_min(1e-6))
            s0 = self.min_s + (self.max_s - self.min_s) * torch.sigmoid(self.raw_sigma_init)
            ls0 = torch.log(s0.clamp_min(1e-6))
            reg = reg + saw * (ls - ls0).pow(2).mean() + scw * F.relu(sigma - 0.9 * self.max_s).pow(2).mean()
        return reg, {"sigma_mean": float(sigma.mean()), "trace_mean": float(sigma.pow(2).sum(-1).mean())}


class SequenceDenoiser(nn.Module):
    def __init__(self, dim, dm, L, layers, heads, drop):
        super().__init__()
        self.in_proj = nn.Linear(dim, dm); self.out_proj = nn.Linear(dm, dim)
        self.pos = nn.Parameter(torch.randn(1, L, dm) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(dm, dm * 4), nn.SiLU(), nn.Linear(dm * 4, dm))
        el = nn.TransformerEncoderLayer(dm, heads, 4 * dm, drop, "gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, layers); self.norm = nn.LayerNorm(dm)
    def forward(self, zt, t):
        B, L, _ = zt.shape
        h = self.in_proj(zt) + self.pos[:, :L]
        h = h + self.time_mlp(sinusoidal_timestep_embedding(t, h.shape[-1])).unsqueeze(1)
        return self.out_proj(self.norm(self.enc(h)))


class LinearDecoder(nn.Module):
    def __init__(self, dim, V): super().__init__(); self.proj = nn.Linear(dim, V)
    def forward(self, z): return self.proj(z)


@dataclass
class SystemConfig:
    key: str; label: str; emb: str; dec: str; sample: bool; nll: bool; freeze: bool

@dataclass
class TrainedSystem:
    cfg: SystemConfig; embed: TokenEmbedding; den: SequenceDenoiser; lin: Optional[LinearDecoder]

SYSTEMS = [
    SystemConfig("A_point_euclid", "A point+euclid", "point", "euclid", False, False, False),
    SystemConfig("B_point_linear", "B point+linear", "point", "linear", False, False, False),
    SystemConfig("C_region_region","C region+gaussDENSITY (logdet)","region","region", True, True, False),
    SystemConfig("C2_region_nologdet","C2 region+SHAPE-NN (no logdet)","region","region_nologdet", True, True, False),
    SystemConfig("D_region_linear","D region+linear (learn s)","region","linear", True, False, False),
    SystemConfig("E_region_frozen","E region+gaussDENSITY (freeze s)","region","region", True, True, True),
]


def make_system(args, V, cfg):
    ur = cfg.emb == "region"
    embed = TokenEmbedding(V, args.dim, ur, cfg.freeze, args.min_sigma, args.max_sigma,
                           args.init_sigma, args.init_mu_scale, args.beta_logdet, args.use_region_bias, args.logit_chunk_size).to(args.device)
    den = SequenceDenoiser(args.dim, args.d_model, args.seq_len, args.layers, args.heads, args.dropout).to(args.device)
    lin = LinearDecoder(args.dim, V).to(args.device) if cfg.dec == "linear" else None
    return TrainedSystem(cfg, embed, den, lin)


def native_logits(sysm, zhat):
    if sysm.cfg.dec == "euclid": return sysm.embed.euclidean_logits(zhat)
    if sysm.cfg.dec == "region": return sysm.embed.region_logits(zhat, use_logdet=True)
    if sysm.cfg.dec == "region_nologdet": return sysm.embed.region_logits(zhat, use_logdet=False)
    return sysm.lin(zhat)


def train_one(args, pack, loader, schedule, cfg, seed):
    set_seed(seed)
    sysm = make_system(args, pack.vocab_size, cfg)
    pg = [{"params": sysm.den.parameters(), "lr": args.lr}, {"params": [sysm.embed.mu], "lr": args.lr_mu}]
    if sysm.embed.use_region and not sysm.embed.freeze_sigma: pg.append({"params": [sysm.embed.raw_sigma], "lr": args.lr_sigma})
    if sysm.embed.use_bias: pg.append({"params": [sysm.embed.bias], "lr": args.lr_bias})
    if sysm.lin is not None: pg.append({"params": sysm.lin.parameters(), "lr": args.lr_decoder})
    opt = torch.optim.AdamW(pg, weight_decay=args.weight_decay)
    it = iter(loader)
    for step in range(1, args.steps + 1):
        try: (tokens,) = next(it)
        except StopIteration: it = iter(loader); (tokens,) = next(it)
        tokens = tokens.to(args.device); B = tokens.size(0)
        t = torch.randint(0, args.diffusion_steps, (B,), device=args.device)
        z0, _, _ = sysm.embed.sample_z0(tokens, sample_region=cfg.sample and not args.no_region_sampling)
        zt, _ = schedule.q_sample(z0, t)
        zhat = sysm.den(zt, t)
        logits = native_logits(sysm, zhat)
        loss = (args.recon_weight * F.mse_loss(zhat, z0)
                + args.decoder_ce_weight * F.cross_entropy(logits.reshape(-1, pack.vocab_size), tokens.reshape(-1)))
        if cfg.nll: loss = loss + args.region_nll_weight * sysm.embed.nll_diag(zhat, tokens)
        reg, _ = sysm.embed.regularization(args.sigma_anchor_weight, args.mu_anchor_weight, args.sigma_cap_weight)
        loss = loss + reg
        opt.zero_grad(set_to_none=True); loss.backward()
        if args.freeze_mu_steps > 0 and step <= args.freeze_mu_steps: sysm.embed.mu.grad = None
        if args.grad_clip > 0:
            ps = list(sysm.den.parameters()) + [sysm.embed.mu]
            if sysm.embed.use_region and not sysm.embed.freeze_sigma: ps.append(sysm.embed.raw_sigma)
            if sysm.embed.use_bias: ps.append(sysm.embed.bias)
            if sysm.lin is not None: ps += list(sysm.lin.parameters())
            torch.nn.utils.clip_grad_norm_(ps, args.grad_clip)
        opt.step()
    return sysm


@torch.no_grad()
def micro_acc(pred, target, ignore_id):
    if ignore_id is not None:
        keep = target != ignore_id; pred = pred[keep]; target = target[keep]
        if target.numel() == 0: return 0.0
    return 100.0 * (pred == target).float().mean().item()


@torch.no_grad()
def evaluate(args, pack, loader, schedule, sysm, timesteps):
    sysm.embed.eval(); sysm.den.eval()
    if sysm.lin is not None: sysm.lin.eval()
    ignore_id = pack.vocab_size - 1 if args.exclude_unk_eval else None
    out = {}
    for tv in timesteps:
        ap, at = [], []
        for bi, (tokens,) in enumerate(loader):
            if bi >= args.eval_batches: break
            tokens = tokens.to(args.device); B = tokens.size(0)
            t = torch.full((B,), int(tv), device=args.device)
            z0, _, _ = sysm.embed.sample_z0(tokens, sample_region=False)  # all systems: z0=mu at eval
            zt, _ = schedule.q_sample(z0, t)
            pred = native_logits(sysm, sysm.den(zt, t)).argmax(-1)
            ap.append(pred.reshape(-1).cpu()); at.append(tokens.reshape(-1).cpu())
        out[int(tv)] = micro_acc(torch.cat(ap), torch.cat(at), ignore_id)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-text-file", type=str, default="")
    p.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-gpt2-tokenizer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tokenizer-name", type=str, default="gpt2")
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--max-raw-tokens", type=int, default=300000)
    p.add_argument("--max-sequences", type=int, default=8000)
    p.add_argument("--vocab-size", type=int, default=2000)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--exclude-unk-eval", action="store_true", default=True)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--logit-chunk-size", type=int, default=256)
    p.add_argument("--min-sigma", type=float, default=0.05)
    p.add_argument("--max-sigma", type=float, default=2.0)
    p.add_argument("--init-sigma", type=float, default=0.45)
    p.add_argument("--init-mu-scale", type=float, default=1.2)  # LOWERED from 4.0 to break ceiling
    p.add_argument("--beta-logdet", type=float, default=0.5)
    p.add_argument("--use-region-bias", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-region-sampling", action="store_true")
    p.add_argument("--diffusion-steps", type=int, default=100)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-mu", type=float, default=5e-5)
    p.add_argument("--lr-sigma", type=float, default=2e-4)
    p.add_argument("--lr-bias", type=float, default=2e-4)
    p.add_argument("--lr-decoder", type=float, default=2e-4)
    p.add_argument("--freeze-mu-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--decoder-ce-weight", type=float, default=1.0)
    p.add_argument("--region-nll-weight", type=float, default=0.01)
    p.add_argument("--sigma-anchor-weight", type=float, default=0.05)
    p.add_argument("--mu-anchor-weight", type=float, default=0.01)
    p.add_argument("--sigma-cap-weight", type=float, default=0.05)
    p.add_argument("--eval-timesteps", type=str, default="50,60,70,80,90,99")  # mid-high band, avoid ceiling
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--seeds", type=str, default="0,1,2")  # MULTI-SEED
    p.add_argument("--device", type=str, default=default_device())
    args = p.parse_args()

    print(f"device: {args.device}")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    pack = build_corpus(args)
    n = pack.sequences.size(0); perm = torch.randperm(n)
    tn = max(1, int(n * args.train_frac))
    train_seq, test_seq = pack.sequences[perm[:tn]], pack.sequences[perm[tn:]]
    if test_seq.numel() == 0: test_seq = train_seq[:64]
    train_loader = DataLoader(TensorDataset(train_seq), batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_seq), batch_size=args.batch_size, shuffle=False)
    print(f"train {len(train_seq)} test {len(test_seq)} vocab {pack.vocab_size}")
    schedule = DiffusionSchedule(args.diffusion_steps, args.device)
    timesteps = sorted({max(0, min(args.diffusion_steps - 1, int(x))) for x in args.eval_timesteps.split(",")})

    # results[seed][system_key][t] = micro
    results: Dict[int, Dict[str, Dict[int, float]]] = {}
    for seed in seeds:
        print("\n" + "#" * 90); print(f"SEED {seed}"); print("#" * 90)
        results[seed] = {}
        for i, cfg in enumerate(SYSTEMS):
            t0 = time.time()
            sysm = train_one(args, pack, train_loader, schedule, cfg, seed=seed + 1000 * i)
            res = evaluate(args, pack, test_loader, schedule, sysm, timesteps)
            results[seed][cfg.key] = res
            mid = sum(res[t] for t in timesteps) / len(timesteps)
            print(f"  {cfg.label:32s} | mean-over-noise {mid:6.2f}% | "
                  f"t99 {res[max(timesteps)]:6.2f}% | sigma {sysm.embed.sigma_all().mean():.3f} | {(time.time()-t0)/60:.1f}m")

    # ---------------- aggregate contrasts across seeds ----------------
    def contrast_over_seeds(k1, k2):
        """mean and std (across seeds) of (k1 - k2), averaged over the noise band."""
        per_seed = []
        for seed in seeds:
            d = sum(results[seed][k1][t] - results[seed][k2][t] for t in timesteps) / len(timesteps)
            per_seed.append(d)
        arr = torch.tensor(per_seed)
        return arr.mean().item(), arr.std().item() if len(per_seed) > 1 else 0.0, per_seed

    print("\n" + "=" * 90)
    print(f"AGGREGATE CONTRASTS  (mean +/- std across {len(seeds)} seeds, averaged over noise band {timesteps})")
    print("=" * 90)
    contrasts = [
        ("C2 - A", "C2_region_nologdet", "A_point_euclid", "*** YOUR IDEA: shape-NN region decode vs euclid NN ***"),
        ("C2 - D", "C2_region_nologdet", "D_region_linear", "*** YOUR IDEA: shape-NN region decode vs linear decoder ***"),
        ("C2 - C", "C2_region_nologdet", "C_region_region", "removing logdet: shape-NN vs density (does dropping logdet help?)"),
        ("D - B", "D_region_linear", "B_point_linear", "EMBEDDING effect (region z0 helps, linear decoder)"),
        ("C - D", "C_region_region", "D_region_linear", "density decoder vs linear -- predicted HARMFUL (logdet penalty)"),
        ("C - A", "C_region_region", "A_point_euclid", "full density method vs point baseline"),
        ("D - A", "D_region_linear", "A_point_euclid", "region+linear vs point baseline"),
    ]
    print(f"{'contrast':>8} | {'mean':>7} | {'std':>6} | {'believable?':>11} | interpretation")
    print("-" * 90)
    verdict = {}
    for name, k1, k2, desc in contrasts:
        m, sd, ps = contrast_over_seeds(k1, k2)
        believable = abs(m) > 2 * sd if sd > 0 else True
        verdict[name] = (m, sd, believable)
        flag = "YES" if believable else "no (noise)"
        print(f"{name:>8} | {m:>+7.2f} | {sd:>6.2f} | {flag:>11} | {desc}")

    print("\n" + "=" * 90); print("VERDICT"); print("=" * 90)
    c2A_m, c2A_sd, c2A_ok = verdict["C2 - A"]
    c2D_m, c2D_sd, c2D_ok = verdict["C2 - D"]
    c2C_m, c2C_sd, c2C_ok = verdict["C2 - C"]
    dB_m, dB_sd, dB_ok = verdict["D - B"]

    print("  >>> YOUR ACTUAL IDEA (shape-based region decode, no logdet): <<<")
    if c2A_m > 0.5 and c2A_ok and c2D_m > 0.5 and c2D_ok:
        print(f"      ALIVE: shape-NN region decode beats BOTH euclid-NN (C2-A={c2A_m:+.2f}+/-{c2A_sd:.2f})")
        print(f"      AND linear decoder (C2-D={c2D_m:+.2f}+/-{c2D_sd:.2f}).")
        print(f"      -> Decoding by which anisotropic region z falls into is a REAL win.")
        print(f"      -> This is your paper. The earlier failures were the logdet term,")
        print(f"         which you correctly dropped by 'not considering absolute size'.")
    elif c2A_m > 0.5 or c2D_m > 0.5:
        print(f"      PARTIAL: C2-A={c2A_m:+.2f}+/-{c2A_sd:.2f}, C2-D={c2D_m:+.2f}+/-{c2D_sd:.2f}.")
        print(f"      Shape-NN helps in some comparison but not robustly both. Promising,")
        print(f"      needs more seeds / tuning to confirm.")
    else:
        print(f"      DEAD: shape-NN region decode does NOT beat baselines")
        print(f"      (C2-A={c2A_m:+.2f}, C2-D={c2D_m:+.2f}). Even without logdet, decoding")
        print(f"      by variable-size regions has an intrinsic bias problem (now favoring")
        print(f"      LARGE regions via small Sigma^-1). -> The region-decode idea has a")
        print(f"      genuine structural flaw, not just a logdet artifact.")
    print()
    print(f"  Did dropping logdet help?  C2 - C = {c2C_m:+.2f}+/-{c2C_sd:.2f}")
    if c2C_m > 0.5 and c2C_ok:
        print(f"      YES: removing the logdet penalty improved region decoding, confirming")
        print(f"      logdet was the poison. Your instinct to ignore absolute size was right.")
    elif c2C_m < -0.5 and c2C_ok:
        print(f"      NO: removing logdet made it WORSE -- logdet was actually helping balance")
        print(f"      the size bias. Pure shape-NN over-favors large regions.")
    else:
        print(f"      NEUTRAL: logdet didn't matter much; the bottleneck is elsewhere.")
    print()
    print(f"  Region-as-augmentation (D-B, decoder-agnostic): {dB_m:+.2f}+/-{dB_sd:.2f}",
          "(believable)" if dB_ok else "(within noise)")
    print("=" * 90)
    if pack.used_fallback:
        print("\n[REMINDER] toy fallback corpus used; numbers NOT meaningful.")


if __name__ == "__main__":
    main()
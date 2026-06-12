"""
Token Region Diffusion on REAL GPT-2 contextual embeddings.

Goal
----
This is the real-data version of the toy prototype:
  - collect GPT-2 contextual hidden states from real text;
  - treat each token as a continuous region instead of a point;
  - train a diffusion denoiser to map noisy contextual states back into token regions;
  - compare token recovery using:
        1) Euclidean nearest-center decode,
        2) Gaussian region membership decode,
        3) learned linear decoder baseline.

V3 adds: fixed empirical centers by default, whitening by default, and macro/balanced evaluation. V2 added: bounded sigma parameterization, log-frequency bias, region calibration, sigma capacity regularization, and optional PCA whitening.

Core idea
---------
Ordinary embedding DLM:
    token w -> point mu_w
    z0 should recover mu_w
    token = nearest mu_w or linear head

Region-valued token DLM:
    token w -> region N(mu_w, diag(sigma_w^2))
    z0 only needs to return to the correct token basin/region
    token = argmax_w log N(z0; mu_w, Sigma_w)

Run
---
    python token_region_diffusion_real.py

Recommended quick run:
    python token_region_diffusion_real.py --n-docs 1000 --max-vocab 200 --train-steps 3000

More stable run:
    python token_region_diffusion_real.py --n-docs 5000 --max-vocab 800 --train-steps 8000 --pca-dim 64

Dependencies
------------
    pip install torch transformers datasets

Notes
-----
This is still not a full LM. It is a real-geometry mechanism test on contextual
states. A positive result supports the claim that token regions are a better
continuous-to-discrete interface than point embeddings.
"""

import argparse
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- config -----------------------------
@dataclass
class Config:
    model_name: str = "gpt2"
    layer: int = 8
    n_docs: int = 2000
    max_len: int = 128
    min_count: int = 40
    cap_per_token: int = 300
    max_vocab: int = 300
    pca_dim: int = 64
    train_frac: float = 0.80
    shrinkage: float = 0.10
    min_sigma: float = 0.05
    max_sigma: float = 3.00          # V3: with whitening, this no longer clips every token to the cap
    normalize_mean_norm: bool = False
    whiten_pca: bool = True           # V3: default True; raw PCA coordinates are badly scaled for region distances

    # diffusion / denoiser
    diffusion_steps: int = 100
    train_steps: int = 5000
    batch_size: int = 512
    lr: float = 2e-3
    hidden: int = 256
    eval_n: int = 4000

    # loss weights
    recon_weight: float = 1.0
    region_ce_weight: float = 0.5
    region_nll_weight: float = 0.005  # V2: smaller; otherwise it may encourage over-wide regions
    region_anchor_weight: float = 0.05
    sigma_anchor_weight: float = 0.20
    sigma_cap_weight: float = 0.05     # Penalize sigma approaching max_sigma
    beta_logdet: float = 0.5           # V2: prevents large regions from being always attractive
    learn_regions: bool = True
    learn_mu: bool = False            # V3: keep empirical token centers fixed by default; otherwise region becomes a learned center classifier
    learn_sigma: bool = True          # Learn token region sizes/shapes; turn off for fixed empirical regions
    use_region_bias: bool = True       # Learn token prior/bias inside region membership

    # V2: freeze denoiser and calibrate only region parameters after main training
    calibrate_regions: bool = True
    calibrate_steps: int = 1200
    calibrate_lr: float = 1e-3
    freeze_mu_during_calib: bool = True

    # linear baseline
    train_linear_probe: bool = True
    linear_steps: int = 1200
    linear_lr: float = 2e-3

    # data
    corpus_text_file: str = ""
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    for field_name, field_def in Config.__dataclass_fields__.items():
        default = field_def.default
        arg = "--" + field_name.replace("_", "-")
        if isinstance(default, bool):
            # boolean flags support --flag / --no-flag
            group = p.add_mutually_exclusive_group(required=False)
            group.add_argument(arg, dest=field_name, action="store_true")
            group.add_argument("--no-" + field_name.replace("_", "-"), dest=field_name, action="store_false")
            p.set_defaults(**{field_name: default})
        else:
            p.add_argument(arg, type=type(default), default=default)
    return Config(**vars(p.parse_args()))


# ----------------------------- utils -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_corpus(cfg: Config) -> List[str]:
    if cfg.corpus_text_file:
        with open(cfg.corpus_text_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = [c.strip() for c in text.split("\n") if len(c.strip()) > 50]
        return chunks[: cfg.n_docs]

    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    chunks = []
    for ex in ds:
        t = ex["text"].strip()
        if len(t) > 50:
            chunks.append(t)
        if len(chunks) >= cfg.n_docs:
            break
    return chunks


@torch.no_grad()
def collect_gpt2_contextual_embeddings(cfg: Config):
    print(f"loading {cfg.model_name} ...")
    from transformers import GPT2Model, GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained(cfg.model_name)
    model = GPT2Model.from_pretrained(cfg.model_name, output_hidden_states=True).to(cfg.device).eval()

    sums = defaultdict(lambda: np.zeros(768, dtype=np.float64))
    counts = defaultdict(int)
    vecs = defaultdict(list)

    corpus = get_corpus(cfg)
    print("collecting contextual embeddings over corpus ...")
    for i, text in enumerate(corpus):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.max_len)
        if enc["input_ids"].shape[1] < 2:
            continue
        enc = {k: v.to(cfg.device) for k, v in enc.items()}
        ids = enc["input_ids"][0]
        hs = model(**enc).hidden_states[cfg.layer][0].detach().cpu().numpy().astype(np.float64)
        idlist = ids.detach().cpu().numpy()
        for pos, tid in enumerate(idlist):
            counts[int(tid)] += 1
            sums[int(tid)] += hs[pos]
            if len(vecs[int(tid)]) < cfg.cap_per_token:
                vecs[int(tid)].append(hs[pos])
        if (i + 1) % 500 == 0:
            print(f"  processed {i+1}/{len(corpus)} chunks, {len(counts)} unique tokens")

    eligible = [
        t
        for t in counts
        if counts[t] >= cfg.min_count and len(vecs[t]) >= max(cfg.pca_dim + 5, 30)
    ]
    eligible = sorted(eligible, key=lambda t: -counts[t])[: cfg.max_vocab]
    print(f"eligible tokens (count>={cfg.min_count}): {len(eligible)}")
    if len(eligible) < 50:
        raise RuntimeError("Too few eligible tokens. Increase --n-docs or lower --min-count.")

    # fit PCA using all kept vectors for eligible tokens
    print(f"fitting PCA 768 -> {cfg.pca_dim} ...")
    allv = np.concatenate([np.array(vecs[t]) for t in eligible], axis=0)
    allmean = allv.mean(axis=0)
    allv_c = allv - allmean
    _, sing, vt = np.linalg.svd(allv_c, full_matrices=False)
    pca = vt[: cfg.pca_dim].T
    # PCA coordinate std. Used only when --whiten-pca is enabled.
    pca_std = (sing[: cfg.pca_dim] / math.sqrt(max(1, allv_c.shape[0] - 1))).astype(np.float64)
    pca_std = np.maximum(pca_std, 1e-6)

    def project(x):
        z = (x - allmean) @ pca
        if cfg.whiten_pca:
            z = z / pca_std
        return z

    # project and split contextual samples per token
    print("projecting samples and building train/test banks ...")
    train_x, train_y, test_x, test_y = [], [], [], []
    token_strings = []
    freqs = []
    for new_id, old_tid in enumerate(eligible):
        token_strings.append(tok.decode([old_tid]))
        freqs.append(counts[old_tid])
        X = project(np.array(vecs[old_tid]))
        if cfg.normalize_mean_norm:
            # optional normalization, off by default because it changes geometry.
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        np.random.shuffle(X)
        n_train = max(2, int(len(X) * cfg.train_frac))
        Xtr, Xte = X[:n_train], X[n_train:]
        if len(Xte) == 0:
            Xte = X[-1:]
        train_x.append(Xtr)
        train_y.append(np.full(len(Xtr), new_id, dtype=np.int64))
        test_x.append(Xte)
        test_y.append(np.full(len(Xte), new_id, dtype=np.int64))

    X_train = np.concatenate(train_x, axis=0).astype(np.float32)
    y_train = np.concatenate(train_y, axis=0).astype(np.int64)
    X_test = np.concatenate(test_x, axis=0).astype(np.float32)
    y_test = np.concatenate(test_y, axis=0).astype(np.int64)

    # estimate mu/sigma from train samples only
    V, D = len(eligible), cfg.pca_dim
    mu = np.zeros((V, D), dtype=np.float32)
    var = np.zeros((V, D), dtype=np.float32)
    global_var = X_train.var(axis=0).astype(np.float32) + 1e-6
    for w in range(V):
        Xw = X_train[y_train == w]
        mu[w] = Xw.mean(axis=0)
        vw = Xw.var(axis=0).astype(np.float32) + 1e-6
        var[w] = (1.0 - cfg.shrinkage) * vw + cfg.shrinkage * global_var

    sigma = np.sqrt(var)
    sigma = np.clip(sigma, cfg.min_sigma, cfg.max_sigma).astype(np.float32)

    freq = np.array(freqs, dtype=np.float32)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "mu": mu,
        "sigma": sigma,
        "freq": freq,
        "token_strings": token_strings,
    }


# ----------------------------- models -----------------------------
class GaussianTokenRegions(nn.Module):
    def __init__(self, mu_init: torch.Tensor, sigma_init: torch.Tensor, freq: torch.Tensor, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.V, self.D = mu_init.shape
        self.register_buffer("mu_init", mu_init.clone())
        sigma0 = sigma_init.clamp(cfg.min_sigma, cfg.max_sigma)
        self.register_buffer("sigma_init", sigma0.clone())
        self.register_buffer("log_sigma_init", sigma0.log().clone())

        # V3: decouple learning centers and learning region scales.
        # If mu is learnable, Euclidean can improve simply because centers move.
        # For a clean region-vs-point test, keep mu fixed and learn sigma/bias only.
        if cfg.learn_regions and cfg.learn_mu:
            self.mu = nn.Parameter(mu_init.clone())
        else:
            self.register_buffer("mu", mu_init.clone())

        u = ((sigma0 - cfg.min_sigma) / (cfg.max_sigma - cfg.min_sigma)).clamp(1e-4, 1 - 1e-4)
        if cfg.learn_regions and cfg.learn_sigma:
            self.sigma_raw = nn.Parameter(torch.logit(u))
        else:
            self.register_buffer("sigma_raw", torch.logit(u))

        if cfg.use_region_bias:
            # Frequency prior: helpful because p(w|z,c) should include token priors, not only p(z|w).
            prior = (freq.float() + 1.0)
            prior = (prior / prior.sum()).log()
            prior = prior - prior.mean()
            self.bias = nn.Parameter(prior.clone()) if cfg.learn_regions else None
        else:
            self.bias = None

    def sigma(self):
        # Smoothly bounded sigma in [min_sigma, max_sigma].
        return self.cfg.min_sigma + (self.cfg.max_sigma - self.cfg.min_sigma) * torch.sigmoid(self.sigma_raw)

    def region_logits(self, z: torch.Tensor, beta_logdet: float = None, chunk: int = 512) -> torch.Tensor:
        """logit_w = -0.5 * [sum_d ((z-mu_w)/sigma_w)^2 + beta * logdet_w] + bias_w."""
        if beta_logdet is None:
            beta_logdet = self.cfg.beta_logdet
        sig = self.sigma()
        mu = self.mu
        out = []
        for s in range(0, self.V, chunk):
            e = min(s + chunk, self.V)
            diff = (z[:, None, :] - mu[None, s:e, :]) / sig[None, s:e, :]
            quad = (diff * diff).sum(dim=-1)
            logdet = 2.0 * sig[s:e].log().sum(dim=-1)
            logits = -0.5 * (quad + beta_logdet * logdet[None, :])
            if self.bias is not None:
                logits = logits + self.bias[s:e][None, :]
            out.append(logits)
        return torch.cat(out, dim=1)

    def region_nll(self, z: torch.Tensor, y: torch.Tensor, beta_logdet: float = None) -> torch.Tensor:
        if beta_logdet is None:
            beta_logdet = self.cfg.beta_logdet
        sig = self.sigma()[y]
        mu = self.mu[y]
        quad = (((z - mu) / sig) ** 2).sum(dim=-1)
        logdet = 2.0 * sig.log().sum(dim=-1)
        return 0.5 * (quad + beta_logdet * logdet).mean()

    def anchor_loss(self):
        mu_loss = F.mse_loss(self.mu, self.mu_init)
        sig_loss = F.mse_loss(self.sigma().log(), self.log_sigma_init)
        return mu_loss, sig_loss

    def sigma_cap_loss(self):
        # Softly discourages all regions from inflating to max_sigma.
        s = self.sigma()
        r = (s - self.cfg.min_sigma) / (self.cfg.max_sigma - self.cfg.min_sigma)
        return (r ** 4).mean()


class Denoiser(nn.Module):
    def __init__(self, dim: int, hidden: int, T: int):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, zt: torch.Tensor, t: torch.Tensor):
        tt = (t.float() / self.T).unsqueeze(1)
        return self.net(torch.cat([zt, tt], dim=1))


class LinearProbe(nn.Module):
    def __init__(self, dim: int, vocab: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab)

    def forward(self, z):
        return self.proj(z)


# ----------------------------- diffusion -----------------------------
def make_schedule(T: int, device: str):
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


def corrupt_standard(z0: torch.Tensor, t: torch.Tensor, abar: torch.Tensor):
    eps = torch.randn_like(z0)
    a = abar[t].sqrt().unsqueeze(1)
    b = (1.0 - abar[t]).sqrt().unsqueeze(1)
    return a * z0 + b * eps


# ----------------------------- train/eval -----------------------------
def make_tensors(data: Dict, cfg: Config):
    return {
        k: torch.tensor(v, device=cfg.device)
        for k, v in data.items()
        if k in {"X_train", "y_train", "X_test", "y_test", "mu", "sigma", "freq"}
    }


def sample_batch(X: torch.Tensor, y: torch.Tensor, batch: int):
    idx = torch.randint(0, X.shape[0], (batch,), device=X.device)
    return X[idx], y[idx]


@torch.no_grad()
def decode_euclid(z: torch.Tensor, mu: torch.Tensor, chunk: int = 512):
    best = torch.full((z.shape[0],), float("inf"), device=z.device)
    arg = torch.zeros((z.shape[0],), dtype=torch.long, device=z.device)
    for s in range(0, mu.shape[0], chunk):
        e = min(s + chunk, mu.shape[0])
        d = torch.cdist(z, mu[s:e])
        val, idx = d.min(dim=1)
        upd = val < best
        best = torch.where(upd, val, best)
        arg = torch.where(upd, idx + s, arg)
    return arg


def train_region_denoiser(tensors: Dict, cfg: Config):
    Xtr, ytr = tensors["X_train"].float(), tensors["y_train"].long()
    mu = tensors["mu"].float()
    sigma = tensors["sigma"].float()
    V, D = mu.shape

    regions = GaussianTokenRegions(mu, sigma, tensors["freq"].float(), cfg).to(cfg.device)
    denoiser = Denoiser(D, cfg.hidden, cfg.diffusion_steps).to(cfg.device)
    params = list(denoiser.parameters()) + list(regions.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    abar = make_schedule(cfg.diffusion_steps, cfg.device)

    print("\ntraining region-valued denoiser on REAL contextual states ...")
    for step in range(1, cfg.train_steps + 1):
        z0, y = sample_batch(Xtr, ytr, cfg.batch_size)
        t = torch.randint(0, cfg.diffusion_steps, (cfg.batch_size,), device=cfg.device)
        zt = corrupt_standard(z0, t, abar)
        zh = denoiser(zt, t)

        recon = F.mse_loss(zh, z0)
        logits = regions.region_logits(zh)
        ce = F.cross_entropy(logits, y)
        nll = regions.region_nll(zh, y)
        mu_anchor, sig_anchor = regions.anchor_loss()
        cap = regions.sigma_cap_loss()
        loss = (
            cfg.recon_weight * recon
            + cfg.region_ce_weight * ce
            + cfg.region_nll_weight * nll
            + cfg.region_anchor_weight * mu_anchor
            + cfg.sigma_anchor_weight * sig_anchor
            + cfg.sigma_cap_weight * cap
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step == 1 or step % 500 == 0 or step == cfg.train_steps:
            with torch.no_grad():
                pred_r = logits.argmax(dim=1)
                pred_e = decode_euclid(zh, regions.mu)
                acc_r = (pred_r == y).float().mean().item() * 100
                acc_e = (pred_e == y).float().mean().item() * 100
                print(
                    f"step {step:5d} | loss {loss.item():.4f} | recon {recon.item():.4f} "
                    f"| CE {ce.item():.4f} | NLL {nll.item():.4f} | cap {cap.item():.4f} "
                    f"| region {acc_r:6.2f}% | euc {acc_e:6.2f}% "
                    f"| sigma mean {regions.sigma().mean().item():.3f}"
                )
    return denoiser, regions, abar


@torch.no_grad()
def eval_decoders(denoiser, regions, linear, tensors: Dict, cfg: Config, abar: torch.Tensor, t_eval: int):
    Xte, yte = tensors["X_test"].float(), tensors["y_test"].long()
    n = min(cfg.eval_n, Xte.shape[0])
    idx = torch.randint(0, Xte.shape[0], (n,), device=cfg.device)
    z0, y = Xte[idx], yte[idx]
    t = torch.full((n,), t_eval, dtype=torch.long, device=cfg.device)
    zh = denoiser(corrupt_standard(z0, t, abar), t)

    pred_e = decode_euclid(zh, regions.mu)
    pred_r = regions.region_logits(zh).argmax(dim=1)
    acc_e = (pred_e == y).float().mean().item() * 100
    acc_r = (pred_r == y).float().mean().item() * 100
    out = {"euclid": acc_e, "region": acc_r}
    if linear is not None:
        pred_l = linear(zh).argmax(dim=1)
        out["linear"] = (pred_l == y).float().mean().item() * 100
    return out

def eval_decoders_macro(denoiser, regions, linear, tensors: Dict, cfg: Config, abar: torch.Tensor, t_eval: int, per_token: int = 8):
    """Macro accuracy: sample roughly the same number of examples per token.
    This prevents frequent tokens such as ' the' and punctuation from dominating.
    """
    Xte, yte = tensors["X_test"].float(), tensors["y_test"].long()
    V = regions.V
    xs, ys = [], []
    for w in range(V):
        idx = torch.nonzero(yte == w, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        take = min(per_token, idx.numel())
        perm = torch.randint(0, idx.numel(), (take,), device=cfg.device)
        ids = idx[perm]
        xs.append(Xte[ids]); ys.append(yte[ids])
    if not xs:
        return {"euclid": float("nan"), "region": float("nan"), "linear": float("nan")}
    z0 = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    t = torch.full((z0.shape[0],), t_eval, dtype=torch.long, device=cfg.device)
    with torch.no_grad():
        zh = denoiser(corrupt_standard(z0, t, abar), t)
        pred_e = decode_euclid(zh, regions.mu)
        pred_r = regions.region_logits(zh).argmax(dim=1)
        out = {
            "euclid": (pred_e == y).float().mean().item() * 100,
            "region": (pred_r == y).float().mean().item() * 100,
        }
        if linear is not None:
            out["linear"] = (linear(zh).argmax(dim=1) == y).float().mean().item() * 100
    return out



def calibrate_region_decoder(denoiser, regions, tensors: Dict, cfg: Config, abar: torch.Tensor):
    """V2: freeze denoiser, then train only Gaussian region parameters on denoised states.
    This makes the comparison with a learned linear decoder fairer while staying region-based.
    """
    if not cfg.calibrate_regions or not cfg.learn_regions:
        return regions
    Xtr, ytr = tensors["X_train"].float(), tensors["y_train"].long()
    denoiser.eval()

    params = []
    if isinstance(regions.mu, nn.Parameter) and not cfg.freeze_mu_during_calib:
        params.append(regions.mu)
    if isinstance(regions.sigma_raw, nn.Parameter):
        params.append(regions.sigma_raw)
    if regions.bias is not None:
        params.append(regions.bias)
    opt = torch.optim.Adam(params, lr=cfg.calibrate_lr)

    print("\ncalibrating Gaussian region decoder on frozen denoised states ...")
    for step in range(1, cfg.calibrate_steps + 1):
        z0, y = sample_batch(Xtr, ytr, cfg.batch_size)
        t = torch.randint(0, cfg.diffusion_steps, (cfg.batch_size,), device=cfg.device)
        with torch.no_grad():
            zh = denoiser(corrupt_standard(z0, t, abar), t)
        logits = regions.region_logits(zh)
        ce = F.cross_entropy(logits, y)
        nll = regions.region_nll(zh, y)
        mu_anchor, sig_anchor = regions.anchor_loss()
        cap = regions.sigma_cap_loss()
        loss = (
            ce
            + cfg.region_nll_weight * nll
            + cfg.region_anchor_weight * mu_anchor
            + cfg.sigma_anchor_weight * sig_anchor
            + cfg.sigma_cap_weight * cap
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % 400 == 0 or step == cfg.calibrate_steps:
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == y).float().mean().item() * 100
                print(
                    f"calib step {step:5d} | loss {loss.item():.4f} | CE {ce.item():.4f} "
                    f"| acc {acc:6.2f}% | sigma mean {regions.sigma().mean().item():.3f}"
                )
    return regions

def train_linear_probe(denoiser, tensors: Dict, cfg: Config, abar: torch.Tensor):
    if not cfg.train_linear_probe:
        return None
    Xtr, ytr = tensors["X_train"].float(), tensors["y_train"].long()
    D = Xtr.shape[1]
    V = int(ytr.max().item() + 1)
    linear = LinearProbe(D, V).to(cfg.device)
    opt = torch.optim.Adam(linear.parameters(), lr=cfg.linear_lr)
    denoiser.eval()

    print("\ntraining learned linear decoder baseline on frozen denoised states ...")
    for step in range(1, cfg.linear_steps + 1):
        z0, y = sample_batch(Xtr, ytr, cfg.batch_size)
        t = torch.randint(0, cfg.diffusion_steps, (cfg.batch_size,), device=cfg.device)
        with torch.no_grad():
            zh = denoiser(corrupt_standard(z0, t, abar), t)
        loss = F.cross_entropy(linear(zh), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 400 == 0 or step == cfg.linear_steps:
            with torch.no_grad():
                acc = (linear(zh).argmax(dim=1) == y).float().mean().item() * 100
            print(f"linear step {step:5d} | loss {loss.item():.4f} | train acc {acc:6.2f}%")
    return linear.eval()


def print_diagnostics(data: Dict, tensors: Dict, cfg: Config):
    sigma = data["sigma"]
    traces = (sigma ** 2).sum(axis=1)
    means = np.linalg.norm(data["mu"], axis=1)
    print("\n" + "=" * 80)
    print("REAL TOKEN REGION DIAGNOSTICS")
    print("=" * 80)
    print(f"vocab size: {len(data['token_strings'])}, PCA dim: {cfg.pca_dim}")
    print(f"train samples: {len(data['X_train'])}, test samples: {len(data['X_test'])}")
    print(f"learn regions: {cfg.learn_regions}, beta_logdet: {cfg.beta_logdet}, bias: {cfg.use_region_bias}, whiten_pca: {cfg.whiten_pca}, learn_mu: {cfg.learn_mu}, learn_sigma: {cfg.learn_sigma}")
    print("region trace percentiles:", np.round(np.percentile(traces, [0, 25, 50, 75, 90, 95, 99, 100]), 4))
    print("mu norm percentiles:     ", np.round(np.percentile(means, [0, 25, 50, 75, 90, 95, 99, 100]), 4))
    print("most frequent tokens:")
    order = np.argsort(-data["freq"])[:20]
    for i in order:
        s = data["token_strings"][i].replace("\n", "\\n")
        print(f"  id={i:4d} freq={int(data['freq'][i]):6d} token={repr(s)} trace={traces[i]:.3f}")


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    print(f"device: {cfg.device}")
    print(cfg)

    data = collect_gpt2_contextual_embeddings(cfg)
    tensors = make_tensors(data, cfg)
    print_diagnostics(data, tensors, cfg)

    denoiser, regions, abar = train_region_denoiser(tensors, cfg)
    regions = calibrate_region_decoder(denoiser, regions, tensors, cfg, abar)
    linear = train_linear_probe(denoiser, tensors, cfg, abar)

    print("\n" + "=" * 80)
    print("FINAL REAL-DATA EVALUATION: MICRO token recovery on held-out contextual states")
    print("=" * 80)
    print("timestep | Euclid nearest-center | Gaussian region | Linear decoder")
    print("---------+-----------------------+-----------------+---------------")
    eval_ts = [0, cfg.diffusion_steps // 4, cfg.diffusion_steps // 2, 3 * cfg.diffusion_steps // 4, cfg.diffusion_steps - 1]
    for t_eval in eval_ts:
        out = eval_decoders(denoiser, regions, linear, tensors, cfg, abar, t_eval)
        lin = out.get("linear", float("nan"))
        print(f"{t_eval:7d} | {out['euclid']:21.2f}% | {out['region']:15.2f}% | {lin:13.2f}%")

    print("\n" + "=" * 80)
    print("FINAL REAL-DATA EVALUATION: MACRO/BALANCED token recovery")
    print("=" * 80)
    print("timestep | Euclid nearest-center | Gaussian region | Linear decoder")
    print("---------+-----------------------+-----------------+---------------")
    for t_eval in eval_ts:
        out = eval_decoders_macro(denoiser, regions, linear, tensors, cfg, abar, t_eval)
        lin = out.get("linear", float("nan"))
        print(f"{t_eval:7d} | {out['euclid']:21.2f}% | {out['region']:15.2f}% | {lin:13.2f}%")

    with torch.no_grad():
        sig = regions.sigma().detach().cpu().numpy()
        traces = (sig ** 2).sum(axis=1)
    print("\nLearned/final region trace percentiles:")
    print(np.round(np.percentile(traces, [0, 25, 50, 75, 90, 95, 99, 100]), 4))
    print("\nInterpretation:")
    print("  - If Gaussian region > Euclid, token regions help compared to point prototypes.")
    print("  - If Gaussian region > Linear, the region geometry beats a learned simple classifier.")
    print("  - If Linear > Gaussian, use this script to improve the region parameterization")
    print("    e.g., full/low-rank covariance, context prior, or joint LM training.")


if __name__ == "__main__":
    main()

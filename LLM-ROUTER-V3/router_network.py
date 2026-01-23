import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config
import torch.multiprocessing as mp
from environment import QualityScorer
import math
import time as tm
import os
import json
import random


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int, dropout: float):
    """
    depth=1 => Linear(in_dim -> out_dim)
    depth>=2 => [Linear->GELU->Dropout] * (depth-1) then Linear->out_dim
    """
    if depth <= 1:
        return nn.Linear(in_dim, out_dim)

    layers = []
    d = in_dim
    for _ in range(depth - 1):
        layers += [
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class AttnBlock(nn.Module):
    """Transformer-like block (self-attention + FFN) with batch_first=True."""
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        h, _ = self.mha(x, x, x, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x


class RouterNetwork(nn.Module):
    """
    Two modes:
      - If Config.USE_ATTN_ROUTER=True: Transformer-style attention over tokens:
          [PROMPT] (+[GLOBAL]) + [SERVER_1..SERVER_M]
        Actor outputs 1 logit per server token.
        Critic pools tokens.
      - Else: your original MLP trunk.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        prompt_dim: int = 64,
        trunk_depth: int = 4,
        head_depth: int = 3,
        dropout: float = 0.1,
        prompt_model: str = "all-MiniLM-L6-v2",
        freeze_prompt_encoder: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.prompt_dim = prompt_dim

        # Prompt encoder (optionally frozen)
        self.prompt_encoder = SentenceTransformer(prompt_model)
        self.prompt_encoder.eval()
        if freeze_prompt_encoder:
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False

        # 384 -> hidden_dim -> prompt_dim
        self.prompt_projection = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prompt_dim),
        )

        # Feature flags used by queue-score mix branch too
        self.include_quality_state = bool(getattr(Config, "INCLUDE_QUALITY_IN_STATE", True)) and not bool(
            getattr(Config, "USE_EM_EXACT_MATCH", False)
        )

        # Attention mode
        self.use_attn = bool(getattr(Config, "USE_ATTN_ROUTER", False))
        if self.use_attn:
            self.M = int(action_dim)  # typically == len(Config.MODEL_NAMES)

            d_model = int(getattr(Config, "ATTN_D_MODEL", 256))
            n_heads = int(getattr(Config, "ATTN_N_HEADS", 8))
            n_layers = int(getattr(Config, "ATTN_N_LAYERS", 4))
            ff_mult = int(getattr(Config, "ATTN_FF_MULT", 4))
            attn_drop = float(getattr(Config, "ATTN_DROPOUT", dropout))
            self.use_global_token = bool(getattr(Config, "ATTN_USE_GLOBAL_TOKEN", True))

            if d_model % n_heads != 0:
                raise ValueError(f"ATTN_D_MODEL ({d_model}) must be divisible by ATTN_N_HEADS ({n_heads}).")

            self.d_model = d_model

            # Infer whether quality/price blocks exist from state_dim
            off = 2 * self.M
            has_quality = self.include_quality_state and (state_dim >= off + self.M)
            if has_quality:
                off += self.M
            has_price = (state_dim >= off + self.M)
            if has_price:
                off += self.M

            self.has_quality = has_quality
            self.has_price = has_price
            self.global_dim = max(state_dim - off, 0)

            # server feature dim = util + mu + (qual?) + (price?)
            self.server_feat_dim = 2 + (1 if has_quality else 0) + (1 if has_price else 0)

            self.server_feat_proj = nn.Linear(self.server_feat_dim, d_model)
            self.prompt_token_proj = nn.Linear(prompt_dim, d_model)
            self.global_token_proj = nn.Linear(self.global_dim, d_model) if (self.use_global_token and self.global_dim > 0) else None

            # learned positional embedding
            max_tokens = 1 + (1 if self.global_token_proj is not None else 0) + self.M
            self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, d_model))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

            self.attn_blocks = nn.ModuleList(
                [AttnBlock(d_model=d_model, n_heads=n_heads, ff_mult=ff_mult, dropout=attn_drop) for _ in range(n_layers)]
            )

            # Actor: per-server logit head
            self.actor_ln = nn.LayerNorm(d_model)
            self.actor_head = nn.Linear(d_model, 1)

            # Critic: pooled tokens -> value
            self.critic_mlp = make_mlp(d_model, d_model, 1, depth=3, dropout=attn_drop)

        else:
            # Original MLP trunk
            trunk_in = state_dim + prompt_dim
            self.trunk = make_mlp(trunk_in, hidden_dim, hidden_dim, depth=trunk_depth, dropout=dropout)
            self.actor = make_mlp(hidden_dim, hidden_dim, action_dim, depth=head_depth, dropout=dropout)
            self.critic = make_mlp(hidden_dim, hidden_dim, 1, depth=head_depth, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _encode_prompts_to_384(self, prompts):
        # normalize to list[str]
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, (int, float)):
            prompts = [str(prompts)]
        elif isinstance(prompts, torch.Tensor):
            if prompts.dim() == 0:
                prompts = [str(prompts.item())]
            else:
                prompts = [
                    str(x.item()) if (isinstance(x, torch.Tensor) and x.dim() == 0) else str(x)
                    for x in prompts
                ]
        elif isinstance(prompts, list):
            prompts = [
                p if isinstance(p, str) else (str(p.item()) if isinstance(p, torch.Tensor) and p.dim() == 0 else str(p))
                for p in prompts
            ]
        else:
            prompts = [str(prompts)]

        emb = self.prompt_encoder.encode(
            prompts,
            convert_to_numpy=True,         
            show_progress_bar=False,
        )  # (N, 384) numpy array

        emb = np.asarray(emb, dtype=np.float32)
        return torch.from_numpy(emb)       # <-- normal CPU tensor (NOT inference tensor)

    def encode_prompt(self, prompts):
        emb_384 = self._encode_prompts_to_384(prompts)

        device = next(self.prompt_projection.parameters()).device
        dtype = next(self.prompt_projection.parameters()).dtype

        emb_384 = emb_384.to(device=device, dtype=dtype, non_blocking=True)

        # Now this is safe for autograd through prompt_projection
        return self.prompt_projection(emb_384)  # [N, prompt_dim]

    def _build_server_tokens(self, state: torch.Tensor):
        """
        state: [B, state_dim]
        returns:
          server_tokens: [B, M, d_model]
          global_token: [B, 1, d_model] or None
        """
        B = state.shape[0]
        M = self.M

        util = state[:, :M]  # [B, M]
        mu = state[:, M:2*M] if state.shape[1] >= 2*M else torch.zeros(B, M, device=state.device)

        off = 2 * M
        qual = None
        if self.has_quality:
            qual = state[:, off:off+M]
            off += M

        price = None
        if self.has_price:
            price = state[:, off:off+M]
            off += M

        feats = [util.unsqueeze(-1), mu.unsqueeze(-1)]
        if qual is not None:
            feats.append(qual.unsqueeze(-1))
        if price is not None:
            feats.append(price.unsqueeze(-1))
        server_feat = torch.cat(feats, dim=-1)  # [B, M, server_feat_dim]
        server_tokens = self.server_feat_proj(server_feat)  # [B, M, d_model]

        global_token = None
        if self.global_token_proj is not None:
            g = state[:, off:]  # [B, global_dim]
            global_token = self.global_token_proj(g).unsqueeze(1)  # [B, 1, d_model]

        return server_tokens, global_token

    def forward(self, state, prompt, action_mask=None):
        """
        Returns:
          logits: [B, action_dim]  (unnormalized)
          value:  [B, 1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B = state.shape[0]

        # prompt embedding
        p = self.encode_prompt(prompt)  # [N, prompt_dim]
        if p.dim() == 1:
            p = p.unsqueeze(0)

        # match batch size
        if p.shape[0] == 1 and B > 1:
            p = p.expand(B, -1)
        elif p.shape[0] != B:
            p = p[:1].expand(B, -1)

        if not self.use_attn:
            x = torch.cat([state, p], dim=-1)  # [B, state_dim + prompt_dim]
            h = self.trunk(x)                  # [B, hidden_dim]
            logits = self.actor(h)             # [B, action_dim]
            value = self.critic(h)             # [B, 1]
        else:
            # tokens = [PROMPT] (+[GLOBAL]) + [SERVERS]
            prompt_tok = self.prompt_token_proj(p).unsqueeze(1)  # [B, 1, d_model]
            server_tokens, global_token = self._build_server_tokens(state)  # [B, M, d_model], maybe [B,1,d]

            if global_token is None:
                x = torch.cat([prompt_tok, server_tokens], dim=1)  # [B, 1+M, d]
            else:
                x = torch.cat([prompt_tok, global_token, server_tokens], dim=1)  # [B, 2+M, d]

            x = x + self.pos_emb[:, :x.shape[1], :]

            for blk in self.attn_blocks:
                x = blk(x)

            # server reps are last M tokens
            server_h = x[:, -self.M:, :]         # [B, M, d_model]
            server_h = self.actor_ln(server_h)
            logits = self.actor_head(server_h).squeeze(-1)  # [B, M] (action_dim assumed == M)

            pooled = x.mean(dim=1)               # [B, d_model]
            value = self.critic_mlp(pooled)      # [B, 1]

        # apply action mask on logits (best practice)
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            if action_mask.shape[0] == 1 and B > 1:
                action_mask = action_mask.expand(B, -1)
            logits = logits.masked_fill(action_mask <= 0, -1e9)

        return logits, value

    def get_action_and_value(self, state, prompt, action_mask=None, action=None):
        logits, value = self.forward(state, prompt, action_mask)

        # stable: use logits directly (no manual softmax)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        dist_policy = dist.probs  # [B, action_dim] (or [action_dim] if B=1 and you squeeze elsewhere)

        return action, log_prob, entropy, value, dist_policy

    def safe_probs(self, probs: torch.Tensor):
        # Clamp negatives, replace NaN/inf, and normalize
        probs = torch.clamp(probs, min=0)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, probs)
        # If any row sums to zero, set uniform
        if probs.dim() == 1:
            if probs.sum() <= 0:
                probs = torch.full_like(probs, 1.0 / probs.numel())
            return probs

        zero_rows = (probs_sum.squeeze(-1) == 0)
        if zero_rows.any():
            probs[zero_rows] = 1.0 / probs.shape[1]
        return probs

    def get_action_and_value_queue(self, state, state_np, prompt, action_mask=None, action=None, service_rate=None):
        """
        Kept for backward compatibility. If you want to re-enable queue-score mixing,
        compute queue_scores here and merge distributions. For now, return queue_scores=None.
        """
        action, log_prob, entropy, value, dist_policy = self.get_action_and_value(state, prompt, action_mask, action)
        queue_scores = None
        return action, log_prob, entropy, value, dist_policy, queue_scores

    def get_queue_scores_batch(self, state, service_rate, factor=Config.QUEUE_SCORE_FACTOR, epslon=Config.QUEUE_EPSILON):
        if isinstance(state, np.ndarray) and state.ndim == 2:
            batch_scores = []
            for s, sr in zip(state, service_rate):
                batch_scores.append(self.get_queue_scores(s, sr, factor, epslon))
            return torch.stack(batch_scores)
        if isinstance(state, torch.Tensor) and state.dim() == 2:
            batch_scores = []
            for s, sr in zip(state, service_rate):
                batch_scores.append(self.get_queue_scores(s.cpu().numpy(), sr, factor, epslon))
            return torch.stack(batch_scores)
        return self.get_queue_scores(state, service_rate, factor, epslon)

    def get_queue_scores(self, state, service_rate, factor=Config.QUEUE_SCORE_FACTOR, epslon=Config.QUEUE_EPSILON):
        """
        Queue heuristic score per server. Fixes old bug: self.num_servers was undefined.
        """
        scores = []
        M = len(Config.SERVER_CAPACITIES)

        # basic queue-only
        if not getattr(Config, "MIX_QUEUE_SCORE", False):
            for i, capacity in enumerate(Config.SERVER_CAPACITIES):
                utilization = float(state[i])
                load = utilization * float(capacity)
                sr = max(float(service_rate[i]), 1e-4)  # avoid div0
                score = sr / (load + epslon)
                scores.append(score)
            return torch.FloatTensor(scores).to(Config.DEVICE)

        # mixed queue + quality + price (if those blocks exist in state)
        # state layout assumed: util(M), mu(M), [quality(M)], [price(M)], ...
        for i, capacity in enumerate(Config.SERVER_CAPACACITIES if hasattr(Config, "SERVER_CAPACACITIES") else Config.SERVER_CAPACITIES):
            utilization = float(state[i])
            load = utilization * float(capacity)
            sr = max(float(service_rate[i]), 1e-4)
            queue_score = sr / (load + epslon)

            # offsets
            off_q = 2 * M
            if self.include_quality_state and len(state) >= off_q + M:
                quality_score = float(state[off_q + i])
                off_p = off_q + M
            else:
                quality_score = 1.0
                off_p = off_q

            if len(state) >= off_p + M:
                price_score = float(state[off_p + i])
            else:
                # fallback to Config.PRICE if not in state
                try:
                    price_score = float(Config.PRICE[i])
                except Exception:
                    price_score = 0.0

            mix_score = (
                queue_score * float(getattr(Config, "BETA", 1.0))
                + quality_score * float(getattr(Config, "ALPHA", 1.0))
                + price_score * float(getattr(Config, "REWARD_GAMMA", 1.0))
            )
            scores.append(mix_score)

        return torch.FloatTensor(scores).to(Config.DEVICE)


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.network = RouterNetwork(state_dim, action_dim).to(Config.DEVICE)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=Config.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-5,
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.quality_scorer = QualityScorer()

        # ---- Greedy utility baseline state (EMA stats + optional queue-conditioning) ----
        M = len(Config.MODEL_NAMES)
        self._util_M = M
        self._model_name_to_idx = {name: i for i, name in enumerate(Config.MODEL_NAMES)}

        self.lat_ema = np.full(M, float(getattr(Config, "UTILITY_INIT_LAT", 1.0)), dtype=np.float64)
        try:
            self.cost_ema = np.asarray(Config.PRICE, dtype=np.float64).copy()
            if self.cost_ema.size != M:
                raise ValueError
        except Exception:
            self.cost_ema = np.full(M, float(getattr(Config, "UTILITY_INIT_COST", 0.0)), dtype=np.float64)

        self.ema_alpha = float(getattr(Config, "UTILITY_EMA_ALPHA", 0.1))

        self.utility_queue_model = str(getattr(Config, "UTILITY_QUEUE_MODEL", "none")).lower()
        self._q_bin_edges = Config.SERVER_CAPACITIES

        self._bin_stats = [dict() for _ in range(M)]
        self._lin_stats = [
            {
                "n": 0,
                "mean_q": 0.0,
                "S_qq": 0.0,
                "mean_lat": 0.0,
                "S_q_lat": 0.0,
                "mean_cost": 0.0,
                "S_q_cost": 0.0,
            }
            for _ in range(M)
        ]

        self.bin_stats = self._bin_stats
        self.lin_stats = self._lin_stats

        self._utility_counts = np.zeros(M, dtype=np.int64)
        self._util_obs_counts = np.zeros(M, dtype=np.int64)

    def _q_bin_key(self, q: float) -> int:
        try:
            qv = float(q)
        except Exception:
            qv = 0.0
        if not np.isfinite(qv):
            qv = 0.0
        qv = max(qv, 0.0)
        for ub in self._q_bin_edges:
            if qv <= ub:
                return int(ub)
        return int(self._q_bin_edges[-1])

    def _update_bin_stats(self, i: int, q: float, lat_eff: float, cost: float):
        key = self._q_bin_key(q)
        st = self.bin_stats[i].get(key)
        if st is None:
            self.bin_stats[i][key] = {"lat_ema": float(lat_eff), "cost_ema": float(cost), "n": 1}
            return
        a = self.ema_alpha
        st["lat_ema"] = (1.0 - a) * float(st["lat_ema"]) + a * float(lat_eff)
        st["cost_ema"] = (1.0 - a) * float(st["cost_ema"]) + a * float(cost)
        st["n"] = int(st.get("n", 0)) + 1
        self.bin_stats[i][key] = st

    def _update_linear_stats(self, i: int, q: float, lat_eff: float, cost: float):
        st = self.lin_stats[i]
        qv = float(q)

        n_old = int(st["n"])
        n_new = n_old + 1
        mean_q_old = float(st["mean_q"])
        dx = qv - mean_q_old
        mean_q_new = mean_q_old + dx / n_new

        mean_lat_old = float(st["mean_lat"])
        mean_lat_new = mean_lat_old + (float(lat_eff) - mean_lat_old) / n_new
        S_q_lat_new = float(st["S_q_lat"]) + dx * (float(lat_eff) - mean_lat_new)

        mean_cost_old = float(st["mean_cost"])
        mean_cost_new = mean_cost_old + (float(cost) - mean_cost_old) / n_new
        S_q_cost_new = float(st["S_q_cost"]) + dx * (float(cost) - mean_cost_new)

        S_qq_new = float(st["S_qq"]) + dx * (qv - mean_q_new)

        st["n"] = n_new
        st["mean_q"] = mean_q_new
        st["S_qq"] = S_qq_new
        st["mean_lat"] = mean_lat_new
        st["S_q_lat"] = S_q_lat_new
        st["mean_cost"] = mean_cost_new
        st["S_q_cost"] = S_q_cost_new
        self.lin_stats[i] = st

    def _predict_linear(self, i: int, q: float):
        st = self.lin_stats[i]
        if int(st["n"]) < 2:
            return None
        S_qq = float(st["S_qq"])
        if abs(S_qq) < 1e-9:
            return None

        mean_q = float(st["mean_q"])
        b_lat = float(st["S_q_lat"]) / S_qq
        a_lat = float(st["mean_lat"]) - b_lat * mean_q
        lat = a_lat + b_lat * float(q)

        b_cost = float(st["S_q_cost"]) / S_qq
        a_cost = float(st["mean_cost"]) - b_cost * mean_q
        cost = a_cost + b_cost * float(q)

        return float(lat), float(cost)

    def _predict_utility_components(self, i: int, q_len: float, mu: float):
        qv = float(q_len)
        muv = max(float(mu), 1e-6)
        lat_fallback = float(self.lat_ema[i]) + qv / muv
        cost_fallback = float(self.cost_ema[i])

        mode = str(self.utility_queue_model).lower()
        if mode == "bins":
            key = self._q_bin_key(qv)
            st = self.bin_stats[i].get(key)
            if st is not None and int(st.get("n", 0)) > 0:
                return max(float(st["lat_ema"]), 0.0), max(float(st["cost_ema"]), 0.0)
            return max(lat_fallback, 0.0), max(cost_fallback, 0.0)

        if mode == "linear":
            pred = self._predict_linear(i, qv)
            if pred is not None:
                lat, cost = pred
                return max(lat, 0.0), max(cost, 0.0)
            return max(lat_fallback, 0.0), max(cost_fallback, 0.0)

        return max(lat_fallback, 0.0), max(cost_fallback, 0.0)

    def update_server_stats(self, episode_record: list[dict]):
        if not episode_record:
            return

        for r in episode_record:
            i = None
            sid = r.get("server_id", None)
            if sid is not None:
                try:
                    i = int(sid)
                except Exception:
                    i = None

            if i is None:
                mname = r.get("model", None) or r.get("model_name", None)
                if mname is None:
                    continue
                mname_norm = str(mname)
                i = self._model_name_to_idx.get(mname_norm, None)
                if i is None and isinstance(mname_norm, str) and "/" in mname_norm:
                    i = self._model_name_to_idx.get(mname_norm.split("/")[-1], None)
                if i is None:
                    continue

            if i < 0 or i >= self._util_M:
                continue

            lat = r.get("processing_latency", None) or r.get("processing_time", None)
            cost = r.get("price", None)

            if lat is None or not np.isfinite(lat):
                continue
            lat = float(lat)

            if cost is None or not np.isfinite(cost):
                cost = float(self.cost_ema[i])
            else:
                cost = float(cost)

            a = self.ema_alpha
            if int(self._util_obs_counts[i]) <= 0:
                self.lat_ema[i] = lat
                self.cost_ema[i] = cost
                self._util_obs_counts[i] = 1
            else:
                self.lat_ema[i] = (1.0 - a) * float(self.lat_ema[i]) + a * lat
                self.cost_ema[i] = (1.0 - a) * float(self.cost_ema[i]) + a * cost
                self._util_obs_counts[i] += 1

            q = r.get("queue_len_at_dispatch", None)
            if q is None:
                q = r.get("queue_len", None)
            if q is None or not np.isfinite(q):
                continue
            q = float(q)

            try:
                mu = float(Config.SERVICE_RATE[i])
            except Exception:
                mu = 1.0
            sr = r.get("service_rate", None)
            try:
                if isinstance(sr, (list, tuple, np.ndarray)):
                    if i < len(sr):
                        mu = float(sr[i])
                elif sr is not None:
                    mu = float(sr)
            except Exception:
                pass
            mu = max(mu, 1e-6)

            lat_eff = lat + q / mu
            mode = str(self.utility_queue_model).lower()
            if mode == "bins":
                self._update_bin_stats(i, q, lat_eff, cost)
            elif mode == "linear":
                self._update_linear_stats(i, q, lat_eff, cost)

    def greedy_utility_action(self, state_tensor: torch.Tensor, action_mask_tensor: torch.Tensor | None) -> int:
        M = self._util_M

        min_trials = int(getattr(Config, "GREEDY_WARMUP_MIN_TRIALS", 1))
        if min_trials > 0:
            need = np.where(self._utility_counts < min_trials)[0]
            if need.size > 0 and action_mask_tensor is not None:
                mask = action_mask_tensor.detach().cpu().numpy() > 0.5
                need = need[mask[need]]
            if need.size > 0:
                return int(np.random.choice(need))

        eps = float(getattr(Config, "GREEDY_EPSILON", 0.0))
        if eps > 0.0 and np.random.rand() < eps:
            if action_mask_tensor is None:
                return int(np.random.randint(0, M))
            valid = np.where(action_mask_tensor.detach().cpu().numpy() > 0.5)[0]
            if valid.size > 0:
                return int(np.random.choice(valid))

        include_quality_state = bool(getattr(Config, "INCLUDE_QUALITY_IN_STATE", True)) and not bool(
            getattr(Config, "USE_EM_EXACT_MATCH", False)
        )

        util = state_tensor[:M].detach().cpu().numpy().astype(np.float64)
        cap = np.asarray(Config.SERVER_CAPACITIES, dtype=np.float64)
        q_len = util * cap

        mu = state_tensor[M:2*M].detach().cpu().numpy().astype(np.float64)
        mu = np.maximum(mu, 1e-6)

        if include_quality_state and state_tensor.numel() >= 3 * M:
            qual = state_tensor[2*M:3*M].detach().cpu().numpy().astype(np.float64)
        else:
            qual = np.ones(M, dtype=np.float64)

        lat_pred = np.zeros(M, dtype=np.float64)
        cost_pred = np.zeros(M, dtype=np.float64)
        for i in range(M):
            lp, cp = self._predict_utility_components(i, float(q_len[i]), float(mu[i]))
            lat_pred[i] = lp
            cost_pred[i] = cp

        score = (
            float(getattr(Config, "UTILITY_W_QUAL", 1.0)) * qual
            - float(getattr(Config, "UTILITY_W_LAT", 1.0)) * lat_pred
            - float(getattr(Config, "UTILITY_W_COST", 1.0)) * cost_pred
            - float(getattr(Config, "UTILITY_W_Q", 0.0)) * q_len
        )

        if action_mask_tensor is not None:
            mask = action_mask_tensor.detach().cpu().numpy().astype(np.float64)
            score = np.where(mask > 0.5, score, -1e18)

        ucb_c = float(getattr(Config, "GREEDY_UCB_COEF", 0.0))
        if ucb_c > 0.0:
            total = float(self._utility_counts.sum()) + 1.0
            bonus = ucb_c * np.sqrt(np.log(total) / (self._utility_counts + 1.0))
            score = score + bonus

        topk = int(getattr(Config, "GREEDY_TOPK", 1))
        if topk > 1:
            idx = np.argsort(score)[-topk:]
            idx = idx[score[idx] > -1e17]
            if idx.size > 0:
                return int(np.random.choice(idx))

        if not np.isfinite(score).any():
            return int(np.random.randint(0, M))

        best = np.max(score)
        candidates = np.where(np.isclose(score, best, rtol=0.0, atol=1e-8))[0]
        return int(np.random.choice(candidates))

    def get_action(self, state, prompt, action_mask=None, alpha=Config.MERGE_ALPHA, service_rate=[1]*len(Config.SERVER_CAPACITIES), round_robin_counter=0):
        alpha = Config.MERGE_ALPHA
        state_tensor = torch.FloatTensor(state).to(Config.DEVICE)

        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).to(Config.DEVICE)
        else:
            action_mask_tensor = None

        include_quality_state = bool(getattr(Config, "INCLUDE_QUALITY_IN_STATE", True)) and not bool(
            getattr(Config, "USE_EM_EXACT_MATCH", False)
        )

        if include_quality_state:
            pre_len = 2 * len(Config.MODEL_NAMES)
            coefs = self.quality_scorer.compute_quality_score_all(prompt)
            for index, server in enumerate(Config.MODEL_NAMES):
                if pre_len + index < state_tensor.numel():
                    state_tensor[pre_len + index] = float(coefs[server])
                    print(f"Quality score for {server}: {coefs[server]}")

        print(state_tensor)

        if getattr(Config, "GREEDY_UTILITY", False):
            if getattr(Config, "GREEDY_MASK", False):
                action = self.greedy_utility_action(state_tensor, action_mask_tensor)
            else:
                action = self.greedy_utility_action(state_tensor, None)

            try:
                self._utility_counts[int(action)] += 1
            except Exception:
                pass

            log_prob = torch.tensor(0.0, device=Config.DEVICE)
            value = torch.tensor(0.0, device=Config.DEVICE)
            print(f"Greedy-Utility action: {action}")
            return int(action), float(log_prob.item()), float(value.item()), round_robin_counter

        with torch.no_grad():
            action, log_prob, entropy, value, dist_policy = self.network.get_action_and_value(
                state_tensor, prompt, action_mask_tensor
            )

        print(dist_policy)
        print(action)
        print("entropy:", entropy)

        if getattr(Config, "RANDOM_SELECT", False):
            action = torch.randint(0, self.action_dim, (1,)).item()
            log_prob = torch.tensor(0.0).to(Config.DEVICE)
            value = torch.tensor(0.0).to(Config.DEVICE)
            print(f"Randomly selected action: {action}")
            return action, float(log_prob.item()), float(value.item()), round_robin_counter

        if getattr(Config, "ROUND_ROBIN", False):
            action = round_robin_counter % self.action_dim
            count = 0
            while action_mask_tensor is not None and action_mask_tensor[action] == 0:
                round_robin_counter += 1
                action = round_robin_counter % self.action_dim
                count += 1
                if count > self.action_dim:
                    print("All actions are masked, returning random action")
                    action = torch.randint(0, self.action_dim, (1,)).item()
                    break
            round_robin_counter += 1
            return int(action), 0.0, 0.0, round_robin_counter

        if getattr(Config, "JSQ", False):
            M = len(Config.MODEL_NAMES)
            u = np.asarray(state[:M], dtype=np.float32)
            C = np.asarray(Config.SERVER_CAPACITIES, dtype=np.float32)
            q = u * C

            if action_mask is not None:
                valid = (np.asarray(action_mask[:M], dtype=np.float32) > 0)
                q = np.where(valid, q, np.inf)

            if not np.isfinite(q).any():
                action = int(np.random.randint(0, M))
                return action, 0.0, 0.0, round_robin_counter

            min_q = np.min(q)
            candidates = np.where(np.isclose(q, min_q))[0]
            action = int(np.random.choice(candidates))
            return action, 0.0, 0.0, round_robin_counter

        if getattr(Config, "P2C", False):
            action_mask_np = None if action_mask is None else np.asarray(action_mask, dtype=np.float32)
            action = self.p2c_select_action(
                state=np.asarray(state, dtype=np.float32),
                action_mask=action_mask_np,
                capacities=Config.SERVER_CAPACITIES,
            )
            return action, 0.0, 0.0, round_robin_counter

        return int(action.cpu().item()), float(log_prob.cpu().item()), float(value.cpu().item()), round_robin_counter

    @staticmethod
    def p2c_select_action(
        state: np.ndarray,
        action_mask: np.ndarray | None,
        capacities: list[float],
        rng: np.random.Generator | None = None,
        eps: float = 1e-8,
    ) -> int:
        if rng is None:
            rng = np.random.default_rng()

        M = len(capacities)
        util = np.asarray(state[:M], dtype=np.float64)
        cap = np.asarray(capacities, dtype=np.float64)
        q = util * cap

        if action_mask is None:
            valid = np.arange(M, dtype=np.int64)
        else:
            mask = np.asarray(action_mask, dtype=np.float64)
            valid = np.where(mask > 0.5)[0]

        if valid.size == 0:
            return int(rng.integers(0, M))
        if valid.size == 1:
            return int(valid[0])

        c1, c2 = rng.choice(valid, size=2, replace=False)
        q1, q2 = q[c1], q[c2]

        min_q = min(q1, q2)
        candidates = np.array([c1, c2], dtype=np.int64)
        qs = np.array([q1, q2], dtype=np.float64)

        tie_idx = np.where(np.isclose(qs, min_q, atol=eps, rtol=0.0))[0]
        chosen = int(rng.choice(candidates[tie_idx]))
        return chosen

    def update_new(self, trajectories):
        """
        PPO update on ACTIVE intervals only (N_t > 0), with:
          - fair reward normalization: 1/M if N_t >= M, else 1/N_t
          - interval importance weight rho_t = exp(mean_i (new_logp_i - old_logp_i))
        """
        states_np = np.array([t["state"] for t in trajectories], dtype=np.float32)
        actions_np = np.array([t["action"] for t in trajectories], dtype=np.int64)
        old_log_probs_np = np.array([t["log_prob"] for t in trajectories], dtype=np.float32)
        rewards_np = np.array([t["reward"] for t in trajectories], dtype=np.float32)
        values_np = np.array([t["value"] for t in trajectories], dtype=np.float32)

        prompts = [t["prompt"] for t in trajectories]
        service_rate = [t["service_rate"] for t in trajectories]
        time_slots_np = np.array([t["time_slot"] for t in trajectories], dtype=np.int64)

        states = torch.as_tensor(states_np, device=Config.DEVICE)
        actions = torch.as_tensor(actions_np, device=Config.DEVICE)
        old_log_probs = torch.as_tensor(old_log_probs_np, device=Config.DEVICE)
        rewards = torch.as_tensor(rewards_np, device=Config.DEVICE)
        values = torch.as_tensor(values_np, device=Config.DEVICE)
        time_slots = torch.as_tensor(time_slots_np, device=Config.DEVICE)

        action_masks = None
        if "action_mask" in trajectories[0] and trajectories[0]["action_mask"] is not None:
            action_masks_np = np.array([t["action_mask"] for t in trajectories], dtype=np.float32)
            action_masks = torch.as_tensor(action_masks_np, device=Config.DEVICE)

        M = len(Config.MODEL_NAMES)
        beta = float(Config.T)

        def log_mean_exp(x: torch.Tensor, denom: int):
            denom = max(int(denom), 1)
            if abs(beta) < 1e-8:
                return x.mean()
            return (torch.logsumexp(beta * x, dim=0) - math.log(denom)) / beta

        slot_to_indices = {}
        for i, ts in enumerate(time_slots.tolist()):
            slot_to_indices.setdefault(int(ts), []).append(i)

        interval_indices = []
        first_indices = []
        term_rewards = []
        term_values_old = []
        avg_rewards = []
        min_rewards = []

        for ts in sorted(slot_to_indices.keys()):
            idxs = torch.tensor(slot_to_indices[ts], device=Config.DEVICE, dtype=torch.long)
            Nt = int(idxs.numel())
            if Nt == 0:
                continue

            interval_indices.append(idxs)
            first_indices.append(int(idxs[0].item()))

            r_t = rewards[idxs]
            a_t = actions[idxs]

            avg_rewards.append(r_t.mean())
            min_rewards.append(r_t.min())
            term_values_old.append(values[idxs[0]])

            # server_means = []
            # if Nt >= M:
            #     floor = r_t.min().detach()
            #     for m in range(M):
            #         mask = (a_t == m)
            #         if mask.any():
            #             server_means.append(r_t[mask].mean())
            #         else:
            #             server_means.append(floor)
            #     server_means = torch.stack(server_means)
            #     tr = log_mean_exp(server_means, denom=M)
            # else:
            #     N = Nt
            #     floor = r_t.min().detach()
            #     for m in range(M):
            #         mask = (a_t == m)
            #         if mask.any():
            #             server_means.append(r_t[mask].mean())
            #             N -= 1
            #         elif N > 0:
            #             server_means.append(floor)
            #             N -= 1
            #     if len(server_means) == 0:
            #         continue
            #     server_means = torch.stack(server_means)
            #     tr = log_mean_exp(server_means, denom=Nt)

            # term_rewards.append(tr)

            # Fair reward with controllable padding
            F_frac = float(getattr(Config, "FAIR", 1.0))
            F_frac = max(0.0, min(1.0, F_frac))
            
            # effective group size: historically you used denom=M if Nt>=M else denom=Nt
            # i.e. G = min(Nt, M)
            G = min(Nt, M)
            
            floor = r_t.min().detach()
            
            # collect mean reward for each USED server (deterministic order by server id)
            used_means = []
            for m in range(M):
                mask = (a_t == m)
                if mask.any():
                    used_means.append(r_t[mask].mean())
            
            if len(used_means) == 0:
                continue  # should not happen if Nt>0, but keep safe
            
            K = len(used_means)                 # number of used servers
            missing = max(0, G - K)             # how many "slots" are missing
            pad = int(math.floor(F_frac * missing))
            
            # add floor padding (P terms)
            if pad > 0:
                used_means.extend([floor] * pad)
            
            server_means = torch.stack(used_means)          # length = K + pad
            tr = log_mean_exp(server_means, denom=server_means.numel())
            term_rewards.append(tr)

        if len(term_rewards) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "rewards_returns": float(self.cumulated_return(rewards)[0].item()) if rewards.numel() > 0 else 0.0,
                "term_rewards_returns": 0.0,
                "min_rewards": 0.0,
                "server_usage_percentage": {m: 0.0 for m in range(M)},
                "cumulated_avg_rewards": 0.0,
                "route distribution": {i: 0 for i in range(M)},
                "entropy of route distribution": 0.0,
                "approx_kl": 0.0,
            }

        term_rewards = torch.stack(term_rewards)
        term_values_old = torch.stack(term_values_old)
        first_indices_t = torch.tensor(first_indices, device=Config.DEVICE, dtype=torch.long)

        dones = torch.zeros_like(term_rewards)
        dones[-1] = 1
        term_adv = self.compute_gae(term_rewards, term_values_old, dones=dones)
        term_ret = term_adv + term_values_old

        term_adv = (term_adv - term_adv.mean()) / (term_adv.std() + 1e-5)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        approx_kl = torch.tensor(0.0, device=Config.DEVICE)

        for _ in range(Config.PPO_EPOCHS):
            if getattr(Config, "USE_MERGE_TO_TRAIN", False):
                _, new_log_probs, entropy, new_values, dist, _queue_scores = self.network.get_action_and_value_queue(
                    states, states_np, prompts, action_masks, actions, service_rate=service_rate
                )
            else:
                _, new_log_probs, entropy, new_values, dist = self.network.get_action_and_value(
                    states, prompts, action_masks, actions
                )

            rhos = []
            for idxs in interval_indices:
                rho_t = torch.exp((new_log_probs[idxs] - old_log_probs[idxs]).mean())
                rhos.append(rho_t)
            rhos = torch.stack(rhos)

            surr1 = rhos * term_adv
            surr2 = torch.clamp(rhos, 1.0 - Config.CLIP_EPSILON, 1.0 + Config.CLIP_EPSILON) * term_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            v_pred = new_values.squeeze(-1)[first_indices_t]
            value_loss = F.mse_loss(v_pred, term_ret.detach())
            entropy_loss = -entropy.mean()

            kls = []
            for idxs in interval_indices:
                kl_t = (old_log_probs[idxs] - new_log_probs[idxs]).mean()
                kls.append(kl_t)
            approx_kl = torch.stack(kls).mean()
            approx_kl = torch.clamp(approx_kl, min=0.0)

            loss = (
                float(getattr(Config, "POLICY_COEF", 1.0)) * policy_loss
                + float(getattr(Config, "VALUE_COEF", 0.5)) * value_loss
                + float(getattr(Config, "ENTROPY_COEF", 0.0)) * entropy_loss
                + float(getattr(Config, "KL_COEF", 0.0)) * approx_kl
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
            self.optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy_loss += float(entropy_loss.item())

        term_return_trajectory = float(self.cumulated_return(term_rewards)[0].item())
        return_trajectory = float(self.cumulated_return(rewards)[0].item())

        avg_rewards_t = torch.stack(avg_rewards) if len(avg_rewards) else torch.tensor(0.0, device=Config.DEVICE)
        avg_rewards_returns = float(self.cumulated_return(avg_rewards_t)[0].item()) if avg_rewards_t.numel() > 0 else 0.0

        actions_list = actions_np.tolist()
        server_usage_percentage = {m: 0.0 for m in range(M)}
        for a in actions_list:
            server_usage_percentage[int(a)] += 1.0
        for m in server_usage_percentage:
            server_usage_percentage[m] /= max(len(actions_list), 1)

        probs_usage = np.array([server_usage_percentage[m] for m in range(M)], dtype=np.float64)
        ent_usage = float(-(probs_usage * np.log(probs_usage + 1e-12)).sum())

        route_dist = {i: int(sum(1 for a in actions_list if a == i)) for i in range(M)}
        mean_min_reward = float(torch.stack(min_rewards).mean().item()) if len(min_rewards) else 0.0

        return {
            "policy_loss": total_policy_loss / Config.PPO_EPOCHS,
            "value_loss": total_value_loss / Config.PPO_EPOCHS,
            "entropy_loss": total_entropy_loss / Config.PPO_EPOCHS,
            "rewards_returns": return_trajectory,
            "term_rewards_returns": term_return_trajectory,
            "min_rewards": mean_min_reward,
            "server_usage_percentage": server_usage_percentage,
            "cumulated_avg_rewards": avg_rewards_returns,
            "route distribution": route_dist,
            "entropy of route distribution": ent_usage,
            "approx_kl": float(approx_kl.detach().cpu().item()),
        }

    def compute_gae(self, rewards, values, dones=None):
        advantages = torch.zeros_like(rewards)
        gae = 0.0

        if dones is None:
            dones = torch.zeros_like(rewards)
            dones[-1] = 1

        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + Config.GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + Config.GAMMA * Config.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def cumulated_return(self, rewards):
        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + Config.GAMMA * returns[t + 1]
        return returns

    def save(self, filepath):
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

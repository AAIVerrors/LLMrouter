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


class RouterNetwork(nn.Module):
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

        # Prompt encoder (frozen)
        self.prompt_encoder = SentenceTransformer(prompt_model)
        self.prompt_encoder.eval()
        if freeze_prompt_encoder:
            for p in self.prompt_encoder.parameters():
                p.requires_grad = False

        # SentenceTransformer output dim for all-MiniLM-L6-v2 is 384
        self.prompt_projection = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prompt_dim),
        )

        # Shared trunk takes [state || prompt]
        trunk_in = state_dim + prompt_dim
        self.trunk = make_mlp(trunk_in, hidden_dim, hidden_dim, depth=trunk_depth, dropout=dropout)

        # Actor + critic heads (both MLP)
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
                prompts = [str(x.item()) if (isinstance(x, torch.Tensor) and x.dim() == 0) else str(x) for x in prompts]
        elif isinstance(prompts, list):
            prompts = [p if isinstance(p, str) else (str(p.item()) if isinstance(p, torch.Tensor) and p.dim() == 0 else str(p))
                       for p in prompts]
        else:
            prompts = [str(prompts)]

        emb = self.prompt_encoder.encode(prompts, convert_to_tensor=True)  # [N, 384]
        return emb

    def encode_prompt(self, prompts):
        emb_384 = self._encode_prompts_to_384(prompts)  # [N, 384]
    
        device = next(self.parameters()).device
        emb_384 = emb_384.to(device)
    
        emb_384 = emb_384.detach().clone()
    
        return self.prompt_projection(emb_384)  # [N, prompt_dim]


    def forward(self, state, prompt, action_mask=None):
        """
        state: [B, state_dim] or [state_dim]
        prompt: str or list[str] (or anything convertible to str)
        action_mask: [B, action_dim] or [action_dim], where 1=valid, 0=invalid
        returns: probs [B, action_dim], value [B, 1]
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B = state.shape[0]

        # Prompt embedding
        p = self.encode_prompt(prompt)  # [N, prompt_dim]
        if p.dim() == 1:
            p = p.unsqueeze(0)

        # Match batch size
        if p.shape[0] == 1 and B > 1:
            p = p.expand(B, -1)
        elif p.shape[0] != B:
            # fallback: use first prompt for all
            p = p[:1].expand(B, -1)

        # Combine state + prompt
        x = torch.cat([state, p], dim=-1)  # [B, state_dim + prompt_dim]

        # Shared trunk
        h = self.trunk(x)  # [B, hidden_dim]

        # Actor
        logits = self.actor(h)  # [B, action_dim]
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            if action_mask.shape[0] == 1 and B > 1:
                action_mask = action_mask.expand(B, -1)
            logits = logits.masked_fill(action_mask == 0, -1e9)

        probs = F.softmax(logits, dim=-1)

        # Critic
        value = self.critic(h)  # [B, 1]
        return probs, value
    
    def get_action_and_value(self, state, prompt, action_mask=None, action=None):
        """Get action and value for given state and prompt"""
        logits, value = self.forward(state, prompt, action_mask)
        probs = logits.clone()
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value, logits

    def safe_probs(self, probs):
        # Clamp negatives, replace NaN/inf, and normalize
        probs = torch.clamp(probs, min=0)
        probs[torch.isnan(probs)] = 0
        probs[torch.isinf(probs)] = 0
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = probs / (probs_sum)
        # If any row sums to zero, set uniform
        zero_rows = (probs_sum.squeeze(-1) == 0)
        if zero_rows.any():
            for i in range(probs.shape[0]):
                if zero_rows[i]:
                    probs[i] = 1.0 / probs.shape[1]
        return probs
    
    def get_action_and_value_queue(self, state, state_np, prompt, action_mask=None, action=None, service_rate=None):
        logits, value = self.forward(state, prompt, action_mask)
        probs = logits.clone()
        
        dist = Categorical(probs)
        
        # queue_scores = self.get_queue_scores_batch(state_np, service_rate=service_rate)
        # queue_probs = torch.softmax(queue_scores, dim=1)  # Softmax over actions
        # queue_log_probs = torch.log(queue_probs + 1e-8)
        # print("queue probs, logits")
        # print(queue_probs)
        # print(logits)

        # Merge probabilities
        # merged_logits = (queue_probs * Config.ALPHA) + (probs * (1 - Config.ALPHA))
        merged_logits = probs
        merged_dist = Categorical(merged_logits)
        
        if action is None:
            action = merged_dist.sample()
        
        log_prob = merged_dist.log_prob(action)
        entropy = merged_dist.entropy()
        
        return action, log_prob, entropy, value, merged_logits, queue_scores
    
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
        scores = []
        
        if not Config.MIX_QUEUE_SCORE:
            for i, capacity in enumerate(Config.SERVER_CAPACITIES):
                utilization = float(state[i])
                load = utilization * capacity
                # Avoid division by zero in service rate
                sr = max(float(service_rate[i]), 1e-4)  # Use max to prevent zero service rate
                # score = (1-factor)*(1-utilization)*(sr/(load+epslon)) \
                #         + (1-factor)*utilization*(1-(load/capacity)) \
                #         + factor*(sr/max(service_rate))
                score = sr/(load+epslon)
                scores.append(score)
            return torch.FloatTensor(scores).to(Config.DEVICE)
        else:
            for i, capacity in enumerate(Config.SERVER_CAPACITIES):
                utilization = float(state[i])
                load = utilization * capacity
                # Avoid division by zero in service rate
                sr = max(float(service_rate[i]), 1e-4)  # Use max to prevent zero service rate
                # queue_score = (1-factor)*(1-utilization)*(sr/(load+epslon)) \
                #         + (1-factor)*utilization*(1-(load/capacity)) \
                #         + factor*(sr/max(service_rate))
                queue_score = (sr/(load+epslon))
                quality_score = float(state[2 * self.num_servers + i])  # Assuming quality score is stored after server loads
                price_score = float(state[3 * self.num_servers + i])  # Assuming price score is stored after quality scores
                # Combine scores with weights   
                mix_score = (queue_score * Config.BETA) + \
                            (quality_score * Config.ALPHA) + \
                            (price_score * Config.REWARD_GAMMA)
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

        # Global (queue-agnostic) EMAs used as a fallback.
        self.lat_ema = np.full(M, float(Config.UTILITY_INIT_LAT), dtype=np.float64)
        # initialize cost from Config.PRICE (fallback if missing)
        try:
            self.cost_ema = np.asarray(Config.PRICE, dtype=np.float64).copy()
            if self.cost_ema.size != M:
                raise ValueError
        except Exception:
            self.cost_ema = np.full(M, float(Config.UTILITY_INIT_COST), dtype=np.float64)

        self.ema_alpha = float(Config.UTILITY_EMA_ALPHA)

        # Queue-conditioned predictor used inside GREEDY_UTILITY.
        # Config.UTILITY_QUEUE_MODEL in {"none", "bins", "linear"}.
        self.utility_queue_model = str(getattr(Config, "UTILITY_QUEUE_MODEL", "none")).lower()

        # Coarse queue-length bins (upper bounds). Used when UTILITY_QUEUE_MODEL == "bins".
        self._q_bin_edges = Config.SERVER_CAPACITIES

        # Per-server per-bin EMAs of *effective* latency (proc + q/mu) and cost.
        self._bin_stats = [dict() for _ in range(M)]  

        # Per-server online linear fit: lat_eff ~= a + b*q, cost ~= a_c + b_c*q
        # We keep Welford-style running moments.
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

    def _q_bin_key(self, q: float) -> int:
        """Return an upper-bound key for the queue-length bin."""
        try:
            qv = float(q)
        except Exception:
            qv = 0.0
        if not np.isfinite(qv):
            qv = 0.0
        if qv < 0:
            qv = 0.0
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
        """Online linear fit y \approx a + b*q using Welford-style covariance updates."""
        st = self.lin_stats[i]
        qv = float(q)

        n_old = int(st["n"])
        n_new = n_old + 1
        mean_q_old = float(st["mean_q"])
        dx = qv - mean_q_old
        mean_q_new = mean_q_old + dx / n_new

        # latency
        mean_lat_old = float(st["mean_lat"])
        dy_lat = float(lat_eff) - mean_lat_old
        mean_lat_new = mean_lat_old + dy_lat / n_new
        S_q_lat_new = float(st["S_q_lat"]) + dx * (float(lat_eff) - mean_lat_new)

        # cost
        mean_cost_old = float(st["mean_cost"])
        dy_cost = float(cost) - mean_cost_old
        mean_cost_new = mean_cost_old + dy_cost / n_new
        S_q_cost_new = float(st["S_q_cost"]) + dx * (float(cost) - mean_cost_new)

        # variance of q
        S_qq_new = float(st["S_qq"]) + dx * (qv - mean_q_new)

        st["n"] = n_new
        st["mean_q"] = mean_q_new
        st["S_qq"] = S_qq_new
        st["mean_lat"] = mean_lat_new
        st["S_q_lat"] = S_q_lat_new
        st["mean_cost"] = mean_cost_new
        st["S_q_cost"] = S_q_cost_new
        self.lin_stats[i] = st

    def _predict_linear(self, i: int, q: float) -> tuple[float, float] | None:
        st = self.lin_stats[i]
        if int(st["n"]) < 2:
            return None
        S_qq = float(st["S_qq"])
        if abs(S_qq) < 1e-9:
            return None

        mean_q = float(st["mean_q"])
        # latency
        b_lat = float(st["S_q_lat"]) / S_qq
        a_lat = float(st["mean_lat"]) - b_lat * mean_q
        lat = a_lat + b_lat * float(q)

        # cost
        b_cost = float(st["S_q_cost"]) / S_qq
        a_cost = float(st["mean_cost"]) - b_cost * mean_q
        cost = a_cost + b_cost * float(q)

        return float(lat), float(cost)

    def _predict_utility_components(self, i: int, q_len: float, mu: float) -> tuple[float, float]:
        """Predict (effective latency, cost) for server i given current q and service rate mu.

        - If UTILITY_QUEUE_MODEL == "none": use (lat_ema + q/mu, cost_ema)
        - If "bins"/"linear": use queue-conditioned predictor trained on
          lat_eff := processing_latency + q_at_dispatch/mu_at_dispatch (fallback to the "none" rule).
        """
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

        # mode == "none" (or unknown)
        return max(lat_fallback, 0.0), max(cost_fallback, 0.0)

    def update_server_stats(self, episode_record: list[dict]):
        """Update per-server predictors from env episode records.

        Expected keys per record (best-effort; missing keys are handled):
          - 'model' (str): model/server name
          - 'processing_latency' (float): inference latency (seconds)
          - 'price' (float): cost signal
          - 'queue_len_at_dispatch' (int/float): queue length when dispatched
          - 'service_rate' (list[float] or float): service rate(s) used that step

        We always update global EMA (processing_latency, price). If queue_len_at_dispatch
        is available and UTILITY_QUEUE_MODEL in {"bins","linear"}, we also update a
        queue-conditioned predictor on
            lat_eff := processing_latency + queue_len_at_dispatch / mu_at_dispatch.
        """
        if not episode_record:
            return

        for r in episode_record:
            mname = r.get("model", None)
            if mname is None:
                continue
            i = self._model_name_to_idx.get(mname)
            if i is None:
                continue

            lat = r.get("processing_latency", None)
            if lat is None:
                lat = r.get("processing_time", None)
            cost = r.get("price", None)

            if lat is None or not np.isfinite(lat):
                continue
            lat = float(lat)
            if cost is None or not np.isfinite(cost):
                cost = float(self.cost_ema[i])
            else:
                cost = float(cost)

            # Global EMA updates
            a = self.ema_alpha
            if int(self._utility_counts[i]) <= 0:
                self.lat_ema[i] = lat
                self.cost_ema[i] = cost
                self._utility_counts[i] = 1
            else:
                self.lat_ema[i] = (1.0 - a) * float(self.lat_ema[i]) + a * lat
                self.cost_ema[i] = (1.0 - a) * float(self.cost_ema[i]) + a * cost
                self._utility_counts[i] += 1

            # Queue-conditioned updates (if we have q)
            q = r.get("queue_len_at_dispatch", None)
            if q is None:
                q = r.get("queue_len", None)
            if q is None:
                continue
            if not np.isfinite(q):
                continue
            q = float(q)

            # extract mu for this server from possibly-vector service_rate.
            # If it's not present in the record, fall back to Config.SERVICE_RATE.
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
        """Greedy baseline: predict next-step reward per server using
        - current queue estimate Q (from utilization * capacity)
        - EMA latency/cost history
        - current quality score in state

        and choose argmax predicted reward.
        """
        M = self._util_M
        # state layout: [util(M), service_rate(M), quality(M), price(M)]
        util = state_tensor[:M].detach().cpu().numpy().astype(np.float64)
        cap = np.asarray(Config.SERVER_CAPACITIES, dtype=np.float64)
        q_len = util * cap

        # service rate (if you store it in state)
        mu = state_tensor[M:2*M].detach().cpu().numpy().astype(np.float64)
        mu = np.maximum(mu, 1e-6)

        qual = state_tensor[2*M:3*M].detach().cpu().numpy().astype(np.float64)

        # Queue-conditioned predictor for latency/cost.
        # - If Config.UTILITY_QUEUE_MODEL == "none": uses (lat_ema + Q/mu, cost_ema).
        # - If "bins"/"linear": predicts an *effective latency* already incorporating queueing.
        lat_pred = np.zeros(M, dtype=np.float64)
        cost_pred = np.zeros(M, dtype=np.float64)
        for i in range(M):
            lp, cp = self._predict_utility_components(i, float(q_len[i]), float(mu[i]))
            lat_pred[i] = lp
            cost_pred[i] = cp

        # predicted reward (you can tune weights in Config)
        score = (
            float(Config.UTILITY_W_QUAL) * qual
            - float(Config.UTILITY_W_LAT) * lat_pred
            - float(Config.UTILITY_W_COST) * cost_pred
            - float(Config.UTILITY_W_Q) * q_len
        )

        if action_mask_tensor is not None:
            mask = action_mask_tensor.detach().cpu().numpy().astype(np.float64)
            score = np.where(mask > 0.5, score, -1e18)

        if not np.isfinite(score).any():
            return int(np.random.randint(0, M))

        best = np.max(score)
        # tie-break among close maxima
        candidates = np.where(np.isclose(score, best, rtol=0.0, atol=1e-8))[0]
        return int(np.random.choice(candidates))

    def get_action(self, state, prompt, action_mask=None, alpha=Config.MERGE_ALPHA, service_rate=[1]*len(Config.SERVER_CAPACITIES), round_robin_counter=0):
        """Get action for given state and prompt"""
        alpha=Config.MERGE_ALPHA
        
        state_tensor = torch.FloatTensor(state).to(Config.DEVICE)
        
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).to(Config.DEVICE)
        else:
            action_mask_tensor = None
        
        # if Config.ROUND_ROBIN:
        #     action = round_robin_counter % self.action_dim
        #     count = 0
        #     while action_mask_tensor is not None and action_mask_tensor[action] == 0:
        #         round_robin_counter += 1
        #         action = round_robin_counter % self.action_dim
        #         count += 1
        #         if count > self.action_dim:
        #             print("All actions are masked, returning random action")
        #             action = torch.randint(0, self.action_dim, (1,)).item()
        #             break
        #     log_prob = torch.tensor(0.0).to(Config.DEVICE)
        #     entropy = torch.tensor(0.0).to(Config.DEVICE)
        #     value = torch.tensor(0.0).to(Config.DEVICE)
        #     dist_policy = torch.zeros(self.action_dim).to(Config.DEVICE)
        #     dist_policy[action] = 1.0  # Set probability for the chosen action
        #     print(f"Round Robin Action: {action}")
        #     round_robin_counter += 1
        #     return action, log_prob.cpu().item(), value.cpu().item(), round_robin_counter
        
        # reset quality
        pre_len = 2*len(Config.MODEL_NAMES)
        coefs = self.quality_scorer.compute_quality_score_all(prompt)
        for index,server in enumerate(Config.MODEL_NAMES):
            state_tensor[pre_len + index] = float(coefs[server])
            print(f"Quality score for {server}: {coefs[server]}")
        
        print(state_tensor)

        # --- Greedy utility baseline (optional) ---
        if getattr(Config, 'GREEDY_UTILITY', False):
            action = self.greedy_utility_action(state_tensor, action_mask_tensor)
            log_prob = torch.tensor(0.0, device=Config.DEVICE)
            value = torch.tensor(0.0, device=Config.DEVICE)
            print(f'Greedy-Utility action: {action}')
            return int(action), float(log_prob.item()), float(value.item()), round_robin_counter

        with torch.no_grad():
            action, log_prob, entropy, value, dist_policy = self.network.get_action_and_value(
                state_tensor, prompt, action_mask_tensor
            )
        print(dist_policy)
        print(action)
        print('entropy:', entropy)
            
        # # Get queue scores and convert to log-probabilities
        # queue_scores = self.network.get_queue_scores(state, service_rate=service_rate)
        # queue_probs = torch.softmax(queue_scores, dim=0)

        # if Config.ENTROPY_BASED_EXPLORATION:
        #     # Use entropy-based exploration
        #     ratio = entropy / math.log(self.action_dim, 2)
        #     print(f"Entropy ratio: {ratio}")
        #     merged_log_probs =  (queue_probs * ratio) +  (dist_policy * (1 - ratio))
        # # Merge log-probs for all actions
        # else:
        #     merged_log_probs =  (queue_probs * alpha) +  (dist_policy * (1 - alpha))
        
        # # Sample action from merged distribution
        # dist = Categorical(merged_log_probs)
        # action = dist.sample()
        # merged_log_prob = dist.log_prob(action)

        # print(merged_log_probs)
        # print(action)
        
        if Config.RANDOM_SELECT:
            action = torch.randint(0, self.action_dim, (1,)).item()
            log_prob = torch.tensor(0.0).to(Config.DEVICE)
            value = torch.tensor(0.0).to(Config.DEVICE)
            print(f"Randomly selected action: {action}")
            return action, log_prob.cpu().item(), value.cpu().item(), round_robin_counter
        
        if Config.ROUND_ROBIN:
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
            log_prob = torch.tensor(0.0).to(Config.DEVICE)
            entropy = torch.tensor(0.0).to(Config.DEVICE)
            value = torch.tensor(0.0).to(Config.DEVICE)
            dist_policy = torch.zeros(self.action_dim).to(Config.DEVICE)
            dist_policy[action] = 1.0  # Set probability for the chosen action
            print(f"Round Robin Action: {action}")
            round_robin_counter += 1
            return action, log_prob.cpu().item(), value.cpu().item(), round_robin_counter
            
        if Config.JSQ:
            M = len(Config.MODEL_NAMES)
        
            # utilization u in [0,1]
            u = np.asarray(state[:M], dtype=np.float32)
        
            # capacity C (max jobs, queue slots, etc.)
            C = np.asarray(Config.SERVER_CAPACITIES, dtype=np.float32)
        
            # shortest queue proxy
            q = u * C  # [M]
        
            # apply action mask if provided (1=valid, 0=invalid)
            if action_mask is not None:
                valid = (np.asarray(action_mask[:M], dtype=np.float32) > 0)
                q = np.where(valid, q, np.inf)
        
            # if all masked, fallback random
            if not np.isfinite(q).any():
                action = int(np.random.randint(0, M))
                return action, 0.0, 0.0, round_robin_counter
        
            # argmin with random tie-break
            min_q = np.min(q)
            candidates = np.where(np.isclose(q, min_q))[0]
            action = int(np.random.choice(candidates))
        
            return action, 0.0, 0.0, round_robin_counter
        if Config.P2C:
            queue_state = state[:len(Config.MODEL_NAMES)]  # utilization
            action_mask_np = None if action_mask is None else np.asarray(action_mask, dtype=np.float32)
        
            action = self.p2c_select_action(
                state=np.asarray(state, dtype=np.float32),
                action_mask=action_mask_np,
                capacities=Config.SERVER_CAPACITIES,
            )

            return action, 0.0, 0.0, round_robin_counter


        return action.cpu().item(), log_prob.cpu().item(), value.cpu().item(), round_robin_counter

    @staticmethod
    def p2c_select_action(
        state: np.ndarray,
        action_mask: np.ndarray | None,
        capacities: list[float],
        rng: np.random.Generator | None = None,
        eps: float = 1e-8,
    ) -> int:
        """
        P2C: randomly sample 2 valid servers, choose the one with smaller estimated queue/load.
    
        state: shape (state_dim,), where state[:M] are utilizations (0~1) for each server.
        action_mask: shape (M,), 1=valid, 0=invalid (optional).
        capacities: list length M.
        """
        if rng is None:
            rng = np.random.default_rng()
    
        M = len(capacities)
        util = np.asarray(state[:M], dtype=np.float64)
        cap = np.asarray(capacities, dtype=np.float64)
    
        # proxy queue length / load
        q = util * cap  # shape (M,)
    
        if action_mask is None:
            valid = np.arange(M, dtype=np.int64)
        else:
            mask = np.asarray(action_mask, dtype=np.float64)
            valid = np.where(mask > 0.5)[0]
    
        # fallback if no valid
        if valid.size == 0:
            return int(rng.integers(0, M))
    
        # if only one valid
        if valid.size == 1:
            return int(valid[0])
    
        # sample 2 distinct candidates
        if valid.size >= 2:
            c1, c2 = rng.choice(valid, size=2, replace=False)
        else:
            # shouldn't happen due to valid.size==1 handled, but keep safe
            c1 = c2 = int(valid[0])
    
        q1, q2 = q[c1], q[c2]
    
        # choose min, tie -> random among ties
        min_q = min(q1, q2)
        candidates = np.array([c1, c2], dtype=np.int64)
        qs = np.array([q1, q2], dtype=np.float64)
    
        tie_idx = np.where(np.isclose(qs, min_q, atol=eps, rtol=0.0))[0]
        chosen = int(rng.choice(candidates[tie_idx]))
        return chosen
        
    # def update(self, trajectories):
    #     """Update network using PPO algorithm"""
    #     # Prepare batch data (convert to numpy first to avoid warning)
    #     states_np = np.array([t['state'] for t in trajectories])
    #     actions_np = np.array([t['action'] for t in trajectories])
    #     old_log_probs_np = np.array([t['log_prob'] for t in trajectories])
    #     rewards_np = np.array([t['reward'] for t in trajectories])
    #     values_np = np.array([t['value'] for t in trajectories])
    #     service_rate = [t['service_rate'] for t in trajectories]
    #     route_times = [t['route_time'] for t in trajectories]
        
    #     states = torch.FloatTensor(states_np).to(Config.DEVICE)
    #     actions = torch.LongTensor(actions_np).to(Config.DEVICE)
    #     old_log_probs = torch.FloatTensor(old_log_probs_np).to(Config.DEVICE)
    #     rewards = torch.FloatTensor(rewards_np).to(Config.DEVICE)
    #     values = torch.FloatTensor(values_np).to(Config.DEVICE)
    #     prompts = [t['prompt'] for t in trajectories]
    #     route_times = torch.FloatTensor(route_times).to(Config.DEVICE)
    #     action_masks = None
        
    #     if 'action_mask' in trajectories[0] and trajectories[0]['action_mask'] is not None:
    #         action_masks_np = np.array([t['action_mask'] for t in trajectories])
    #         action_masks = torch.FloatTensor(action_masks_np).to(Config.DEVICE)
        
    #     # Compute advantages and returns
    #     advantages = self.compute_gae(rewards, values)
    #     returns = advantages + values
        
    #     # Normalize advantages
    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
    #     # PPO update
    #     total_policy_loss = 0
    #     total_value_loss = 0
    #     total_entropy_loss = 0
        
    #     intervel_subgroup_rewards = []
    #     term_values = []
    #     for j in range(1, Config.EPISODE_TIME_INTERVAL + 1):
    #         dic = {}
    #         ls = []
    #         for i in range(len(actions)):
    #             time = route_times[i].item()  # Extract scalar time
    #             if j - 1 <= time <= j:
    #                 ls.append(values[i].item())  # Extract scalar value
    #                 action = actions[i].item()  # Extract scalar action
    #                 if action not in dic:
    #                     dic[action] = []
    #                 dic[action].append(rewards[i].item())  # Extract scalar reward
    #         intervel_subgroup_rewards.append(dic)
    #         term_values.append(np.mean(ls))

    #     # Rewards transformation
    #     term_rewards = []
        
    #     for dic in intervel_subgroup_rewards:
    #         # Compute mean rewards for each action
    #         for k in dic.keys():
    #             dic[k] = np.mean(dic[k]) 
    #         # Compute term_reward only if dic is not empty
    #         values = torch.tensor(list(dic.values()), dtype=torch.float32)
    #         if len(values) > 0:  
    #             term_reward = (1 / Config.T) * torch.log(torch.exp(Config.T * values).mean())
    #             term_rewards.append(term_reward.item())  # Append scalar result
        
    #     # Returns
    #     cumulated_returns = self.cumulated_return(rewards)
    #     return_trajectory = cumulated_returns[0]  # Return of the trajectory
        
    #     # Compute mean return for each action group
    #     subgroup_returns = {}
    #     for i, v in enumerate(actions):
    #         v_item = v.item()  # Get integer value from tensor
    #         if v_item not in subgroup_returns:
    #             subgroup_returns[v_item] = []
    #         subgroup_returns[v_item].append(rewards[i].item())

    #     # Group by actions and compute mean rewards
    #     sub_mean_reward = {}
    #     for i, v in enumerate(actions):
    #         v_item = v.item()  # Get integer value from tensor
    #         if v_item not in sub_mean_reward:
    #             sub_mean_reward[v_item] = []
    #         sub_mean_reward[v_item].append(rewards[i].item())
            
    #     # Compute means reward for each action group
    #     for k in sub_mean_reward:
    #         sub_mean_reward[k] = np.mean(sub_mean_reward[k])
            
    #     # Compute means for each action group
    #     for k in subgroup_returns:
    #         subgroup_returns[k] = self.cumulated_return(torch.FloatTensor(subgroup_returns[k]).to(Config.DEVICE))[0].item()
        
    #     for _ in range(Config.PPO_EPOCHS):
    #         # Get current policy outputs
    #         if Config.USE_MERGE_TO_TRAIN:

    #             _, new_log_probs, entropy, new_values, dist, queue_scores = self.network.get_action_and_value_queue(
    #                 states,          # state (tensor)
    #                 states_np,       # state_np (numpy array)
    #                 prompts,         # prompt (list of strings)
    #                 action_masks,    # action_mask (tensor or None)
    #                 actions,         # action (tensor)
    #                 service_rate=service_rate  # service_rate (list)
    #             )
    #             print(queue_scores)

    #         else:
    #             _, new_log_probs, entropy, new_values, dist = self.network.get_action_and_value(
    #                 states, prompts, action_masks, actions
    #             )
            
    #         clip_low = 0
    #         clip_high = 0
            
    #         if Config.ADAPTIVE_EPSILON:
    #             # Select queue score for the chosen action at each time step
    #             selected_scores = queue_scores[torch.arange(queue_scores.size(0)), actions]  # Shape (153,)
    #             # Compute min and max per time step
    #             min_scores, _ = torch.min(queue_scores, dim=1)  # Shape (153,)
    #             max_scores, _ = torch.max(queue_scores, dim=1)  # Shape (153,)
    #             norm_low = (selected_scores - min_scores) / (max_scores - min_scores + 1e-5)  # Shape (153,)
    #             norm_high = (max_scores - selected_scores) / (max_scores - min_scores + 1e-5)  # Shape (153,)
    #             clip_low =  - norm_low * (1 - Config.CLIP_EPSILON) - Config.CLIP_EPSILON  # Shape (153,)
    #             clip_high = norm_high * (1 - Config.CLIP_EPSILON) + Config.CLIP_EPSILON  # Shape (153,)
    #             # print(f"Adaptive epsilon shapes: selected_scores={selected_scores}, clip_low={clip_low}, clip_high={clip_high}")
                
    #         # Term policy loss
            
    #         # Policy loss
    #         ratio = torch.exp(new_log_probs - old_log_probs)
    #         surr1 = ratio * advantages
    #         if Config.ADAPTIVE_EPSILON:
    #             surr2 = torch.clamp(ratio, clip_low, clip_high) * advantages
    #         else:
    #             surr2 = torch.clamp(ratio, 1 - Config.CLIP_EPSILON, 1 + Config.CLIP_EPSILON) * advantages
            
    #         # Group by actions and compute mean surrogate objectives
    #         subgroup = {}
    #         for i, v in enumerate(actions):
    #             v_item = v.item()  # Get integer value from tensor
    #             if v_item not in subgroup:
    #                 subgroup[v_item] = []
    #             subgroup[v_item].append(torch.min(surr1[i], surr2[i]))
            
    #         # Compute means for each action group
    #         for k in subgroup:
    #             subgroup[k] = torch.stack(subgroup[k]).mean()
            
    #         # Custom policy loss using softmin
    #         policy_values = torch.stack([subgroup[k] for k in sorted(subgroup.keys())])
    #         exp_values = torch.exp(Config.T * policy_values)
    #         softmin = (1/Config.T) * torch.log(exp_values.mean())
    #         policy_loss = -softmin  # Negative because we want to maximize
            
    #         # policy_loss = -torch.min(surr1, surr2).mean()
            
    #         # Value loss
    #         value_loss = F.mse_loss(new_values.squeeze(), returns, reduction='mean')
            
    #         # Entropy loss
    #         entropy_loss = -entropy.mean()
            
    #         # Total loss
    #         loss = (1-Config.VALUE_COEF) * policy_loss + Config.VALUE_COEF * value_loss 

    #         # Backward pass
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
    #         self.optimizer.step()
            
    #         total_policy_loss += policy_loss.item()
    #         total_value_loss += value_loss.item()
    #         total_entropy_loss += entropy_loss.item()
        
    #     return {
    #         'policy_loss': total_policy_loss / Config.PPO_EPOCHS,
    #         'value_loss': total_value_loss / Config.PPO_EPOCHS,
    #         'entropy_loss': total_entropy_loss / Config.PPO_EPOCHS,
    #         'each_server_score': subgroup,
    #         'each_server_returns': subgroup_returns,
    #         'mean_reward_per_server': sub_mean_reward,
    #         'min_score_server': min(subgroup, key=subgroup.get),
    #         'min_mean_reward_server': min(sub_mean_reward, key=sub_mean_reward.get),
    #         'min_return_server':min(subgroup_returns, key=subgroup_returns.get),
    #         'min_return_server_value':min(subgroup_returns.values()),
    #         'min_score_server_value':min(subgroup.values()),
    #         'min_mean_reward_server_value':min(sub_mean_reward.values()),
    #         'returns': return_trajectory.item()
    #     }

    def update_new(self, trajectories):
        """
        PPO update on ACTIVE intervals only (N_t > 0), with:
          - fair reward normalization: 1/M if N_t >= M, else 1/N_t (Eq. 10 vs Eq. 12)
          - interval importance weight rho_t = exp(mean_i (new_logp_i - old_logp_i)) (Eq. 14)
        """
    
        # --------- pack rollout ---------
        states_np = np.array([t["state"] for t in trajectories], dtype=np.float32)
        actions_np = np.array([t["action"] for t in trajectories], dtype=np.int64)
        old_log_probs_np = np.array([t["log_prob"] for t in trajectories], dtype=np.float32)
        rewards_np = np.array([t["reward"] for t in trajectories], dtype=np.float32)
        values_np = np.array([t["value"] for t in trajectories], dtype=np.float32)
    
        prompts = [t["prompt"] for t in trajectories]
        service_rate = [t["service_rate"] for t in trajectories]
    
        # IMPORTANT: treat time_slot as integer label
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
    
        M = len(Config.MODEL_NAMES)  # number of servers/models
        beta = float(Config.T)       # your tilting parameter used in log-mean-exp
    
        def log_mean_exp(x: torch.Tensor, denom: int):
            """
            (1/beta) * log( (1/denom) * sum exp(beta * x) )
            Robust to beta ~ 0.
            """
            denom = max(int(denom), 1)
            if abs(beta) < 1e-8:
                return x.mean()
            return (torch.logsumexp(beta * x, dim=0) - math.log(denom)) / beta
    
        # --------- build ACTIVE interval summaries ---------
        # group indices by time_slot
        slot_to_indices = {}
        for i, ts in enumerate(time_slots.tolist()):
            slot_to_indices.setdefault(int(ts), []).append(i)
    
        active_slots = []
        interval_indices = []      # list[LongTensor] each interval's step indices
        first_indices = []         # first step index per active interval
        term_rewards = []          # scalar per active interval
        term_values_old = []       # baseline V(s_t) from rollout (no grad)
    
        # optional metrics
        avg_rewards = []
        min_rewards = []
    
        for ts in sorted(slot_to_indices.keys()):
            idxs = torch.tensor(slot_to_indices[ts], device=Config.DEVICE, dtype=torch.long)
            Nt = int(idxs.numel())
            if Nt == 0:
                continue  # inactive
    
            # active interval bookkeeping
            active_slots.append(ts)
            interval_indices.append(idxs)
            first_indices.append(int(idxs[0].item()))
    
            # per-interval tensors
            r_t = rewards[idxs]      # [Nt]
            a_t = actions[idxs]      # [Nt]
    
            avg_rewards.append(r_t.mean())
            min_rewards.append(r_t.min())
    
            # baseline value at interval start (from rollout)
            term_values_old.append(values[idxs[0]])
    
            # ---- fair reward aggregation ----
            # compute per-server average rewards rbar_{m,t} over servers that were used this interval
            # Eq. (11)/(13): rbar_{m,t} = mean of rewards routed to server m in interval t
            server_means = []
            if Nt >= M:
                # Eq. (10): sum over all M servers; for servers with Nm_t=0, assign a floor (penalize "unused")
                floor = r_t.min().detach()
                for m in range(M):
                    mask = (a_t == m)
                    if mask.any():
                        server_means.append(r_t[mask].mean())
                    else:
                        server_means.append(floor)
                server_means = torch.stack(server_means)  # [M]
                tr = log_mean_exp(server_means, denom=M)
            else:
                # Eq. (12): normalize by Nt (limited arrivals)
                # Use ONLY servers that actually received traffic (no forced penalty for "unreachable" servers when Nt<M)
                for m in range(M):
                    mask = (a_t == m)
                    if mask.any():
                        server_means.append(r_t[mask].mean())
                # if for some reason none (shouldn't happen if Nt>0), skip
                if len(server_means) == 0:
                    continue
                server_means = torch.stack(server_means)  # [K], K<=Nt
                tr = log_mean_exp(server_means, denom=Nt)
    
            term_rewards.append(tr)
    
        # if no active intervals, do nothing safely
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
            }
    
        term_rewards = torch.stack(term_rewards)                  # [T+]
        term_values_old = torch.stack(term_values_old)            # [T+]
        first_indices_t = torch.tensor(first_indices, device=Config.DEVICE, dtype=torch.long)
    
        # advantages/returns on ACTIVE intervals only (paper: exclude Nt=0)
        dones = torch.zeros_like(term_rewards)
        dones[-1] = 1
        term_adv = self.compute_gae(term_rewards, term_values_old, dones=dones)
        term_ret = term_adv + term_values_old
    
        # normalize advantages
        term_adv = (term_adv - term_adv.mean()) / (term_adv.std() + 1e-5)
    
        # --------- PPO epochs ---------
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
    
        for _ in range(Config.PPO_EPOCHS):
            if Config.USE_MERGE_TO_TRAIN:
                _, new_log_probs, entropy, new_values, dist, _queue_scores = self.network.get_action_and_value_queue(
                    states, states_np, prompts, action_masks, actions, service_rate=service_rate
                )
            else:
                _, new_log_probs, entropy, new_values, dist = self.network.get_action_and_value(
                    states, prompts, action_masks, actions
                )
    
            # interval importance weights rho_t (Eq. 14)
            rhos = []
            for idxs in interval_indices:
                # rho_t = exp( mean_i (new_logp_i - old_logp_i) )
                rho_t = torch.exp((new_log_probs[idxs] - old_log_probs[idxs]).mean())
                rhos.append(rho_t)
            rhos = torch.stack(rhos)  # [T+]
    
            # PPO clipped surrogate on intervals
            surr1 = rhos * term_adv
            surr2 = torch.clamp(rhos, 1.0 - Config.CLIP_EPSILON, 1.0 + Config.CLIP_EPSILON) * term_adv
            policy_loss = -torch.min(surr1, surr2).mean()
    
            # critic loss
            v_pred = new_values.squeeze(-1)[first_indices_t]      # [T+], has grad
            value_loss = F.mse_loss(v_pred, term_ret.detach())    # detach target
    
            entropy_loss = -entropy.mean()

            kls = []
            for idxs in interval_indices:
                # KL(old||new) approx = E_old[logp_old - logp_new]
                kl_t = (old_log_probs[idxs] - new_log_probs[idxs]).mean()
                kls.append(kl_t)
            
            approx_kl = torch.stack(kls).mean()
            approx_kl = torch.clamp(approx_kl, min=0.0)  # optional safety

            loss = (
                Config.POLICY_COEF * policy_loss
                + Config.VALUE_COEF * value_loss
                + Config.ENTROPY_COEF * entropy_loss
                + Config.KL_COEF * approx_kl
            )

    
            # loss = (
            #     Config.POLICY_COEF * policy_loss
            #     + Config.VALUE_COEF * value_loss
            #     + Config.ENTROPY_COEF * entropy_loss
            # )
    
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
            self.optimizer.step()
    
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy_loss += float(entropy_loss.item())
    
        # metrics
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
    
        # empirical routing entropy
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

    
    # def update_new(self, trajectories):
        # """Update network using PPO algorithm"""
        # # Prepare batch data (convert to numpy first to avoid warning)
        # states_np = np.array([t['state'] for t in trajectories])
        # actions_np = np.array([t['action'] for t in trajectories])
        # old_log_probs_np = np.array([t['log_prob'] for t in trajectories])
        # rewards_np = np.array([t['reward'] for t in trajectories])
        # values_np = np.array([t['value'] for t in trajectories])
        # service_rate = [t['service_rate'] for t in trajectories]
        # route_times = [t['route_time'] for t in trajectories]
        
        # states = torch.FloatTensor(states_np).to(Config.DEVICE)
        # actions = torch.LongTensor(actions_np).to(Config.DEVICE)
        # old_log_probs = torch.FloatTensor(old_log_probs_np).to(Config.DEVICE)
        # rewards = torch.FloatTensor(rewards_np).to(Config.DEVICE)
        # values = torch.FloatTensor(values_np).to(Config.DEVICE)
        # prompts = [t['prompt'] for t in trajectories]
        # route_times = torch.FloatTensor(route_times).to(Config.DEVICE)
        # time_slots = torch.FloatTensor([t['time_slot'] for t in trajectories]).to(Config.DEVICE)
        # action_masks = None
        
        # #save trajectories as json in the folder
        # # os.makedirs('trajectories', exist_ok=True)
        # # current_time = tm.strftime("%Y%m%d-%H%M%S")
        # # serializable_trajectories = []
        # # for t in trajectories:
        # #     serializable_trajectory = t.copy()
        # #     # Convert any non-serializable items to lists
        # #     for key, value in serializable_trajectory.items():
        # #         if isinstance(value, np.ndarray):
        # #             serializable_trajectory[key] = value.tolist()
        # #         elif isinstance(value, torch.Tensor):
        # #             serializable_trajectory[key] = value.cpu().numpy().tolist()
        # #     serializable_trajectories.append(serializable_trajectory)
        # # with open(f'trajectories/trajectories-{current_time}.json', 'w') as f:
        # #     json.dump(serializable_trajectories, f, indent=4)

        # if 'action_mask' in trajectories[0] and trajectories[0]['action_mask'] is not None:
        #     action_masks_np = np.array([t['action_mask'] for t in trajectories])
        #     action_masks = torch.FloatTensor(action_masks_np).to(Config.DEVICE)
        
        # # Compute advantages and returns
        # # advantages = self.compute_gae(rewards, values)
        # # returns = advantages + values
        
        # # Normalize advantages
        # # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # # PPO update
        # total_policy_loss = 0
        # total_value_loss = 0
        # total_entropy_loss = 0
        
        # term_values = []
        # null_timeslots = []
        # intervel_subgroup_rewards = [{server: [] for server in range(len(Config.MODEL_NAMES))} for _ in range(Config.EPISODE_TIME_INTERVAL+1)]

        # for i in range(len(actions)):
        #     time = time_slots[i].item()  # Extract scalar time
        #     action = actions[i].item()  # Extract scalar action
        #     reward = rewards[i].item()  # Extract scalar reward
        #     intervel_subgroup_rewards[int(time)][int(action)].append(reward)

        # # term values is the the first value in each time slot
        # for j in range(0, Config.EPISODE_TIME_INTERVAL+1):
        #     found_value = False
        #     for i in range(len(actions)):
        #         time = time_slots[i].item()  # Extract scalar time
        #         if time == j: 
        #             if not found_value:
        #                 term_values.append(values[i].item())
        #                 found_value = True
        #             break
        #     if not found_value:
        #         null_timeslots.append(j)

        # # for j in range(0, Config.EPISODE_TIME_INTERVAL + 1):
        # #     dic = {}
        # #     for i in range(len(Config.MODEL_NAMES)):
        # #         dic[i] = []
        # #     found_value = False
        # #     for i in range(len(actions)):
        # #         time = time_slots[i].item()  # Extract scalar time
        # #         if time == j: 
        # #             if not found_value:
        # #                 term_values.append(values[i].item())
        # #                 found_value = True
        # #             action = actions[i].item()  # Extract scalar action
        # #             dic[action].append(rewards[i].item())  # Extract scalar reward
        # #     if not found_value:
        # #         term_values.append(0.0)  # Append default value if no actions in this interval
        # #     intervel_subgroup_rewards.append(dic)
        
        # # Rewards transformation
        # term_rewards = []
        # avg_rewards = []
        # min_rewards = []
        # for dic in intervel_subgroup_rewards:
        #     # Compute mean rewards for each action
        #     cur = dic.copy()
        #     num_routes = sum(len(v) for v in cur.values())
        #     num_zeros = sum(1 for v in cur.values() if len(v) == 0)
        #     values = torch.tensor([item for sublist in dic.values() for item in sublist], dtype=torch.float32)
        #     if len(values) > 0:
        #         avg_rewards.append(values.mean().item())
        #         min_rewards.append(values.min().item())
        #     for k in cur.keys():
        #         if len(cur[k]) > 0:
        #             cur[k] = np.mean(cur[k])
        #         else:
        #             cur[k] = -Config.BETA - Config.REWARD_GAMMA  # Penalty for no selections
        #     # If there are fewer routes than models, remove some zeros to avoid excessive penalties
        #     if num_routes < len(Config.MODEL_NAMES):
        #         times = len(Config.MODEL_NAMES) - num_routes
        #         for k in list(cur.keys()):
        #             if cur[k] == -Config.BETA - Config.REWARD_GAMMA and times > 0:
        #                 del cur[k]
        #                 times -= 1
        #     values = torch.tensor(list(cur.values()), dtype=torch.float32)
        #     print(cur)
        #     if len(values) == 0:
        #         continue
        #     if Config.USE_AVG == False:
        #         term_reward = (1 / Config.T) * torch.log(torch.exp(Config.T * values).mean() + 1e-5)  # Add small epsilon to avoid log(0)
        #     else:
        #         term_reward = values.mean()
        #     term_rewards.append(term_reward.item())  # Append scalar result

        # # Convert to tensors
        # term_rewards = torch.FloatTensor(term_rewards).to(Config.DEVICE)
        # term_values = torch.FloatTensor(term_values).to(Config.DEVICE)
        # min_rewards = torch.FloatTensor(min_rewards).to(Config.DEVICE)
        # avg_rewards = torch.FloatTensor(avg_rewards).to(Config.DEVICE)
        
        # print('time_slots:', time_slots)
        # print('actions:', actions)
        # print('rewards:', rewards)
        # print('intervel_subgroup_rewards:', intervel_subgroup_rewards)
        # print('term_rewards:', term_rewards)
        # print('term_values:', term_values)
        # print('min_rewards:', min_rewards)
        # print('avg_rewards:', avg_rewards)

        # # Verify lengths
        # if len(term_rewards) != len(term_values):
        #     raise ValueError(f"Mismatch in term_rewards ({len(term_rewards)}) and term_values ({len(term_values)}) lengths")

        # # Calculate advantages
        # term_advantages = self.compute_gae(term_rewards, term_values)
        # term_returns = term_advantages + term_values
        # term_advantages = (term_advantages - term_advantages.mean()) / (term_advantages.std() + 1e-5)
        
        # print('term_advantages:', term_advantages)
        
        
        # for _ in range(Config.PPO_EPOCHS):
        #     # Get current policy outputs
        #     if Config.USE_MERGE_TO_TRAIN:

        #         _, new_log_probs, entropy, new_values, dist, queue_scores = self.network.get_action_and_value_queue(
        #             states,          # state (tensor)
        #             states_np,       # state_np (numpy array)
        #             prompts,         # prompt (list of strings)
        #             action_masks,    # action_mask (tensor or None)
        #             actions,         # action (tensor)
        #             service_rate=service_rate  # service_rate (list)
        #         )
        #         # print(queue_scores)

        #     else:
        #         _, new_log_probs, entropy, new_values, dist = self.network.get_action_and_value(
        #             states, prompts, action_masks, actions
        #         )

                
        #     # Term policy loss
        #     new_log_probs_grouped = torch.zeros(Config.EPISODE_TIME_INTERVAL+1).to(Config.DEVICE)
        #     new_dic = {}
        #     old_dic = {}
        #     for i, j in enumerate(time_slots):
        #         key = j.item()
        #         if key not in new_dic:
        #             new_dic[key] = []
        #         new_dic[key].append(new_log_probs[i])
        #         if key not in old_dic:
        #             old_dic[key] = []
        #         old_dic[key].append(old_log_probs[i])
        #     for k in new_dic:
        #         # if k didnt exist in old_dic, then skip
        #         new_tensor = torch.stack(new_dic[k])
        #         old_tensor = torch.stack(old_dic[k])
        #         diff = torch.clamp(new_tensor - old_tensor, min=1e-5)
        #         new_log_probs_grouped[int(k)-1] = torch.exp(torch.log(diff).mean())
            
        #     # delete the time slots that are null
        #     if len(null_timeslots) > 0:
        #         new_log_probs_grouped = torch.tensor([new_log_probs_grouped[i] for i in range(len(new_log_probs_grouped)) if i not in null_timeslots], dtype=torch.float32).to(Config.DEVICE)
                
        #     surr1 = new_log_probs_grouped * term_advantages
        #     surr2 = torch.clamp(new_log_probs_grouped, 1 - Config.CLIP_EPSILON, 1 + Config.CLIP_EPSILON) * term_advantages
            
        #     term_values = []

        #     for j in range(0, Config.EPISODE_TIME_INTERVAL+1):
        #         t = False
        #         for i in range(len(new_values)):
        #             time = time_slots[i]  # Extract scalar time
        #             if time == j: 
        #                 term_values.append(new_values[i].item())
        #                 t = True
        #                 break
        #         if not t:
        #             continue  # Default value if no action in this interval
                    
        #     print('term_values_new:', term_values)

        #     term_values = torch.FloatTensor(term_values).to(Config.DEVICE)

        #     policy_loss = -torch.min(surr1, surr2).mean()
        #     # term_policy_loss = - 1/Config.T * torch.log(torch.exp(Config.T * torch.min(surr1, surr2)).mean() + 1e-8)
        #     # policy_loss = term_policy_loss
        #     value_loss = F.mse_loss(term_values.squeeze(), term_returns, reduction='mean')
                
            
        #     # Policy loss
        #     # ratio = torch.exp(new_log_probs - old_log_probs)
        #     # surr1 = ratio * advantages
        #     # if Config.ADAPTIVE_EPSILON:
        #     #     surr2 = torch.clamp(ratio, clip_low, clip_high) * advantages
        #     # else:
        #     #     surr2 = torch.clamp(ratio, 1 - Config.CLIP_EPSILON, 1 + Config.CLIP_EPSILON) * advantages
            
        #     # # Group by actions and compute mean surrogate objectives
        #     # subgroup = {}
        #     # for i, v in enumerate(actions):
        #     #     v_item = v.item()  # Get integer value from tensor
        #     #     if v_item not in subgroup:
        #     #         subgroup[v_item] = []
        #     #     subgroup[v_item].append(torch.min(surr1[i], surr2[i]))
            
        #     # # Compute means for each action group
        #     # for k in subgroup:
        #     #     subgroup[k] = torch.stack(subgroup[k]).mean()

            
        #     # policy_loss = -torch.min(surr1, surr2).mean()
            
        #     # Value loss
        #     # value_loss = F.mse_loss(new_values.squeeze(), returns, reduction='mean')
            
        #     # Entropy loss
        #     entropy_loss = -entropy.mean()
            
        #     # Total loss
        #     loss = Config.POLICY_COEF * policy_loss + Config.VALUE_COEF * value_loss + Config.ENTROPY_COEF * entropy_loss

        #     # Backward pass
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
        #     self.optimizer.step()
            
        #     total_policy_loss += policy_loss.item()
        #     total_value_loss += value_loss.item()
        #     total_entropy_loss += entropy_loss.item()
            
        # term_cumulated_returns = self.cumulated_return(term_rewards)
        # term_return_trajectory = term_cumulated_returns[0]  # Return of the trajectory
        
        # cumulated_returns = self.cumulated_return(rewards)
        # return_trajectory = cumulated_returns[0]  # Return of the trajectory
        
        # cumulated_avg_rewards = self.cumulated_return(avg_rewards)
        # avg_rewards_returns = cumulated_avg_rewards[0]  # Return of the trajectory

        # server_usage_percentage = {server: 0 for server in range(len(Config.SERVER_CAPACITIES))}
        # for i in range(len(actions_np)):
        #     action = actions_np[i]
        #     server_usage_percentage[action] += 1
        # for k in server_usage_percentage:
        #     server_usage_percentage[k] /= len(actions_np)
        
        # return {
        #     'policy_loss': total_policy_loss / Config.PPO_EPOCHS,
        #     'value_loss': total_value_loss / Config.PPO_EPOCHS,
        #     'entropy_loss': total_entropy_loss / Config.PPO_EPOCHS,
        #     'rewards_returns': return_trajectory.item(),
        #     'term_rewards_returns': term_return_trajectory.item(),
        #     'min_rewards': torch.mean(min_rewards).item(),
        #     'server_usage_percentage': server_usage_percentage,
        #     'cumulated_avg_rewards': avg_rewards_returns,
        #     'route distribution': {i: len([a for a in actions_np if a == i]) for i in range(len(Config.SERVER_CAPACITIES))},
        #     'entropy of route distribution': -sum((len([a for a in actions_np if a == i])/len(actions_np)) * math.log((len([a for a in actions_np if a == i])+1e-5)/len(actions_np)+1e-5) for i in range(len(Config.SERVER_CAPACITIES))),
        #     # entropy of route distribution is calculated to measure the diversity of the routing decisions
        #     # math of route distribution is calculated to measure the average uncertainty in the routing decisions
        #     # higher entropy indicates more diverse routing decisions, while lower entropy indicates more concentrated routing decisions
        #     # both metrics can provide insights into the exploration-exploitation balance of the routing policy
        #     # in a scenario where one server is heavily favored, the entropy will be low, indicating less exploration
        #     # in a scenario where all servers are equally used, the entropy will be high, indicating
        #     # a good balance between exploration and exploitation
        #     # math expression: -sum(p * log(p) for p in probabilities if p > 0)
        # }
    
    def compute_gae(self, rewards, values, dones=None):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # If no dones provided, assume all episodes continue except the last
        if dones is None:
            dones = torch.zeros_like(rewards)
            dones[-1] = 1
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + Config.GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + Config.GAMMA * Config.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages

    def cumulated_return(self, rewards):
        """Compute cumulated returns"""
        returns = torch.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + Config.GAMMA * returns[t + 1]
        return returns

    def save(self, filepath):
        """Save model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
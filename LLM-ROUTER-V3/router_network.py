import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from sentence_transformers import SentenceTransformer
try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None
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

def make_actor_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int, dropout: float):
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

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, d_model]
        # key_padding_mask: [B, T], True means the token is padding and should be ignored.
        h, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x



class RouterNetwork(nn.Module):
    """
    RouterNetwork with a token-level prompt encoder for USE_CLIP_FUSION_ROUTER=True.

    Key change:
      - Old CLIP fusion path:
          prompt text -> SentenceTransformer pooled embedding [B, H]
          -> prompt_tower(...).unsqueeze(1) -> [B, 1, d]
        This compresses the whole prompt into one token.

      - New LLaVA-style fusion path:
          prompt text -> AutoTokenizer + AutoModel
          -> last_hidden_state [B, L, H]
          -> prompt_token_proj -> [B, L, d]
          -> concat [ROUTE] + prompt tokens + server tokens
          -> lightweight fusion Transformer
          -> actor_head(server_h) -> logits [B, M]

    Non-CLIP branches are kept compatible with your old SentenceTransformer pooled embedding path.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = Config.HIDDEN_DIM,
        prompt_dim: int = 64,
        trunk_depth: int = 4,
        head_depth: int = 2,
        dropout: float = 0,
        prompt_model: str = "all-MiniLM-L6-v2",
        freeze_prompt_encoder: bool = True,
    ):
        super().__init__()

        # self.actor_log_temp = nn.Parameter(torch.zeros(1))

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.use_clip_fusion = bool(getattr(Config, "USE_CLIP_FUSION_ROUTER", False))
        self.use_attn = bool(getattr(Config, "USE_ATTN_ROUTER", False))

        self.include_quality_state = bool(getattr(Config, "INCLUDE_QUALITY_IN_STATE", True)) and not bool(
            getattr(Config, "USE_EM_EXACT_MATCH", False)
        )

        self.server_feat_dim = (
            int(getattr(Config, "SERVER_DYN_DIM", 1))
            + int(getattr(Config, "SERVER_STAT_DIM", 3))
        )

        # =========================================================
        # Prompt encoder
        # =========================================================
        if self.use_clip_fusion:
            # Token-level encoder: do NOT pool the prompt into one vector.
            if AutoTokenizer is None or AutoModel is None:
                raise ImportError("transformers is required for token-level prompt encoding. Please install `transformers`.")

            self.prompt_model_name = getattr(Config, "PROMPT_MODEL", prompt_model)

            self.prompt_tokenizer = AutoTokenizer.from_pretrained(
                self.prompt_model_name,
                trust_remote_code=True,
            )

            self.prompt_encoder = AutoModel.from_pretrained(
                self.prompt_model_name,
                trust_remote_code=True,
            )
            self.prompt_encoder.eval()

            if freeze_prompt_encoder:
                for p in self.prompt_encoder.parameters():
                    p.requires_grad = False

            self.prompt_emb_dim = int(getattr(self.prompt_encoder.config, "hidden_size", 768))
            self.prompt_dim = self.prompt_emb_dim
            self.prompt_projection = None
            self.use_prompt_projection = False

        else:
            # Old pooled sentence embedding path for non-CLIP branches.
            self.prompt_encoder = SentenceTransformer(getattr(Config, "PROMPT_MODEL", prompt_model))
            self.prompt_encoder.eval()
            if freeze_prompt_encoder:
                for p in self.prompt_encoder.parameters():
                    p.requires_grad = False

            self.prompt_emb_dim = int(getattr(self.prompt_encoder, "get_sentence_embedding_dimension", lambda: 384)())

            self.use_prompt_projection = bool(getattr(Config, "USE_PROMPT_PROJECTION", False))
            if self.use_prompt_projection:
                self.prompt_projection = nn.Sequential(
                    nn.Linear(self.prompt_emb_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, prompt_dim),
                )
                self.prompt_dim = int(prompt_dim)
            else:
                self.prompt_projection = None
                self.prompt_dim = int(self.prompt_emb_dim)

        # =========================================================
        # LLaVA-style branch: token-level prompt + server tokens -> fusion Transformer
        # =========================================================
        if self.use_clip_fusion:
            self.M = int(action_dim)

            d_model = int(getattr(Config, "ATTN_D_MODEL", 256))
            n_heads = int(getattr(Config, "ATTN_N_HEADS", 8))
            n_layers = int(getattr(Config, "ATTN_N_LAYERS", 2))
            ff_mult = int(getattr(Config, "ATTN_FF_MULT", 4))
            attn_drop = float(getattr(Config, "ATTN_DROPOUT", dropout))

            if d_model % n_heads != 0:
                raise ValueError(f"ATTN_D_MODEL ({d_model}) must be divisible by ATTN_N_HEADS ({n_heads}).")

            self.d_model = d_model

            # Per-server layout:
            #   [util, slot_count, mu, price_in, price_out]
            self.dyn_feat_dim = int(getattr(Config, "SERVER_DYN_DIM", 2))
            self.stat_feat_dim = int(getattr(Config, "SERVER_STAT_DIM", 3))
            self.server_feat_dim = self.dyn_feat_dim + self.stat_feat_dim

            # Dynamic/static server channel projections
            self.server_dyn_proj = nn.Linear(self.dyn_feat_dim, d_model)
            self.server_stat_proj = nn.Linear(self.stat_feat_dim, d_model)

            self.dyn_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            self.stat_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.dyn_type_emb, std=0.02)
            nn.init.normal_(self.stat_type_emb, std=0.02)

            self.server_pos_emb = nn.Parameter(torch.zeros(1, 2 * self.M, d_model))
            nn.init.normal_(self.server_pos_emb, std=0.02)

            # Server self-attention: servers see other servers before looking at prompt.
            self.server_self_attn = nn.ModuleList([
                AttnBlock(d_model, n_heads, ff_mult, attn_drop)
                for _ in range(n_layers)
            ])

            # Token-level prompt projection: [B, L, H] -> [B, L, d_model]
            self.prompt_token_proj = nn.Sequential(
                nn.LayerNorm(self.prompt_emb_dim),
                nn.Linear(self.prompt_emb_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )

            # ---- LLaVA-style fusion transformer ----
            # Treat server states as structured "routing tokens" and project them into
            # the same d_model space as prompt tokens. Then concatenate:
            #   [ROUTE] + prompt_tokens + server_tokens
            # and let a lightweight Transformer perform joint prompt-server interaction.
            self.route_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.route_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            self.prompt_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            self.server_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.route_token, std=0.02)
            nn.init.normal_(self.route_type_emb, std=0.02)
            nn.init.normal_(self.prompt_type_emb, std=0.02)
            nn.init.normal_(self.server_type_emb, std=0.02)

            fusion_layers = int(getattr(Config, "LLAVA_FUSION_LAYERS", 2))
            self.fusion_blocks = nn.ModuleList([
                AttnBlock(d_model, n_heads, ff_mult, attn_drop)
                for _ in range(fusion_layers)
            ])
            self.fusion_ln = nn.LayerNorm(d_model)

            # Fuse dynamic and static channels into one server representation.
            self.server_fuse = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )

            # Actor reads one prompt-conditioned server_h per server.
            self.actor_head = make_actor_mlp(
                in_dim=2 * d_model,
                hidden_dim=d_model,
                out_dim=1,
                depth=3,
                dropout=attn_drop,
            )
            
            # Critic attention pooling over servers.
            # route_h tells the critic which server states are important for this prompt/state.
            self.critic_server_score = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(attn_drop),
                nn.Linear(d_model, 1),
            )

            # Critic pools token-level prompt and server representations.
            self.critic_mlp = make_mlp(
                2 * d_model,
                d_model,
                1,
                depth=3,
                dropout=attn_drop,
            )

        elif self.use_attn:
            self.M = int(action_dim)

            d_model = int(getattr(Config, "ATTN_D_MODEL", 256))
            n_heads = int(getattr(Config, "ATTN_N_HEADS", 8))
            n_layers = int(getattr(Config, "ATTN_N_LAYERS", 4))
            ff_mult = int(getattr(Config, "ATTN_FF_MULT", 4))
            attn_drop = float(getattr(Config, "ATTN_DROPOUT", dropout))
            self.use_global_token = bool(getattr(Config, "ATTN_USE_GLOBAL_TOKEN", True))

            if d_model % n_heads != 0:
                raise ValueError(f"ATTN_D_MODEL ({d_model}) must be divisible by ATTN_N_HEADS ({n_heads}).")

            self.d_model = d_model

            self.dyn_feat_dim = int(getattr(Config, "SERVER_DYN_DIM", 1))
            self.stat_feat_dim = int(getattr(Config, "SERVER_STAT_DIM", 3))
            self.server_feat_dim = self.dyn_feat_dim + self.stat_feat_dim

            self.server_dyn_proj = nn.Linear(self.dyn_feat_dim, d_model)
            self.server_stat_proj = nn.Linear(self.stat_feat_dim, d_model)

            self.dyn_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            self.stat_type_emb = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.dyn_type_emb, mean=0.0, std=0.02)
            nn.init.normal_(self.stat_type_emb, mean=0.0, std=0.02)

            self.prompt_token_proj = nn.Linear(self.prompt_dim, d_model)

            self.global_token_proj = None
            self.has_quality = False
            self.has_price = True
            self.global_dim = 0

            max_tokens = 1 + 2 * self.M
            self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, d_model))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

            self.attn_blocks = nn.ModuleList(
                [AttnBlock(d_model=d_model, n_heads=n_heads, ff_mult=ff_mult, dropout=attn_drop) for _ in range(n_layers)]
            )

            self.actor_ln = nn.LayerNorm(3 * d_model)
            self.actor_head = make_actor_mlp(
                in_dim=3 * d_model,
                hidden_dim=d_model,
                out_dim=1,
                depth=3,
                dropout=0,
            )

            self.critic_mlp = make_mlp(2 * d_model, d_model, 1, depth=3, dropout=attn_drop)

        else:
            self.M = int(action_dim)
            self.use_serverwise_mlp = bool(getattr(Config, "USE_SERVERWISE_MLP", True))

            if self.use_serverwise_mlp:
                trunk_in = self.server_feat_dim + self.prompt_dim

                self.server_feat_ln = nn.LayerNorm(self.server_feat_dim)
                self.prompt_ln = nn.LayerNorm(self.prompt_dim)

                self.server_mlp = make_mlp(
                    trunk_in,
                    hidden_dim,
                    hidden_dim,
                    depth=trunk_depth,
                    dropout=dropout,
                )

                self.server_actor = make_mlp(
                    hidden_dim,
                    hidden_dim,
                    1,
                    depth=head_depth,
                    dropout=dropout,
                )

                self.server_critic = make_mlp(
                    hidden_dim,
                    hidden_dim,
                    1,
                    depth=head_depth,
                    dropout=dropout,
                )
            else:
                trunk_in = state_dim + self.prompt_dim
                self.state_ln = nn.LayerNorm(state_dim)
                self.prompt_ln = nn.LayerNorm(self.prompt_dim)
                self.trunk = make_mlp(trunk_in, hidden_dim, hidden_dim, depth=trunk_depth, dropout=dropout)
                self.actor = make_mlp(hidden_dim, hidden_dim, action_dim, depth=head_depth, dropout=dropout)
                self.critic = make_mlp(hidden_dim, hidden_dim, 1, depth=head_depth, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        # Do not touch pretrained prompt_encoder weights.
        pretrained_ids = set()
        if hasattr(self, "prompt_encoder"):
            pretrained_ids = {id(p) for p in self.prompt_encoder.parameters()}

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if id(m.weight) in pretrained_ids:
                    continue
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None and id(m.bias) not in pretrained_ids:
                    nn.init.zeros_(m.bias)

        actor_out, critic_out = self._get_output_layers()

        if actor_out is not None:
            nn.init.orthogonal_(actor_out.weight, gain=0.5)
            if actor_out.bias is not None:
                nn.init.zeros_(actor_out.bias)

        if critic_out is not None:
            nn.init.orthogonal_(critic_out.weight, gain=1.0)
            if critic_out.bias is not None:
                nn.init.zeros_(critic_out.bias)

    def _get_output_layers(self):
        def last_linear(module):
            if isinstance(module, nn.Linear):
                return module
            if isinstance(module, nn.Sequential):
                for m in reversed(module):
                    if isinstance(m, nn.Linear):
                        return m
            return None

        if getattr(self, "use_clip_fusion", False):
            return last_linear(self.actor_head), last_linear(self.critic_mlp)
        elif self.use_attn:
            return last_linear(self.actor_head), last_linear(self.critic_mlp)
        elif getattr(self, "use_serverwise_mlp", False):
            return last_linear(self.server_actor), last_linear(self.server_critic)
        else:
            return last_linear(self.actor), last_linear(self.critic)

    # =========================================================
    # Prompt preprocessing
    # =========================================================
    @staticmethod
    def _normalize_prompt_list(prompts):
        if isinstance(prompts, str):
            return [prompts]
        elif isinstance(prompts, (int, float)):
            return [str(prompts)]
        elif isinstance(prompts, torch.Tensor):
            if prompts.dim() == 0:
                return [str(prompts.item())]
            return [
                str(x.item()) if (isinstance(x, torch.Tensor) and x.dim() == 0) else str(x)
                for x in prompts
            ]
        elif isinstance(prompts, list):
            return [
                p if isinstance(p, str) else (
                    str(p.item()) if isinstance(p, torch.Tensor) and p.dim() == 0 else str(p)
                )
                for p in prompts
            ]
        else:
            return [str(prompts)]

    @staticmethod
    def _make_routing_text(text: str) -> str:
        """
        Routing-only text.

        Keep useful task/question/choice information, remove final-answer boilerplate.
        The generation model still receives the original full prompt elsewhere.
        """
        raw = str(text).strip()

        # Remove output-format instruction, but keep choices/options.
        for stop in (
            "\nOutput the final answer",
            "\nPlease output",
            "\nFinal answer",
            "\nReturn only",
        ):
            idx = raw.find(stop)
            if idx != -1:
                raw = raw[:idx].strip()

        # MMLU / multiple-choice format: keep question + choices.
        if "Question:" in raw:
            q = raw.split("Question:", 1)[1].strip()
            ans_idx = q.find("\nAnswer:")
            if ans_idx != -1:
                q = q[:ans_idx].strip()
            return "Task: multiple_choice\n" + q

        # Explicit QA context format. Do not feed full context into router;
        # keep question and context stats.
        for marker in ("\nContext:", "\ncontext:", "\nContext :", "Context:"):
            if marker in raw:
                question, context = raw.split(marker, 1)
                question = question.strip()
                context = context.strip()
                return (
                    "Task: qa_with_context\n"
                    f"Question: {question}\n"
                    f"Context stats: chars={len(context)}"
                )

        # Long reading-comprehension passage without explicit Context marker.
        # Keep the tail because the actual question is usually near the end.
        if len(raw) > 900:
            return (
                "Task: reading_comprehension\n"
                f"Question/context_tail: {raw[-700:].strip()}\n"
                f"Text stats: chars={len(raw)}"
            )

        return raw

    @staticmethod
    def _strip_to_question(text: str) -> str:
        """Compatibility for old pooled embedding branches."""
        if not isinstance(text, str):
            return str(text)
        return RouterNetwork._make_routing_text(text)

    # @torch.no_grad()
    def encode_prompt_tokens(self, prompts):
        """
        Token-level prompt encoding for USE_CLIP_FUSION_ROUTER=True.

        Returns:
          prompt_tokens: [B, L, d_model]
          prompt_mask:   [B, L] bool, True = valid token
        """
        prompts = self._normalize_prompt_list(prompts)
        prompts = [self._make_routing_text(p) for p in prompts]

        if bool(getattr(Config, "ROUTER_DEBUG_TEXT", False)):
            print(f"[Routing TEXT] {prompts[:2]}")

        enc = self.prompt_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(getattr(Config, "PROMPT_MAX_TOKENS", 512)),
        )

        device = next(self.parameters()).device
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(device, non_blocking=True)

        self.prompt_encoder.eval()

        with torch.no_grad():
            out = self.prompt_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        h = out.last_hidden_state.detach()

        dtype = next(self.prompt_token_proj.parameters()).dtype
        h = h.to(dtype=dtype)

        prompt_tokens = self.prompt_token_proj(h)  # [B, L, d_model]
        prompt_mask = attention_mask.bool()

        return prompt_tokens, prompt_mask

    @torch.no_grad()
    def _encode_prompts(self, prompts):
        """Old pooled SentenceTransformer embedding path for non-CLIP branches."""
        prompts = self._normalize_prompt_list(prompts)
        prompts = [self._make_routing_text(p) for p in prompts]

        if bool(getattr(Config, "ROUTER_DEBUG_TEXT", False)):
            print(f"[Routing TEXT] {prompts[:2]}")

        emb = self.prompt_encoder.encode(
            prompts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        emb = np.asarray(emb, dtype=np.float32)
        return torch.from_numpy(emb)

    def encode_prompt(self, prompts):
        """Return pooled prompt embedding tensor [B, prompt_dim] for non-CLIP branches."""
        emb_st = self._encode_prompts(prompts)

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        emb_st = emb_st.to(device=device, dtype=dtype, non_blocking=True)

        if self.prompt_projection is None:
            return emb_st

        return self.prompt_projection(emb_st)

    # =========================================================
    # Server token builder for old USE_ATTN_ROUTER branch
    # =========================================================
    def _build_server_tokens(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B = state.shape[0]
        M = int(self.M)
        F_dyn = int(self.dyn_feat_dim)
        F_stat = int(self.stat_feat_dim)
        F_total = F_dyn + F_stat

        if state.shape[1] != M * F_total:
            raise ValueError(
                f"Flat state_dim mismatch in attention router: got {state.shape[1]}, "
                f"expected {M * F_total} (M={M}, F_dyn={F_dyn}, F_stat={F_stat})"
            )

        server_feat = state.view(B, M, F_total)
        dyn_feat = server_feat[..., :F_dyn]
        stat_feat = server_feat[..., F_dyn:]

        dyn_tok = self.server_dyn_proj(dyn_feat)
        stat_tok = self.server_stat_proj(stat_feat)

        dyn_tok = dyn_tok + self.dyn_type_emb
        stat_tok = stat_tok + self.stat_type_emb

        combined = torch.stack([dyn_tok, stat_tok], dim=2)
        server_tokens = combined.view(B, 2 * M, self.d_model)

        return server_tokens, None

    def forward(self, state, prompt, action_mask=None):
        """
        Returns:
          logits: [B, action_dim]
          value:  [B, 1]
        """
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32, device=Config.DEVICE)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        B = state.shape[0]

        if getattr(self, "use_clip_fusion", False):
            prompt_tokens, prompt_mask = self.encode_prompt_tokens(prompt)
        else:
            p = self.encode_prompt(prompt)
            if p.dim() == 1:
                p = p.unsqueeze(0)

            if p.shape[0] == 1 and B > 1:
                p = p.expand(B, -1)
            elif p.shape[0] != B:
                p = p[:1].expand(B, -1)

        # =========================================================
        # Token-level LLaVA-style fusion router
        # =========================================================
        if getattr(self, "use_clip_fusion", False):
            M = int(self.M)
            F_dyn = int(self.dyn_feat_dim)
            F_stat = int(self.stat_feat_dim)
            F_total = F_dyn + F_stat

            if state.shape[1] != M * F_total:
                raise ValueError(
                    f"state_dim mismatch in token-level fusion: got {state.shape[1]}, "
                    f"expected {M * F_total} (M={M}, F_total={F_total})"
                )

            if prompt_tokens.shape[0] == 1 and B > 1:
                prompt_tokens = prompt_tokens.expand(B, -1, -1)
                prompt_mask = prompt_mask.expand(B, -1)
            elif prompt_tokens.shape[0] != B:
                prompt_tokens = prompt_tokens[:1].expand(B, -1, -1)
                prompt_mask = prompt_mask[:1].expand(B, -1)

            # ---- Server tower ----
            sf = state.view(B, M, F_total)
            dyn_in = sf[..., :F_dyn]
            stat_in = sf[..., F_dyn:]

            dyn_tok = self.server_dyn_proj(dyn_in) + self.dyn_type_emb
            stat_tok = self.server_stat_proj(stat_in) + self.stat_type_emb

            server_tokens = torch.stack([dyn_tok, stat_tok], dim=2).view(B, 2 * M, self.d_model)
            server_tokens = server_tokens + self.server_pos_emb[:, :2 * M, :]

            for blk in self.server_self_attn:
                server_tokens = blk(server_tokens)

            # ---- LLaVA-style joint fusion ----
            # Build a single sequence:
            #   [ROUTE] + prompt tokens + server tokens
            # Prompt padding is masked; server tokens and route token are always valid.
            route_tok = self.route_token.expand(B, -1, -1) + self.route_type_emb
            prompt_tokens_f = prompt_tokens + self.prompt_type_emb
            server_tokens_f = server_tokens + self.server_type_emb

            fusion_x = torch.cat([route_tok, prompt_tokens_f, server_tokens_f], dim=1)

            route_mask = torch.ones((B, 1), device=prompt_mask.device, dtype=torch.bool)
            server_mask = torch.ones((B, 2 * M), device=prompt_mask.device, dtype=torch.bool)
            fusion_valid_mask = torch.cat([route_mask, prompt_mask, server_mask], dim=1)
            key_padding_mask = ~fusion_valid_mask

            for blk in self.fusion_blocks:
                fusion_x = blk(fusion_x, key_padding_mask=key_padding_mask)
            fusion_x = self.fusion_ln(fusion_x)

            route_h = fusion_x[:, 0, :]
            prompt_len = prompt_tokens.shape[1]
            server_tokens = fusion_x[:, 1 + prompt_len: 1 + prompt_len + 2 * M, :]

            # ---- Split dyn/stat and fuse ----
            dyn_h = server_tokens[:, 0::2, :]    # [B, M, d]
            stat_h = server_tokens[:, 1::2, :]   # [B, M, d]

            server_h = self.server_fuse(torch.cat([dyn_h, stat_h], dim=-1))  # [B, M, d]

            # ---- Actor: one logit per server ----
            # logits = self.actor_head(server_h).squeeze(-1)  # [B, M]
            route_expand = route_h.unsqueeze(1).expand(-1, M, -1)
            actor_in = torch.cat([server_h, route_expand], dim=-1)
            logits = self.actor_head(actor_in).squeeze(-1)

            # temp = torch.exp(self.actor_log_temp).clamp(min=0.1, max=10.0)
            # logits = logits / temp

            # # ---- Critic: route token summarizes prompt-server interaction ----
            # server_pool = server_h.mean(dim=1)
            # critic_in = torch.cat([route_h, server_pool], dim=-1)
            # value = self.critic_mlp(critic_in)
            # ---- Critic: route-conditioned attention pooling over servers ----
            # route_h:   [B, d]
            # server_h: [B, M, d]
            route_expand = route_h.unsqueeze(1).expand(-1, M, -1)  # [B, M, d]

            critic_score_in = torch.cat([server_h, route_expand], dim=-1)  # [B, M, 2d]
            critic_scores = self.critic_server_score(critic_score_in).squeeze(-1)  # [B, M]

            # Optional: if action_mask exists, do not pool invalid/full servers too much.
            if action_mask is not None:
                if isinstance(action_mask, np.ndarray):
                    critic_mask = torch.as_tensor(action_mask, dtype=torch.float32, device=critic_scores.device)
                else:
                    critic_mask = action_mask.to(device=critic_scores.device, dtype=torch.float32)

                if critic_mask.dim() == 1:
                    critic_mask = critic_mask.unsqueeze(0)

                if critic_mask.shape[0] == 1 and B > 1:
                    critic_mask = critic_mask.expand(B, -1)

                critic_scores = critic_scores.masked_fill(critic_mask <= 0, -1e9)

            critic_weights = torch.softmax(critic_scores, dim=1)  # [B, M]
            server_pool = torch.sum(critic_weights.unsqueeze(-1) * server_h, dim=1)  # [B, d]

            critic_in = torch.cat([route_h, server_pool], dim=-1)  # [B, 2d]
            value = self.critic_mlp(critic_in)

        elif not self.use_attn:
            if getattr(self, "use_serverwise_mlp", False):
                F_dim = int(self.server_feat_dim)
                M = int(self.M)
                if state.shape[1] != M * F_dim:
                    raise ValueError(f"Flat state_dim mismatch: got {state.shape[1]}, expected {M * F_dim} (M={M}, F={F_dim})")

                s = state.view(B, M, F_dim)
                s_norm = self.server_feat_ln(s)

                p_norm = self.prompt_ln(p)
                p_srv = p_norm.unsqueeze(1).expand(B, M, p_norm.shape[-1])

                x = torch.cat([s_norm, p_srv], dim=-1)

                h = self.server_mlp(x)
                logits = self.server_actor(h).squeeze(-1)

                h_pool = h.mean(dim=1)
                value = self.server_critic(h_pool)
            else:
                s_norm = self.state_ln(state)
                p_norm = self.prompt_ln(p)
                x = torch.cat([s_norm, p_norm], dim=-1)
                h = self.trunk(x)
                logits = self.actor(h)
                value = self.critic(h)
        else:
            prompt_tok = self.prompt_token_proj(p).unsqueeze(1)
            server_tokens, global_token = self._build_server_tokens(state)

            x = torch.cat([prompt_tok, server_tokens], dim=1)
            x = x + self.pos_emb[:, :x.shape[1], :]

            for blk in self.attn_blocks:
                x = blk(x)

            prompt_h = x[:, 0:1, :]
            server_block = x[:, 1:1 + 2 * self.M, :]
            dyn_h = server_block[:, 0::2, :]
            stat_h = server_block[:, 1::2, :]

            prompt_h_exp = prompt_h.expand(-1, self.M, -1)

            actor_in = torch.cat([prompt_h_exp, dyn_h, stat_h], dim=-1)
            actor_in = self.actor_ln(actor_in)
            logits = self.actor_head(actor_in).squeeze(-1)

            pooled = x.mean(dim=1)
            critic_in = torch.cat([prompt_h.squeeze(1), pooled], dim=-1)
            value = self.critic_mlp(critic_in)

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.as_tensor(action_mask, dtype=torch.float32, device=logits.device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            if action_mask.shape[0] == 1 and B > 1:
                action_mask = action_mask.expand(B, -1)
            logits = logits.masked_fill(action_mask <= 0, -1e9)

        return logits, value

    def get_action_and_value(self, state, prompt, action_mask=None, action=None):
        logits, value = self.forward(state, prompt, action_mask)

        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        dist_policy = dist.probs

        return action, log_prob, entropy, value, dist_policy

    def safe_probs(self, probs: torch.Tensor):
        probs = torch.clamp(probs, min=0)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(probs_sum > 0, probs / probs_sum, probs)

        if probs.dim() == 1:
            if probs.sum() <= 0:
                probs = torch.full_like(probs, 1.0 / probs.numel())
            return probs

        zero_rows = (probs_sum.squeeze(-1) == 0)
        if zero_rows.any():
            probs[zero_rows] = 1.0 / probs.shape[1]
        return probs

    def get_action_and_value_queue(self, state, state_np, prompt, action_mask=None, action=None, service_rate=None):
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
        scores = []
        M = len(Config.SERVER_CAPACITIES)

        if not getattr(Config, "MIX_QUEUE_SCORE", False):
            for i, capacity in enumerate(Config.SERVER_CAPACITIES):
                utilization = float(state[i])
                load = utilization * float(capacity)
                sr = max(float(service_rate[i]), 1e-4)
                score = sr / (load + epslon)
                scores.append(score)
            return torch.FloatTensor(scores).to(Config.DEVICE)

        for i, capacity in enumerate(Config.SERVER_CAPACACITIES if hasattr(Config, "SERVER_CAPACACITIES") else Config.SERVER_CAPACITIES):
            utilization = float(state[i])
            load = utilization * float(capacity)
            sr = max(float(service_rate[i]), 1e-4)
            queue_score = sr / (load + epslon)

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


# =========================================================
# LLM-backed router policy (optional)
# =========================================================
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None


class LLMRouterNetwork(nn.Module):
    """
    Use a HuggingFace causal LM as an encoder, and learn small actor/critic heads on top.

    Inputs (same information as your current router):
      - numeric state vector (queue / service-rate / quality / price features)
      - the user prompt text

    Output:
      - logits over servers (action_dim)
      - value estimate (critic)

    NOTE:
      - This is MUCH heavier than the MLP router. Keep ROUTER_LLM_FREEZE_BASE=True for feasibility.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is required for LLMRouterNetwork. Please install `transformers`.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.model_name = getattr(Config, "ROUTER_LLM_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
        self.use_chat_template = bool(getattr(Config, "ROUTER_LLM_USE_CHAT_TEMPLATE", True))
        self.max_input_tokens = int(getattr(Config, "ROUTER_LLM_MAX_INPUT_TOKENS", 1024))
        self.state_decimals = int(getattr(Config, "ROUTER_LLM_STATE_DECIMALS", 4))
        self.state_max_elems = int(getattr(Config, "ROUTER_LLM_STATE_MAX_ELEMS", 256))
        self.include_model_names = bool(getattr(Config, "ROUTER_LLM_INCLUDE_MODEL_NAMES", False))
        # Tuning mode:
        #   - "heads": train only small actor/critic heads (no prefix).
        #   - "prefix": train a learnable soft prefix (embedding prefix) + heads; freeze base LM.
        #   - "full": fine-tune the whole LM (NOT recommended for memory unless you use LoRA).
        self.tune_mode = str(getattr(Config, "ROUTER_LLM_TUNE_MODE", "heads")).lower()
        self.prefix_len = int(getattr(Config, "ROUTER_LLM_PREFIX_LEN", 16))
        self.prefix_init_std = float(getattr(Config, "ROUTER_LLM_PREFIX_INIT_STD", 0.02))

        # Actor/Critic adapter heads (map LM hidden -> action logits / value)
        self.actor_hidden = int(getattr(Config, "ROUTER_LLM_ACTOR_HIDDEN", 256))
        self.actor_depth = int(getattr(Config, "ROUTER_LLM_ACTOR_DEPTH", 2))
        self.actor_dropout = float(getattr(Config, "ROUTER_LLM_ACTOR_DROPOUT", 0.0))

        self.critic_hidden = int(getattr(Config, "ROUTER_LLM_CRITIC_HIDDEN", 256))
        self.critic_depth = int(getattr(Config, "ROUTER_LLM_CRITIC_DEPTH", 2))
        self.critic_dropout = float(getattr(Config, "ROUTER_LLM_CRITIC_DROPOUT", 0.0))


        # Base LM freezing behavior
        # For prefix-tuning we always freeze the base LM by default (prefix+heads only).
        if self.tune_mode == "full":
            self.freeze_base = bool(getattr(Config, "ROUTER_LLM_FREEZE_BASE", False))
        else:
            self.freeze_base = True
        self.dtype_name = str(getattr(Config, "ROUTER_LLM_DTYPE", "float16")).lower()
        self.attn_impl = getattr(Config, "ROUTER_LLM_ATTN_IMPL", None)
        self.grad_ckpt = bool(getattr(Config, "ROUTER_LLM_GRAD_CHECKPOINTING", False))

        if self.dtype_name in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif self.dtype_name in ("fp32", "float32"):
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # Left padding is often better when using last-token representations
        try:
            self.tokenizer.padding_side = "left"
        except Exception:
            pass
        if self.tokenizer.pad_token is None:
            # fall back to eos_token for padding
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Base LM (encoder)
        base_kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True)
        if self.attn_impl:
            try:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, attn_implementation=self.attn_impl, **base_kwargs
                )
            except Exception as e:
                print(f"[LLMRouterNetwork] attn_implementation={self.attn_impl} failed; falling back. ({type(e).__name__}: {e})")
                self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **base_kwargs)
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name, **base_kwargs)

        # Important for training: disable kv-cache
        try:
            self.base_model.config.use_cache = False
        except Exception:
            pass

        if self.grad_ckpt:
            try:
                self.base_model.gradient_checkpointing_enable()
            except Exception as e:
                print(f"[LLMRouterNetwork] gradient_checkpointing_enable failed: {e}")

        # Move to the same device as the router
        self.base_model.to(Config.DEVICE)
        self.base_model.train()

        hidden_size = int(getattr(self.base_model.config, "hidden_size", 4096))

        # Small heads (trainable)
        self.ln = nn.LayerNorm(hidden_size)

        # Actor: hidden -> action logits (softmax is applied later by Categorical/softmax)
        self.actor_head = make_mlp(
            in_dim=hidden_size,
            hidden_dim=self.actor_hidden,
            out_dim=self.action_dim,
            depth=self.actor_depth,
            dropout=self.actor_dropout,
        )

        # Critic: hidden -> scalar value
        self.critic_head = make_mlp(
            in_dim=hidden_size,
            hidden_dim=self.critic_hidden,
            out_dim=1,
            depth=self.critic_depth,
            dropout=self.critic_dropout,
        )

        # Prefix parameters (embedding-prefix tuning)
        self.prefix_embed = None
        if self.tune_mode == "prefix":
            if self.prefix_len <= 0:
                raise ValueError("ROUTER_LLM_PREFIX_LEN must be > 0 when ROUTER_LLM_TUNE_MODE='prefix'")
            pe = torch.randn(self.prefix_len, hidden_size) * float(self.prefix_init_std)
            self.prefix_embed = nn.Parameter(pe)  # [P, H]

        if self.freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # Always train the small heads
        for p in self.ln.parameters():
            p.requires_grad = True
        for p in self.actor_head.parameters():
            p.requires_grad = True
        for p in self.critic_head.parameters():
            p.requires_grad = True

        if self.prefix_embed is not None:
            self.prefix_embed.requires_grad = True

    def _format_state(self, s: np.ndarray) -> str:
        # s is 1D float array
        s = np.asarray(s, dtype=np.float32).flatten()
        if self.state_max_elems > 0 and s.size > self.state_max_elems:
            s = s[: self.state_max_elems]
        fmt = f"{{:.{self.state_decimals}f}}"
        parts = [f"s{i}=" + fmt.format(float(v)) for i, v in enumerate(s.tolist())]
        return " ".join(parts)

    def _build_text(self, state_row: np.ndarray, prompt: str) -> str:
        M = self.action_dim

        sys = (
            "You are a routing policy for an LLM serving cluster. "
            f"Choose the best server id in [0, {M-1}] for the given request, "
            "considering queue/service/price/quality information. "
            "Do not answer the user question; only decide the server."
        )

        mapping = ""
        if self.include_model_names:
            try:
                mapping = "Servers: " + " | ".join([f"{i}:{name}" for i, name in enumerate(Config.MODEL_NAMES)]) + "\n"
            except Exception:
                mapping = ""

        state_txt = self._format_state(state_row)
        user = (
            f"{mapping}"
            f"StateVector: {state_txt}\n"
            f"UserPrompt: {prompt}\n"
            f"Return only the server id."
        )

        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ]
                # add_generation_prompt=True appends the assistant prefix (good as "decision point")
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        return sys + "\n\n" + user + "\n\nServerId:"

    def _encode_batch(self, state: torch.Tensor, prompts) -> tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B = int(state.shape[0])

        # normalize prompts to list[str]
        if isinstance(prompts, str):
            prompts_list = [prompts] * B
        elif isinstance(prompts, list):
            if len(prompts) == 1 and B > 1:
                prompts_list = prompts * B
            else:
                prompts_list = [str(p) for p in prompts[:B]]
                if len(prompts_list) < B:
                    prompts_list = prompts_list + [prompts_list[-1]] * (B - len(prompts_list))
        else:
            prompts_list = [str(prompts)] * B

        # Move state to CPU for string formatting (cheap compared to LLM forward)
        state_cpu = state.detach().float().cpu().numpy()
        texts = [self._build_text(state_cpu[i], prompts_list[i]) for i in range(B)]

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max(1, self.max_input_tokens - (self.prefix_len if self.tune_mode == 'prefix' else 0)),
        )
        input_ids = enc["input_ids"].to(Config.DEVICE)
        attn = enc["attention_mask"].to(Config.DEVICE)
        return input_ids, attn

    def forward(self, state, prompt, action_mask=None):
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32, device=Config.DEVICE)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        input_ids, attn = self._encode_batch(state, prompt)

                # Base LM forward
        if self.tune_mode == "prefix" and self.prefix_embed is not None:
            # Build inputs_embeds with a learned prefix in embedding space (soft prefix).
            # This avoids touching attention internals and works with any CausalLM that supports inputs_embeds.
            emb = self.base_model.get_input_embeddings()(input_ids)  # [B, T, H]
            B = int(emb.shape[0])
            p = self.prefix_embed.to(device=emb.device, dtype=emb.dtype).unsqueeze(0).expand(B, -1, -1)  # [B, P, H]
            emb = torch.cat([p, emb], dim=1)  # [B, P+T, H]
            p_mask = torch.ones((B, self.prefix_len), device=attn.device, dtype=attn.dtype)
            attn2 = torch.cat([p_mask, attn], dim=1)  # [B, P+T]
            out = self.base_model(
                inputs_embeds=emb,
                attention_mask=attn2,
                output_hidden_states=True,
                return_dict=True,
            )
            attn = attn2  # use updated mask for last-token indexing
        else:
            out = self.base_model(
                input_ids=input_ids,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True,
            )

        # last hidden state: [B, T, H]
        h = out.hidden_states[-1]

        # gather last non-pad token representation per row
        last_idx = attn.sum(dim=1) - 1  # [B]
        last_idx = torch.clamp(last_idx, min=0)
        B = h.shape[0]
        h_last = h[torch.arange(B, device=h.device), last_idx]  # [B, H]

        # ---- dtype safety ----
        # Some HF models (or hidden-state paths) may return float32 hidden states
        # even when weights are fp16/bf16. LayerNorm requires input dtype to match
        # its parameters, otherwise you'll see:
        #   expected scalar type Half but found Float
        ln_dtype = self.ln.weight.dtype if hasattr(self.ln, "weight") and self.ln.weight is not None else h_last.dtype
        if h_last.dtype != ln_dtype:
            h_last = h_last.to(dtype=ln_dtype)

        h_last = self.ln(h_last)

                # Cast to head dtypes (Sequential MLP heads don't have `.weight`)
        try:
            actor_dtype = next(self.actor_head.parameters()).dtype
        except StopIteration:
            actor_dtype = h_last.dtype
        if h_last.dtype != actor_dtype:
            h_last = h_last.to(dtype=actor_dtype)

        logits = self.actor_head(h_last)          # [B, action_dim]

        try:
            critic_dtype = next(self.critic_head.parameters()).dtype
        except StopIteration:
            critic_dtype = h_last.dtype
        h_v = h_last if h_last.dtype == critic_dtype else h_last.to(dtype=critic_dtype)
        value = self.critic_head(h_v)             # [B, 1]

        # apply action mask on logits (same behavior as RouterNetwork)
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.as_tensor(action_mask, dtype=torch.float32, device=logits.device)
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            if action_mask.shape[0] == 1 and logits.shape[0] > 1:
                action_mask = action_mask.expand(logits.shape[0], -1)
            logits = logits.masked_fill(action_mask <= 0, -1e9)

        return logits, value

    def get_action_and_value(self, state, prompt, action_mask=None, action=None):
        logits, value = self.forward(state, prompt, action_mask)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        dist_policy = dist.probs
        return action, log_prob, entropy, value, dist_policy


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int):
        backbone = str(getattr(Config, "ROUTER_POLICY_BACKBONE", "mlp")).lower()
        if backbone == "llm":
            self.network = LLMRouterNetwork(state_dim, action_dim).to(Config.DEVICE)
        else:
            self.network = RouterNetwork(state_dim, action_dim).to(Config.DEVICE)
        # self.optimizer = torch.optim.Adam(
        #     self.network.parameters(),
        #     lr=Config.LEARNING_RATE
        # )
        actor_lr = float(getattr(Config, "ACTOR_LEARNING_RATE", Config.LEARNING_RATE))
        critic_lr = float(getattr(Config, "CRITIC_LEARNING_RATE", Config.LEARNING_RATE))

        critic_params = []
        actor_params = []

        for name, param in self.network.named_parameters():
            if not param.requires_grad:
                continue

            # Critic/value-specific modules
            if (
                "critic" in name
                or "value" in name
                or "server_critic" in name
                or "critic_mlp" in name
                or "critic_head" in name
            ):
                critic_params.append(param)
            else:
                actor_params.append(param)

        param_groups = []

        if actor_params:
            param_groups.append({
                "params": actor_params,
                "lr": actor_lr,
            })

        if critic_params:
            param_groups.append({
                "params": critic_params,
                "lr": critic_lr,
            })

        self.optimizer = torch.optim.Adam(
            param_groups,
            eps=1e-5,
        )
        
        # ============================================================
        # LR scheduler (episode-level decay)
        # trainer calls self.agent.scheduler.step() once per episode,
        # so the schedule unit is EPISODES.
        # LambdaLR multiplies each param group's base lr by the same
        # factor, preserving the actor:critic LR ratio.
        # ============================================================
        use_lr_decay = bool(getattr(Config, "USE_LR_DECAY", False))
        if use_lr_decay:
            decay_type = str(getattr(Config, "LR_DECAY_TYPE", "cosine")).lower()
            min_ratio = float(getattr(Config, "LR_DECAY_MIN_RATIO", 0.1))
            decay_eps = getattr(Config, "LR_DECAY_EPISODES", None)
            if decay_eps is None:
                decay_eps = int(getattr(Config, "MAX_EPISODES", 200))
            decay_eps = max(1, int(decay_eps))

            if decay_type == "linear":
                def _lr_lambda(ep):
                    progress = min(ep / decay_eps, 1.0)
                    return 1.0 - (1.0 - min_ratio) * progress
            else:  # cosine
                def _lr_lambda(ep):
                    progress = min(ep / decay_eps, 1.0)
                    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=_lr_lambda
            )
        else:
            self.scheduler = None

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

        F = int(getattr(self.network, "server_feat_dim", 5))
        sf = state_tensor.view(M, F)

        # Layout: [util, slot_count, mu, price_in, price_out]  (F=5)
        # Fallback for F=4 legacy: [util, mu, price_in, price_out]
        dyn_dim = int(getattr(Config, "SERVER_DYN_DIM", 1))
        util      = sf[:, 0].detach().cpu().numpy().astype(np.float64)
        mu        = sf[:, dyn_dim].detach().cpu().numpy().astype(np.float64)
        price_in  = sf[:, dyn_dim + 1].detach().cpu().numpy().astype(np.float64)
        price_out = sf[:, dyn_dim + 2].detach().cpu().numpy().astype(np.float64)

        cap = np.asarray(Config.SERVER_CAPACITIES, dtype=np.float64)
        q_len = util * cap
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
            # Flat *interleaved* layout per server:
            #   [util, mu, q, price_in, price_out]  => q is at offset 2 in each bundle
            coefs = self.quality_scorer.compute_quality_score_all(prompt)
            F = int(getattr(self.network, "server_feat_dim", 0))
            if F <= 0:
                # fallback to legacy assumption util+mu+q+price_in+price_out
                F = 5
            q_off = 2
            for i, server in enumerate(Config.MODEL_NAMES):
                idx = i * F + q_off
                if idx < state_tensor.numel():
                    state_tensor[idx] = float(coefs[server])
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

        if Config.MASK:
            with torch.no_grad():
                action, log_prob, entropy, value, dist_policy = self.network.get_action_and_value(
                    state_tensor, prompt, action_mask_tensor
                )
        else:
            with torch.no_grad():
                action, log_prob, entropy, value, dist_policy = self.network.get_action_and_value(
                    state_tensor, prompt, None
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
            sf = np.asarray(state, dtype=np.float32).reshape(M, 4)
            u = sf[:, 0]
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
        - interval importance weight rho_t = exp(mean_i(new_logp_i - old_logp_i))

        Added:
        - Config-controlled interval mini-batch PPO:
            USE_PER_INTERVAL_MINIBATCH
            PPO_INTERVAL_MINIBATCH_SIZE
            PPO_SHUFFLE_INTERVALS

        Important:
        - Does NOT change your interval objective.
        - Does NOT change rho formula.
        - Does NOT change value target style.
        - Does NOT change FAIR aggregation.
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

        # ============================================================
        # Build interval groups
        # ============================================================
        slot_to_indices = {}
        for i, ts in enumerate(time_slots.tolist()):
            slot_to_indices.setdefault(int(ts), []).append(i)

        arrivals_per_interval = {
            ts: len(idxs)
            for ts, idxs in sorted(slot_to_indices.items())
        }
        total_arrivals = sum(arrivals_per_interval.values())
        print(
            f"[Arrivals] intervals={len(arrivals_per_interval)} "
            f"total={total_arrivals} "
            f"per_slot={arrivals_per_interval}"
        )

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

            # Keep your current value baseline style:
            # interval old value = mean of values inside this interval.
            term_values_old.append(values[idxs].mean())

            # ========================================================
            # FAIR reward aggregation: keep your original logic
            # ========================================================
            F_frac = float(getattr(Config, "FAIR", 1.0))
            F_frac = max(0.0, min(1.0, F_frac))

            # Effective group size:
            # historically denom=M if Nt>=M else denom=Nt, i.e. G=min(Nt, M)
            G = min(Nt, M)

            if Config.FAIR_REWARD_MIN_FLOOR:
                floor = r_t.min().detach()
            else:
                floor = torch.tensor(
                    -Config.BETA - Config.REWARD_GAMMA,
                    device=Config.DEVICE,
                    dtype=r_t.dtype,
                )

            used_server_terms = []
            for m in range(M):
                mask = (a_t == m)
                if mask.any():
                    r_m = r_t[mask]
                    server_term = r_m.mean()
                    used_server_terms.append(server_term)

            if len(used_server_terms) == 0:
                continue

            K = len(used_server_terms)
            missing = max(0, G - K)
            pad = int(math.floor(F_frac * missing))

            if pad > 0:
                used_server_terms.extend([floor] * pad)

            server_terms = torch.stack(used_server_terms)

            # Tilted aggregation across used/padded server terms.
            tr = log_mean_exp(server_terms, denom=server_terms.numel())
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
                "actual_updates": 0,
                "early_stopped_kl": 0,
            }

        term_rewards = torch.stack(term_rewards)
        term_values_old = torch.stack(term_values_old)
        first_indices_t = torch.tensor(first_indices, device=Config.DEVICE, dtype=torch.long)

        # ============================================================
        # GAE on interval-level rewards
        # ============================================================
        dones = torch.zeros_like(term_rewards)
        dones[-1] = 1

        term_adv = self.compute_gae(term_rewards, term_values_old, dones=dones)
        term_ret = term_adv + term_values_old

        # Keep your single-interval fix:
        # if only one interval, do not subtract itself.
        if term_adv.numel() > 1:
            term_adv = (term_adv - term_adv.mean()) / (term_adv.std(unbiased=False) + 1e-5)

        # This mapping is not required for the current mean-value interval objective,
        # but keeping it here does not change behavior and preserves compatibility.
        step_to_interval_pos = torch.zeros(len(trajectories), device=Config.DEVICE, dtype=torch.long)
        for pos, idxs in enumerate(interval_indices):
            step_to_interval_pos[idxs] = pos

        # ============================================================
        # PPO update
        # ============================================================
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        approx_kl = torch.tensor(0.0, device=Config.DEVICE)

        actual_updates = 0
        early_stopped = False

        use_interval_mb = bool(getattr(Config, "USE_PER_INTERVAL_MINIBATCH", False))
        target_kl = float(getattr(Config, "TARGET_KL", 0.02))
        use_kl_stop = bool(getattr(Config, "USE_TARGET_KL_STOP", False))

        # Do not mutate original action_masks.
        train_action_masks = action_masks if (Config.MASK and action_masks is not None) else None

        # ============================================================
        # Path A: original full-episode PPO update
        # ============================================================
        if not use_interval_mb:
            for epoch in range(Config.PPO_EPOCHS):
                if getattr(Config, "USE_MERGE_TO_TRAIN", False):
                    _, new_log_probs, entropy, new_values, dist, _queue_scores = (
                        self.network.get_action_and_value_queue(
                            states,
                            states_np,
                            prompts,
                            train_action_masks,
                            actions,
                            service_rate=service_rate,
                        )
                    )
                else:
                    _, new_log_probs, entropy, new_values, dist = (
                        self.network.get_action_and_value(
                            states,
                            prompts,
                            train_action_masks,
                            actions,
                        )
                    )

                # ----------------------------------------------------
                # Interval-level rho_t, unchanged:
                # rho_t = exp(mean_i(new_logp_i - old_logp_i))
                # ----------------------------------------------------
                rhos = []
                for idxs in interval_indices:
                    rho_t = torch.exp((new_log_probs[idxs] - old_log_probs[idxs]).mean())
                    rhos.append(rho_t)
                rhos = torch.stack(rhos)

                surr1 = rhos * term_adv
                surr2 = torch.clamp(
                    rhos,
                    1.0 - Config.CLIP_EPSILON,
                    1.0 + Config.CLIP_EPSILON,
                ) * term_adv

                policy_loss = -torch.min(surr1, surr2).mean()

                # ----------------------------------------------------
                # Value loss, unchanged:
                # interval value = mean value inside interval
                # ----------------------------------------------------
                v_all = new_values.squeeze(-1)
                v_interval = torch.stack([
                    v_all[idxs].mean()
                    for idxs in interval_indices
                ])

                value_loss = F.mse_loss(v_interval, term_ret.detach())

                # ----------------------------------------------------
                # Entropy loss, unchanged style
                # ----------------------------------------------------
                entropy_loss = -entropy.mean()

                # ----------------------------------------------------
                # approx_kl monitor, unchanged estimator
                # ----------------------------------------------------
                with torch.no_grad():
                    log_ratio_step = new_log_probs - old_log_probs
                    ratio_step = torch.exp(log_ratio_step)
                    approx_kl_step = (ratio_step - 1.0) - log_ratio_step
                    approx_kl = torch.stack([
                        approx_kl_step[idxs].mean()
                        for idxs in interval_indices
                    ]).mean()

                if use_kl_stop and float(approx_kl.detach().cpu().item()) > target_kl:
                    print(
                        f"[PPO] early stop at epoch {epoch}: "
                        f"approx_kl={float(approx_kl.detach().cpu().item()):.6f} "
                        f"> target_kl={target_kl}"
                    )
                    early_stopped = True
                    break

                loss = (
                    float(getattr(Config, "POLICY_COEF", 1.0)) * policy_loss
                    + float(getattr(Config, "VALUE_COEF", 0.5)) * value_loss
                    + float(getattr(Config, "ENTROPY_COEF", 0.0)) * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
                self.optimizer.step()

                actual_updates += 1
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy_loss += float(entropy_loss.item())

        # ============================================================
        # Path B: interval mini-batch PPO update
        # Same objective as full-episode path, only batches intervals.
        # ============================================================
        else:
            num_intervals = len(interval_indices)

            interval_mb_size = int(getattr(Config, "PPO_INTERVAL_MINIBATCH_SIZE", num_intervals))
            interval_mb_size = max(1, min(interval_mb_size, num_intervals))

            shuffle_intervals = bool(getattr(Config, "PPO_SHUFFLE_INTERVALS", True))

            for epoch in range(Config.PPO_EPOCHS):
                if early_stopped:
                    break

                if shuffle_intervals:
                    interval_order = torch.randperm(num_intervals, device=Config.DEVICE)
                else:
                    interval_order = torch.arange(num_intervals, device=Config.DEVICE)

                for mb_start in range(0, num_intervals, interval_mb_size):
                    mb_pos = interval_order[mb_start: mb_start + interval_mb_size]
                    mb_pos_list = [int(x) for x in mb_pos.detach().cpu().tolist()]

                    # ------------------------------------------------
                    # Collect all prompt steps belonging to selected intervals.
                    # ------------------------------------------------
                    step_idx_parts = []
                    local_interval_indices = []

                    cursor = 0
                    for pos in mb_pos_list:
                        idxs = interval_indices[pos].to(device=Config.DEVICE, dtype=torch.long)
                        n_i = int(idxs.numel())

                        if n_i <= 0:
                            continue

                        step_idx_parts.append(idxs)

                        local_interval_indices.append(
                            torch.arange(
                                cursor,
                                cursor + n_i,
                                device=Config.DEVICE,
                                dtype=torch.long,
                            )
                        )

                        cursor += n_i

                    if len(step_idx_parts) == 0:
                        continue

                    step_idx = torch.cat(step_idx_parts, dim=0)
                    step_idx_cpu = step_idx.detach().cpu().tolist()

                    mb_states = states[step_idx]
                    mb_actions = actions[step_idx]
                    mb_old_log_probs = old_log_probs[step_idx]
                    mb_prompts = [prompts[i] for i in step_idx_cpu]

                    if train_action_masks is not None:
                        mb_action_masks = train_action_masks[step_idx]
                    else:
                        mb_action_masks = None

                    # ------------------------------------------------
                    # Forward only this interval mini-batch.
                    # ------------------------------------------------
                    if getattr(Config, "USE_MERGE_TO_TRAIN", False):
                        mb_states_np = states_np[step_idx_cpu]
                        mb_service_rate = [service_rate[i] for i in step_idx_cpu]

                        _, new_log_probs, entropy, new_values, dist, _queue_scores = (
                            self.network.get_action_and_value_queue(
                                mb_states,
                                mb_states_np,
                                mb_prompts,
                                mb_action_masks,
                                mb_actions,
                                service_rate=mb_service_rate,
                            )
                        )
                    else:
                        _, new_log_probs, entropy, new_values, dist = (
                            self.network.get_action_and_value(
                                mb_states,
                                mb_prompts,
                                mb_action_masks,
                                mb_actions,
                            )
                        )

                    # ------------------------------------------------
                    # Interval-level rho_t, same formula as full path:
                    # rho_t = exp(mean_i(new_logp_i - old_logp_i))
                    # ------------------------------------------------
                    rhos = []
                    for local_idxs in local_interval_indices:
                        rho_t = torch.exp(
                            (new_log_probs[local_idxs] - mb_old_log_probs[local_idxs]).mean()
                        )
                        rhos.append(rho_t)

                    rhos = torch.stack(rhos)

                    mb_term_adv = term_adv[mb_pos]
                    mb_term_ret = term_ret[mb_pos]

                    surr1 = rhos * mb_term_adv
                    surr2 = torch.clamp(
                        rhos,
                        1.0 - Config.CLIP_EPSILON,
                        1.0 + Config.CLIP_EPSILON,
                    ) * mb_term_adv

                    policy_loss = -torch.min(surr1, surr2).mean()

                    # ------------------------------------------------
                    # Value loss, same logic as full path:
                    # interval value = mean value inside interval.
                    # ------------------------------------------------
                    v_all = new_values.squeeze(-1)

                    v_interval = torch.stack([
                        v_all[local_idxs].mean()
                        for local_idxs in local_interval_indices
                    ])

                    value_loss = F.mse_loss(v_interval, mb_term_ret.detach())

                    # ------------------------------------------------
                    # Entropy loss, same style as full path.
                    # ------------------------------------------------
                    entropy_loss = -entropy.mean()

                    # ------------------------------------------------
                    # approx_kl monitor, same estimator as full path.
                    # ------------------------------------------------
                    with torch.no_grad():
                        kl_terms = []
                        for local_idxs in local_interval_indices:
                            log_ratio_step = new_log_probs[local_idxs] - mb_old_log_probs[local_idxs]
                            ratio_step = torch.exp(log_ratio_step)
                            approx_kl_step = (ratio_step - 1.0) - log_ratio_step
                            kl_terms.append(approx_kl_step.mean())

                        approx_kl = torch.stack(kl_terms).mean()

                    if use_kl_stop and float(approx_kl.detach().cpu().item()) > target_kl:
                        print(
                            f"[PPO] early stop at epoch {epoch}, mb_start={mb_start}: "
                            f"approx_kl={float(approx_kl.detach().cpu().item()):.6f} "
                            f"> target_kl={target_kl}"
                        )
                        early_stopped = True
                        break

                    loss = (
                        float(getattr(Config, "POLICY_COEF", 1.0)) * policy_loss
                        + float(getattr(Config, "VALUE_COEF", 0.5)) * value_loss
                        + float(getattr(Config, "ENTROPY_COEF", 0.0)) * entropy_loss
                    )

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
                    self.optimizer.step()

                    actual_updates += 1
                    total_policy_loss += float(policy_loss.item())
                    total_value_loss += float(value_loss.item())
                    total_entropy_loss += float(entropy_loss.item())

        # ============================================================
        # Metrics
        # ============================================================
        term_return_trajectory = float(self.cumulated_return(term_rewards)[0].item())
        return_trajectory = float(self.cumulated_return(rewards)[0].item())

        avg_rewards_t = (
            torch.stack(avg_rewards)
            if len(avg_rewards)
            else torch.tensor(0.0, device=Config.DEVICE)
        )
        avg_rewards_returns = (
            float(self.cumulated_return(avg_rewards_t)[0].item())
            if avg_rewards_t.numel() > 0
            else 0.0
        )

        actions_list = actions_np.tolist()

        server_usage_percentage = {m: 0.0 for m in range(M)}
        for a in actions_list:
            server_usage_percentage[int(a)] += 1.0

        for m in server_usage_percentage:
            server_usage_percentage[m] /= max(len(actions_list), 1)

        probs_usage = np.array(
            [server_usage_percentage[m] for m in range(M)],
            dtype=np.float64,
        )
        ent_usage = float(-(probs_usage * np.log(probs_usage + 1e-12)).sum())

        route_dist = {
            i: int(sum(1 for a in actions_list if a == i))
            for i in range(M)
        }

        mean_min_reward = (
            float(torch.stack(min_rewards).mean().item())
            if len(min_rewards)
            else 0.0
        )

        den = max(actual_updates, 1)

        return {
            "policy_loss": total_policy_loss / den,
            "value_loss": total_value_loss / den,
            "entropy_loss": total_entropy_loss / den,
            "rewards_returns": return_trajectory,
            "term_rewards_returns": term_return_trajectory,
            "min_rewards": mean_min_reward,
            "server_usage_percentage": server_usage_percentage,
            "cumulated_avg_rewards": avg_rewards_returns,
            "route distribution": route_dist,
            "entropy of route distribution": ent_usage,
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "actual_updates": actual_updates,
            "early_stopped_kl": int(early_stopped),
        }

    # def update_new(self, trajectories):
        """
        PPO update on ACTIVE intervals only (N_t > 0), with:
        - fair reward normalization: 1/M if N_t >= M, else 1/N_t
        - interval importance weight rho_t = exp(mean_i (new_logp_i - old_logp_i))
        
        Bug fixes applied:
        [FIX #1] Single-interval advantage no longer zeroed by mean subtraction
        [FIX #2] Entropy aggregated per-interval to match policy_loss scale
        [FIX #3] Critic trained on ALL steps (broadcast interval return)
        [FIX #4] approx_kl uses Schulman v3 estimator (always >= 0)
        [NOTE ] KL term removed from loss (approx_kl is monitoring only now)
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
            
        arrivals_per_interval = {ts: len(idxs) for ts, idxs in sorted(slot_to_indices.items())}
        total_arrivals = sum(arrivals_per_interval.values())
        print(f"[Arrivals] intervals={len(arrivals_per_interval)} "
            f"total={total_arrivals} "
            f"per_slot={arrivals_per_interval}")

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
            # term_values_old.append(values[idxs[0]])
            # Use interval-level value baseline.
            # Since all prompts in the same interval share the same telemetry state s_t,
            # average prompt-conditioned values to reduce dependence on the first prompt.
            term_values_old.append(values[idxs].mean())

            # Fair reward with controllable padding
            F_frac = float(getattr(Config, "FAIR", 1.0))
            F_frac = max(0.0, min(1.0, F_frac))

            # effective group size: historically you used denom=M if Nt>=M else denom=Nt
            # i.e. G = min(Nt, M)
            G = min(Nt, M)

            if Config.FAIR_REWARD_MIN_FLOOR:
                floor = r_t.min().detach()
            else:
                floor = torch.tensor(
                            - Config.BETA - Config.REWARD_GAMMA,
                            # -1,
                            device=Config.DEVICE,
                            dtype=r_t.dtype
                        )

            # collect mean reward for each USED server (deterministic order by server id)
            used_server_terms = []

            for m in range(M):
                mask = (a_t == m)

                if mask.any():
                    r_m = r_t[mask]  # rewards of prompts routed to server m

                    # Old:
                    server_term = r_m.mean()

                    # New:
                    # tilted aggregation inside this server, using the same Config.T
                    # server_term = log_mean_exp(r_m, denom=r_m.numel())

                    used_server_terms.append(server_term)

            if len(used_server_terms) == 0:
                continue

            K = len(used_server_terms)
            missing = max(0, G - K)
            pad = int(math.floor(F_frac * missing))

            if pad > 0:
                used_server_terms.extend([floor] * pad)

            server_terms = torch.stack(used_server_terms)

            # Tilted aggregation across servers, also using the same Config.T
            tr = log_mean_exp(server_terms, denom=server_terms.numel())
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

        # ====================================================================
        # [FIX #1] Single-interval advantage zeroing bug
        # --------------------------------------------------------------------
        # OLD code:
        #   if term_adv.numel() > 1:
        #       term_adv = (term_adv - term_adv.mean()) / (term_adv.std(unbiased=False) + 1e-5)
        #   else:
        #       term_adv = term_adv - term_adv.mean()   # <-- BUG: single element = 0
        #
        # When EPISODE_TIME_INTERVAL / INTERVAL_LENGTH == 1, there is only ONE
        # interval per episode, so term_adv.numel() == 1. Old "else" branch
        # subtracted the mean from itself, making advantage exactly 0, which
        # makes policy_loss always 0 -> policy never updates.
        # ====================================================================
        if term_adv.numel() > 1:
            term_adv = (term_adv - term_adv.mean()) / (term_adv.std(unbiased=False) + 1e-5)
        # else: keep term_adv as-is (do NOT subtract mean when only 1 element)
        # NOTE: Still recommend EPISODE_TIME_INTERVAL >> INTERVAL_LENGTH so we
        # have multiple intervals for meaningful advantage normalization.

        # ====================================================================
        # [FIX #3 - Prep] Build per-step -> interval-position mapping.
        # Used so critic can regress every step's value to its interval return.
        # ====================================================================
        step_to_interval_pos = torch.zeros(len(trajectories), device=Config.DEVICE, dtype=torch.long)
        for pos, idxs in enumerate(interval_indices):
            step_to_interval_pos[idxs] = pos

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        approx_kl = torch.tensor(0.0, device=Config.DEVICE)
        
        actual_updates = 0

        for epoch in range(Config.PPO_EPOCHS):
            if getattr(Config, "USE_MERGE_TO_TRAIN", False):
                _, new_log_probs, entropy, new_values, dist, _queue_scores = self.network.get_action_and_value_queue(
                    states, states_np, prompts, action_masks, actions, service_rate=service_rate
                )
            else:
                if not Config.MASK:
                    action_masks = None
                _, new_log_probs, entropy, new_values, dist = self.network.get_action_and_value(
                    states, prompts, action_masks, actions
                )

            # ---- Interval-level rho_t (your method, unchanged) ----
            rhos = []
            for idxs in interval_indices:
                rho_t = torch.exp((new_log_probs[idxs] - old_log_probs[idxs]).mean())
                rhos.append(rho_t)
            rhos = torch.stack(rhos)

            surr1 = rhos * term_adv
            surr2 = torch.clamp(rhos, 1.0 - Config.CLIP_EPSILON, 1.0 + Config.CLIP_EPSILON) * term_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # # ---- per-step ----
            # step_adv = torch.zeros(len(trajectories), device=Config.DEVICE)
            # for pos, idxs in enumerate(interval_indices):
            #     step_adv[idxs] = term_adv[pos]  

            # rho_step = torch.exp(new_log_probs - old_log_probs)   # [N_total]

            # surr1 = rho_step * step_adv
            # surr2 = torch.clamp(rho_step, 1.0 - Config.CLIP_EPSILON, 1.0 + Config.CLIP_EPSILON) * step_adv
            # policy_loss = -torch.min(surr1, surr2).mean()

            # ================================================================
            # [FIX #3] Critic trained on ALL steps, not just interval-first steps
            # ----------------------------------------------------------------
            # OLD code:
            v_pred = new_values.squeeze(-1)[first_indices_t]
            value_loss = F.mse_loss(v_pred, term_ret.detach())
            #
            # Old version only used 1 step per interval as critic target,
            # wasting ~99% of value predictions. Since interval state is frozen
            # but prompts differ, V(s_t, prompt_i) can still learn prompt
            # difficulty. We regress every step's value to its interval's
            # return (broadcast). Semantics preserved: target is still the
            # interval-level return.
            # ================================================================
            # step_ret = term_ret[step_to_interval_pos]                 # [N_total]
            # v_pred_all = new_values.squeeze(-1)                       # [N_total]
            # value_loss = F.mse_loss(v_pred_all, step_ret.detach())
            
            # v_pred = new_values.squeeze(-1)[first_indices_t]
            # value_loss = F.mse_loss(v_pred, term_ret.detach())
            v_all = new_values.squeeze(-1)  # [N_total]

            v_interval = torch.stack([
                v_all[idxs].mean()
                for idxs in interval_indices
            ])  # [num_intervals]

            value_loss = F.mse_loss(v_interval, term_ret.detach())

            # ================================================================
            # [FIX #2] Entropy scale aligned with policy_loss (per-interval mean)
            # ----------------------------------------------------------------
            # OLD code:
            #   entropy_loss = -entropy.mean()   # mean over N_total steps
            #
            # policy_loss averages over N_intervals, so old entropy_loss was
            # effectively diluted by avg_steps_per_interval (often 20-100x).
            # With ENTROPY_COEF=0.001, effective entropy pressure was ~1e-5,
            # far too weak to prevent collapse.
            # ================================================================
            # entropy_per_interval = torch.stack(
            #     [entropy[idxs].mean() for idxs in interval_indices]
            # )
            # entropy_loss = -entropy_per_interval.mean()
            entropy_loss = -entropy.mean()   # mean over N_total steps

            # ================================================================
            # [FIX #4] approx_kl uses Schulman v3 estimator (always >= 0)
            # ----------------------------------------------------------------
            # OLD code:
            #   kls = []
            #   for idxs in interval_indices:
            #       kl_t = (old_log_probs[idxs] - new_log_probs[idxs]).mean()
            #       kls.append(kl_t)
            #   approx_kl = torch.stack(kls).mean()
            #   approx_kl = torch.clamp(approx_kl, min=0.0)
            #
            # Old used (old - new).mean() which is 1st-order KL approximation.
            # It can go negative due to finite-sample noise; clamping to 0
            # hides the issue. Use Schulman v3: ((ratio - 1) - log_ratio),
            # always >= 0 and unbiased. Compute under no_grad (monitoring only).
            # ================================================================
            with torch.no_grad():
                log_ratio_step = new_log_probs - old_log_probs
                ratio_step = torch.exp(log_ratio_step)
                approx_kl_step = (ratio_step - 1.0) - log_ratio_step
                approx_kl = torch.stack(
                    [approx_kl_step[idxs].mean() for idxs in interval_indices]
                ).mean()

            # ================================================================
            # [NOTE] KL_COEF * approx_kl removed from loss.
            # Reason: approx_kl is now no_grad (monitoring only). PPO already
            # controls trust region via the clip on rho_t. If you want an
            # explicit KL penalty, build a differentiable KL separately.
            # ================================================================
            loss = (
                float(getattr(Config, "POLICY_COEF", 1.0)) * policy_loss
                + float(getattr(Config, "VALUE_COEF", 0.5)) * value_loss
                + float(getattr(Config, "ENTROPY_COEF", 0.0)) * entropy_loss
            )
            
            if bool(getattr(Config, "USE_TARGET_KL_STOP", False)):
                target_kl = float(getattr(Config, "TARGET_KL", 0.02))
                if float(approx_kl.detach().cpu().item()) > target_kl:
                    print(
                        f"[PPO] early stop at epoch {epoch}: "
                        f"approx_kl={float(approx_kl.detach().cpu().item()):.6f} > target_kl={target_kl}"
                    )
                    break
                
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
            self.optimizer.step()
            
            actual_updates += 1
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

        den = max(actual_updates, 1)
        return {
            "policy_loss": total_policy_loss / den,
            "value_loss": total_value_loss / den,
            "entropy_loss": total_entropy_loss / den,
            "rewards_returns": return_trajectory,
            "term_rewards_returns": term_return_trajectory,
            "min_rewards": mean_min_reward,
            "server_usage_percentage": server_usage_percentage,
            "cumulated_avg_rewards": avg_rewards_returns,
            "route distribution": route_dist,
            "entropy of route distribution": ent_usage,
            "approx_kl": float(approx_kl.detach().cpu().item()),
            "actual_updates": actual_updates,
            "early_stopped_kl": int(actual_updates < Config.PPO_EPOCHS),
        }

    # def update_new(self, trajectories):
    #     """
    #     PPO update on ACTIVE intervals only (N_t > 0), with:
    #       - fair reward normalization: 1/M if N_t >= M, else 1/N_t
    #       - interval importance weight rho_t = exp(mean_i (new_logp_i - old_logp_i))
    #     """
    #     states_np = np.array([t["state"] for t in trajectories], dtype=np.float32)
    #     actions_np = np.array([t["action"] for t in trajectories], dtype=np.int64)
    #     old_log_probs_np = np.array([t["log_prob"] for t in trajectories], dtype=np.float32)
    #     rewards_np = np.array([t["reward"] for t in trajectories], dtype=np.float32)
    #     values_np = np.array([t["value"] for t in trajectories], dtype=np.float32)

    #     prompts = [t["prompt"] for t in trajectories]
    #     service_rate = [t["service_rate"] for t in trajectories]
    #     time_slots_np = np.array([t["time_slot"] for t in trajectories], dtype=np.int64)

    #     states = torch.as_tensor(states_np, device=Config.DEVICE)
    #     actions = torch.as_tensor(actions_np, device=Config.DEVICE)
    #     old_log_probs = torch.as_tensor(old_log_probs_np, device=Config.DEVICE)
    #     rewards = torch.as_tensor(rewards_np, device=Config.DEVICE)
    #     values = torch.as_tensor(values_np, device=Config.DEVICE)
    #     time_slots = torch.as_tensor(time_slots_np, device=Config.DEVICE)

    #     action_masks = None
    #     if "action_mask" in trajectories[0] and trajectories[0]["action_mask"] is not None:
    #         action_masks_np = np.array([t["action_mask"] for t in trajectories], dtype=np.float32)
    #         action_masks = torch.as_tensor(action_masks_np, device=Config.DEVICE)

    #     M = len(Config.MODEL_NAMES)
    #     beta = float(Config.T)

    #     def log_mean_exp(x: torch.Tensor, denom: int):
    #         denom = max(int(denom), 1)
    #         if abs(beta) < 1e-8:
    #             return x.mean()
    #         return (torch.logsumexp(beta * x, dim=0) - math.log(denom)) / beta

    #     slot_to_indices = {}
    #     for i, ts in enumerate(time_slots.tolist()):
    #         slot_to_indices.setdefault(int(ts), []).append(i)

    #     interval_indices = []
    #     first_indices = []
    #     term_rewards = []
    #     term_values_old = []
    #     avg_rewards = []
    #     min_rewards = []

    #     for ts in sorted(slot_to_indices.keys()):
    #         idxs = torch.tensor(slot_to_indices[ts], device=Config.DEVICE, dtype=torch.long)
    #         Nt = int(idxs.numel())
    #         if Nt == 0:
    #             continue

    #         interval_indices.append(idxs)
    #         first_indices.append(int(idxs[0].item()))

    #         r_t = rewards[idxs]
    #         a_t = actions[idxs]

    #         avg_rewards.append(r_t.mean())
    #         min_rewards.append(r_t.min())
    #         term_values_old.append(values[idxs[0]])

    #         # server_means = []
    #         # if Nt >= M:
    #         #     floor = r_t.min().detach()
    #         #     for m in range(M):
    #         #         mask = (a_t == m)
    #         #         if mask.any():
    #         #             server_means.append(r_t[mask].mean())
    #         #         else:
    #         #             server_means.append(floor)
    #         #     server_means = torch.stack(server_means)
    #         #     tr = log_mean_exp(server_means, denom=M)
    #         # else:
    #         #     N = Nt
    #         #     floor = r_t.min().detach()
    #         #     for m in range(M):
    #         #         mask = (a_t == m)
    #         #         if mask.any():
    #         #             server_means.append(r_t[mask].mean())
    #         #             N -= 1
    #         #         elif N > 0:
    #         #             server_means.append(floor)
    #         #             N -= 1
    #         #     if len(server_means) == 0:
    #         #         continue
    #         #     server_means = torch.stack(server_means)
    #         #     tr = log_mean_exp(server_means, denom=Nt)

    #         # term_rewards.append(tr)

    #         # Fair reward with controllable padding
    #         F_frac = float(getattr(Config, "FAIR", 1.0))
    #         F_frac = max(0.0, min(1.0, F_frac))
            
    #         # effective group size: historically you used denom=M if Nt>=M else denom=Nt
    #         # i.e. G = min(Nt, M)
    #         G = min(Nt, M)

    #         if Config.FAIR_REWARD_MIN_FLOOR:
    #             floor = r_t.min().detach()
    #         else:
    #             floor = torch.tensor(
    #                         - Config.BETA - Config.REWARD_GAMMA,
    #                         device=Config.DEVICE,
    #                         dtype=r_t.dtype
    #                     )
            
    #         # collect mean reward for each USED server (deterministic order by server id)
    #         used_means = []
    #         for m in range(M):
    #             mask = (a_t == m)
    #             if mask.any():
    #                 used_means.append(r_t[mask].mean())
            
    #         if len(used_means) == 0:
    #             continue  # should not happen if Nt>0, but keep safe
            
    #         K = len(used_means)                 # number of used servers
    #         missing = max(0, G - K)             # how many "slots" are missing
    #         pad = int(math.floor(F_frac * missing))
            
    #         # add floor padding (P terms)
    #         if pad > 0:
    #             used_means.extend([floor] * pad)
            
    #         server_means = torch.stack(used_means)          # length = K + pad
    #         tr = log_mean_exp(server_means, denom=server_means.numel())
    #         term_rewards.append(tr)

    #     if len(term_rewards) == 0:
    #         return {
    #             "policy_loss": 0.0,
    #             "value_loss": 0.0,
    #             "entropy_loss": 0.0,
    #             "rewards_returns": float(self.cumulated_return(rewards)[0].item()) if rewards.numel() > 0 else 0.0,
    #             "term_rewards_returns": 0.0,
    #             "min_rewards": 0.0,
    #             "server_usage_percentage": {m: 0.0 for m in range(M)},
    #             "cumulated_avg_rewards": 0.0,
    #             "route distribution": {i: 0 for i in range(M)},
    #             "entropy of route distribution": 0.0,
    #             "approx_kl": 0.0,
    #         }

    #     term_rewards = torch.stack(term_rewards)
    #     term_values_old = torch.stack(term_values_old)
    #     first_indices_t = torch.tensor(first_indices, device=Config.DEVICE, dtype=torch.long)

    #     dones = torch.zeros_like(term_rewards)
    #     dones[-1] = 1
    #     term_adv = self.compute_gae(term_rewards, term_values_old, dones=dones)
    #     term_ret = term_adv + term_values_old

    #     if term_adv.numel() > 1:
    #         term_adv = (term_adv - term_adv.mean()) / (term_adv.std(unbiased=False) + 1e-5)
    #     else:
    #         term_adv = term_adv - term_adv.mean() 

    #     total_policy_loss = 0.0
    #     total_value_loss = 0.0
    #     total_entropy_loss = 0.0
    #     approx_kl = torch.tensor(0.0, device=Config.DEVICE)

    #     for _ in range(Config.PPO_EPOCHS):
    #         if getattr(Config, "USE_MERGE_TO_TRAIN", False):
    #             _, new_log_probs, entropy, new_values, dist, _queue_scores = self.network.get_action_and_value_queue(
    #                 states, states_np, prompts, action_masks, actions, service_rate=service_rate
    #             )
    #         else:
    #             if not Config.MASK:
    #                 action_masks = None
    #             _, new_log_probs, entropy, new_values, dist = self.network.get_action_and_value(
    #                 states, prompts, action_masks, actions
    #             )

    #         rhos = []
    #         for idxs in interval_indices:
    #             rho_t = torch.exp((new_log_probs[idxs] - old_log_probs[idxs]).mean())
    #             rhos.append(rho_t)
    #         rhos = torch.stack(rhos)

    #         surr1 = rhos * term_adv
    #         surr2 = torch.clamp(rhos, 1.0 - Config.CLIP_EPSILON, 1.0 + Config.CLIP_EPSILON) * term_adv
    #         policy_loss = -torch.min(surr1, surr2).mean()

    #         v_pred = new_values.squeeze(-1)[first_indices_t]
    #         value_loss = F.mse_loss(v_pred, term_ret.detach())
    #         entropy_loss = -entropy.mean()

    #         kls = []
    #         for idxs in interval_indices:
    #             kl_t = (old_log_probs[idxs] - new_log_probs[idxs]).mean()
    #             kls.append(kl_t)
    #         approx_kl = torch.stack(kls).mean()
    #         approx_kl = torch.clamp(approx_kl, min=0.0)

    #         loss = (
    #             float(getattr(Config, "POLICY_COEF", 1.0)) * policy_loss
    #             + float(getattr(Config, "VALUE_COEF", 0.5)) * value_loss
    #             + float(getattr(Config, "ENTROPY_COEF", 0.0)) * entropy_loss
    #             + float(getattr(Config, "KL_COEF", 0.0)) * approx_kl
    #         )

    #         self.optimizer.zero_grad(set_to_none=True)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
    #         self.optimizer.step()

    #         total_policy_loss += float(policy_loss.item())
    #         total_value_loss += float(value_loss.item())
    #         total_entropy_loss += float(entropy_loss.item())

    #     term_return_trajectory = float(self.cumulated_return(term_rewards)[0].item())
    #     return_trajectory = float(self.cumulated_return(rewards)[0].item())

    #     avg_rewards_t = torch.stack(avg_rewards) if len(avg_rewards) else torch.tensor(0.0, device=Config.DEVICE)
    #     avg_rewards_returns = float(self.cumulated_return(avg_rewards_t)[0].item()) if avg_rewards_t.numel() > 0 else 0.0

    #     actions_list = actions_np.tolist()
    #     server_usage_percentage = {m: 0.0 for m in range(M)}
    #     for a in actions_list:
    #         server_usage_percentage[int(a)] += 1.0
    #     for m in server_usage_percentage:
    #         server_usage_percentage[m] /= max(len(actions_list), 1)

    #     probs_usage = np.array([server_usage_percentage[m] for m in range(M)], dtype=np.float64)
    #     ent_usage = float(-(probs_usage * np.log(probs_usage + 1e-12)).sum())

    #     route_dist = {i: int(sum(1 for a in actions_list if a == i)) for i in range(M)}
    #     mean_min_reward = float(torch.stack(min_rewards).mean().item()) if len(min_rewards) else 0.0

    #     return {
    #         "policy_loss": total_policy_loss / Config.PPO_EPOCHS,
    #         "value_loss": total_value_loss / Config.PPO_EPOCHS,
    #         "entropy_loss": total_entropy_loss / Config.PPO_EPOCHS,
    #         "rewards_returns": return_trajectory,
    #         "term_rewards_returns": term_return_trajectory,
    #         "min_rewards": mean_min_reward,
    #         "server_usage_percentage": server_usage_percentage,
    #         "cumulated_avg_rewards": avg_rewards_returns,
    #         "route distribution": route_dist,
    #         "entropy of route distribution": ent_usage,
    #         "approx_kl": float(approx_kl.detach().cpu().item()),
    #     }

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

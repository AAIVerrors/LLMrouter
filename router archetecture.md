┌─────────────────────────┐         ┌─────────────────────────────────┐
│   Prompt Tower          │         │   Server Tower                  │
│                         │         │                                 │
│ prompt_emb (P)          │         │ [util, mu, p_in, p_out] per srv │
│   │                     │         │   │                             │
│   ▼ MLP (P→d)           │         │   ▼ dual-channel proj           │
│ prompt token [B,1,d]    │         │ [dyn_i, stat_i] tokens [B,2M,d] │
│                         │         │   │                             │
│                         │         │   ▼ self-attention × L          │
│                         │         │ servers see each other          │
│                         │         │   (relative load, alternatives) │
└──────────┬──────────────┘         └──────────────┬──────────────────┘
           │                                       │
           │   ┌───────────────────────────────────┘
           ▼   ▼
     ┌─────────────────────────────┐
     │ Cross-Attention             │
     │   Q = prompt token          │
     │   K, V = server tokens      │
     │   (prompt asks "who fits?") │
     └────────────┬────────────────┘
                  │
                  ▼
        ┌──────────────────────┐
        │ Actor: dot-product   │   logits[i] = ⟨q*, k_i⟩ / τ
        │ (CLIP-style scoring) │
        └──────────────────────┘
        ┌──────────────────────┐
        │ Critic: pooled MLP   │   V(s, prompt)
        └──────────────────────┘
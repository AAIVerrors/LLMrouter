import torch

class Config:
    # ================================================================
    # 6-way ALL NON-REASONING fleet, DESIGNED for prompt-dependent routing.
    # Two structural properties (vs a monotone price ladder) let a prompt-aware
    # router reach quality a prompt-blind P2C/JSQ cannot:
    #   * The per-task winner CHANGES with the task. Measured on n=24 per
    #     endpoint x task: math_hard -> Llama 70B (0.917), multi-hop QA ->
    #     GPT-4.1 mini (0.478). That rank crossover is invisible to a
    #     queue-only baseline.
    #     CAVEAT after swapping Ministral 14B out on 2026-07-26 for
    #     Qwen3.5-9B (Mistral-side timeouts): 14B was the sole MMLU-Pro
    #     specialist (0.750 there, worst at math), and Qwen3.5-9B scores 0.583
    #     on MMLU-Pro, so GPT-4.1 mini (0.750) now wins BOTH MMLU-Pro and
    #     multi-hop QA and the fleet has two distinct winners rather than
    #     three. The crossover is weaker; re-run bench_quality_matrix.py before
    #     making any claim that rests on it.
    #   * idx 4 gpt-4.1-mini is strong AND mid-speed -> decouples quality from
    #     latency, breaking P2C's "avoid long queue == avoid slow/expensive"
    #     implicit optimization.
    # Reasoning models are excluded on purpose: they return content="" while
    # spending the whole token budget on a hidden reasoning field, so their
    # quality score would be identically zero (verified on Qwen3.5-9B and
    # Gemma 3n). Keep MODEL_NAMES, PRICE, SERVICE_RATE and all per-server
    # arrays in the SAME order: the router action is the list index.
    # ================================================================
    # Trimmed 8 -> 6 on 2026-07-26 (dropped entries commented out in place, in
    # every per-server array, so the fleet can be restored by uncommenting).
    # Gemma 3n E4B and Ministral 8B together carried only 14% of fleet
    # throughput (0.225 of 1.563 req/s) and neither wins any task type, so
    # each was a strictly weaker copy of a retained endpoint. All three
    # per-task winners survive (math -> Llama 70B, MMLU-Pro -> Ministral 14B,
    # multi-hop QA -> GPT-4.1 mini), which is what the prompt-aware claim
    # rests on. Ministral 14B is kept despite having the lowest mu in the
    # fleet: it is the only MMLU-Pro specialist (best there, worst at math),
    # and without it GPT-4.1 mini wins two of three tasks and the crossover
    # collapses to two winners.
    MODEL_NAMES = [
        "ministral-3b-2512",                                     # 0 Mistral   cheap/fast floor
        # "together/google/gemma-3n-E4B-it",                      #   Together  DROPPED: slowest (mu 0.125), wins nothing
        "ministral-8b-2512",                                     # 1 Mistral   cheap output ($0.15 vs Small's $0.60)
        "gpt-4.1-nano-2025-04-14",                               # 2 OpenAI    fastest endpoint
        # "ministral-8b-2512",                                    #   Mistral   DROPPED: mu 0.100, dominated by Mistral Small
        "mistral-small-2506",                                    # 3 Mistral   mid general (non-hybrid)
        "gpt-4.1-mini",                                          # 4 OpenAI    multi-hop QA winner
        # "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",     #   Together  DROPPED: mu 0.1456, sole source of the 4.6x spread
    ]

    PRICE = [
        (0.00000010, 0.00000010),   # 0 Ministral 3B:      $0.10 / $0.10 per 1M tokens
        # (0.00000006, 0.00000012), #   Gemma 3n E4B:      $0.06 / $0.12 per 1M tokens  (DROPPED)
        (0.00000015, 0.00000015),   # 1 Ministral 8B:      $0.15 / $0.15 per 1M tokens
        # (0.00000017, 0.00000025), #   Qwen3.5-9B:        $0.17 / $0.25 per 1M tokens  (REPLACED)
        # (0.00000020, 0.00000020), #   Ministral 14B:     $0.20 / $0.20 per 1M tokens  (REPLACED)
        (0.00000010, 0.00000040),   # 2 GPT-4.1 nano:      $0.10 / $0.40 per 1M tokens
        # (0.00000015, 0.00000015), #   Ministral 8B:      $0.15 / $0.15 per 1M tokens  (DROPPED)
        (0.00000015, 0.00000060),   # 3 Mistral Small:     $0.15 / $0.60 per 1M tokens
        (0.00000040, 0.00000160),   # 4 GPT-4.1 mini:      $0.40 / $1.60 per 1M tokens
        # (0.00000104, 0.00000104), #   Llama 3.3 70B:     $1.04 / $1.04 per 1M tokens  (DROPPED)
    ]

    # Effective requests/second per endpoint, i.e. 1 / mean service time. Each
    # endpoint is a single-threaded worker process draining one FIFO queue
    # (see ServerProcess), so this serial measurement is the right notion of
    # capacity; SERVER_CAPACITIES is the queue buffer size, NOT a concurrency
    # limit. Do not substitute a figure taken from a live run's online EMA
    # under concurrent load -- that measures a different quantity and reads
    # roughly 1.7x higher, which would silently set an arrival rate above the
    # stability limit.
    #
    # This list is ALSO the frozen fairness reference (see QUOTA_USE_FROZEN_MU):
    # the online EMA keeps adapting inside a run and feeds the state, but the
    # quota is computed against these fixed values so the fairness target does
    # not drift with the workload being routed. Re-measure and update here if
    # the fleet changes; do not expect a run to correct it for you.
    #
    # Measured 2026-07-26 by bench_fleet_service_rate.py on the 6-endpoint
    # fleet at GEN_MAX_NEW_TOKENS=1024 with MCQ_COT on. For reference the same
    # fleet measures 1.338 req/s at 2048 and the pre-CoT 8-endpoint fleet
    # measured 3.639 at 512: generation length dominates fleet throughput far
    # more than fleet size does. Truncation from the 1024 cap is uneven, which
    # is why this is re-measured rather than rescaled -- Ministral 14B gains
    # 42% (its long math answers were the most truncated) while GPT-4.1 nano
    # is unchanged (it never reached the cap). Caveat: n=10 per endpoint and
    # the latency tail is heavy (Llama 70B p90 = 20 s), so individual entries
    # carry roughly +/-15% noise; the Llama figure moving down from 0.1730 is
    # within that band, not a real regression.
    # Measured 2026-07-26, n=30 per endpoint, all six in one batch, and the
    # first measurement taken with every precondition correct at once:
    # MATH_BRIEF_REASONING on, MCQ_COT on, and reasoning={"enabled": False}
    # passed to Together (bench_fleet_service_rate.py used to omit it, which
    # made a hybrid-thinking model spend its whole budget on a hidden field
    # and read 3x too slow). Earlier numbers in this file's history each
    # violated at least one of those and should not be compared against.
    #
    # The brief math prompt is what moved the fleet from 1.756 to 2.293 req/s:
    # every endpoint gained 27-67% as median completion tokens fell from
    # 330-420 to 229-332.
    #
    # Slot 1 measured separately on 2026-07-26 under the same protocol (n=30,
    # sequential, current prompt config) after replacing Qwen3.5-9B: 0.3052
    # req/s, 3.28 s mean, 113 median completion tokens, 30/30 with no failures.
    # Qwen measured 0.1022 there (three n=30 runs: 0.1587 before the brief math
    # prompt, then 0.0884 and 0.1159), i.e. 3x slower while emitting 342 tokens,
    # and it was what pushed fleet heterogeneity to 6.6x.
    #
    # Treat every entry as +/-25%: repeat measurements of the same endpoint under
    # identical settings varied 20-50% batch to batch, because n=30 draws a
    # random mix of three task types whose service times differ by an order of
    # magnitude.
    #
    # NOTE the quality side of this swap is NOT settled. The two candidates were
    # scored under different prompt configs and wildly different sample sizes
    # (Ministral 8B n=7/10/13 with the brief math prompt; Qwen n=24 per task
    # without it), so the apparent quality gap is inside the noise. Re-run
    # bench_quality_matrix.py before any claim that depends on it.
    SERVICE_RATE = [
        0.6726, # 0 Ministral 3B
        # 0.1251, #   Gemma 3n E4B   (DROPPED)
        0.3052, # 1 Ministral 8B     (replaced Qwen3.5-9B at 0.1022)
        0.5964, # 2 GPT-4.1 nano
        # 0.1002, #   Ministral 8B   (DROPPED)
        0.4614, # 3 Mistral Small
        0.3285, # 4 GPT-4.1 mini
        # 0.1456, #   Llama 3.3 70B  (DROPPED)
    ]   # total 2.293 req/s over the retained 6
    # Derived, so it cannot silently disagree with MODEL_NAMES: trainer.py
    # takes the server count from this list's length.
    SERVER_CAPACITIES = [50] * len(MODEL_NAMES)

    USE_UTIL = True  # in the state use load/capability or load + capability

    # Dataset settings (ONE dataset per run)
    # Examples:
    #   - "tatsu-lab/alpaca"
    #   - "hotpotqa/hotpot_qa"   (set DATASET_CONFIG to "distractor" or "fullwiki")
    #   - "squad"
    #   - "cais/mmlu" - DATASET_CONFIG = "all" - DATASET_SPLIT = "auxiliary_train[:20000]"
    #   - "mixed" - "None" - "Train"
    # [MATH-only test run] previous single-dataset settings:
    #   DATASET_NAME = "cais/mmlu"; DATASET_CONFIG = "all"
    #   DATASET_SPLIT = "auxiliary_train[:50000]"
    DATASET_NAME = "qwedsacf/competition_math"
    DATASET_CONFIG = None     # Optional HF config name (e.g., HotpotQA: "distractor" / "fullwiki")
    DATASET_SPLIT = "train"   # "train" / "validation" / "test" (must exist in the dataset)
    MAX_SAMPLES = 50000         # Optional cap for faster experiments
    SHUFFLE_DATASET = True
    DATASET_SEED = 42
    # Single-dataset filters (None disables). All five MATH levels are used;
    # "boxed" keeps every problem with a \boxed answer (~12.5k samples), and
    # expression answers are scored by math-verify symbolic equivalence with
    # numeric fallback. Set DATASET_LEVELS = ["Level 3","Level 4","Level 5"]
    # for the hard subset only, or DATASET_FILTER = "numeric_boxed" to
    # restrict to plain-number answers.
    DATASET_LEVELS = None
    DATASET_FILTER = "boxed"

    # Add this inside class Config
    USE_MIXED_DATASET = True  # If True, use a mixture of datasets instead of a single one.

    MIXED_DATASETS = [
        # Multi-hop QA, needs context; metric uses token F1
        {
            "name": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "train",
            "weight": 1/3,
            "metric": "f1",
            "task_type": "multihop_qa",
            "max_samples": 20000,
        },
        # Reading comprehension / easier QA, metric uses token F1
        {
            "name": "squad",
            "config": None,
            "split": "train[:20000]",
            "weight": 0,
            "metric": "f1",
            "task_type": "qa",
        },
        # Multiple-choice knowledge/reasoning, metric checks A/B/C/D
        {
            "name": "cais/mmlu",
            "config": "all",
            "split": "auxiliary_train[:20000]",
            "weight": 0,
            "metric": "mmlu",
            "task_type": "mmlu",
        },
        # Harder MMLU replacement (MMLU-Pro): 12k questions, 10 options
        # (A-J, random floor 10%), reasoning-heavy, low contamination —
        # weak-strong gap roughly 2x wider than MMLU. Activate by setting
        # weight > 0 (typically also set cais/mmlu weight to 0).
        {
            "name": "TIGER-Lab/MMLU-Pro",
            "config": None,
            "split": "test",
            "weight": 1/3,
            "metric": "mmlu",
            "task_type": "mmlu_pro",
            "max_samples": 20000,
        },
        # Math reasoning, metric extracts exact final number
        {
            "name": "openai/gsm8k",
            "config": "main",
            "split": "train[:20000]",
            "weight": 0,
            "metric": "number",
            "task_type": "math",
        },
        # Competition math (MATH, full 12.5k train set, all five levels),
        # \boxed answers scored by math-verify symbolic equivalence with
        # numeric fallback. Set "levels": ["Level 3","Level 4","Level 5"]
        # to keep only the hard subset.
        {
            "name": "qwedsacf/competition_math",
            "config": None,
            "split": "train",
            "weight": 1/3,
            "metric": "math_verify",
            "task_type": "math_hard",
            "filter": "boxed",
            "levels": ["Level 1","Level 2","Level 3"],
            "max_samples": 20000,
        },
    ]


    # Prompt encoder settings (RouterNetwork)
    # Any SentenceTransformer model name, e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', etc.
    PROMPT_MODEL = "Alibaba-NLP/gte-modernbert-base"

    # If False: use raw SentenceTransformer embedding directly (no projection).
    # If True: learn a small projection emb_dim -> PROMPT_DIM.
    USE_PROMPT_PROJECTION = False
    PROMPT_DIM = 128


    # =========================================================
    # Router policy backbone (PPOAgent)
    # =========================================================
    # "mlp": current RouterNetwork (SentenceTransformer + MLP/Attn)
    # "llm": LLMRouterNetwork (HF causal LM encoder + small actor/critic heads)
    ROUTER_POLICY_BACKBONE = "mlp"  # "mlp" or "llm"

    # If ROUTER_POLICY_BACKBONE="llm", use this HF model as the router policy backbone.
    ROUTER_LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

    # LLM policy input formatting
    ROUTER_LLM_USE_CHAT_TEMPLATE = True
    ROUTER_LLM_MAX_INPUT_TOKENS = 4096
    ROUTER_LLM_STATE_DECIMALS = 4
    ROUTER_LLM_STATE_MAX_ELEMS = 256   # cap state vector serialization length
    ROUTER_LLM_INCLUDE_MODEL_NAMES = False  # include server->model mapping text (longer)

    # LLM tuning mode:
    #   - "heads": train only actor/critic heads (default)
    #   - "prefix": prefix-tuning (train soft prefix embeddings + heads; base LM frozen)
    #   - "full": fine-tune base LM (not recommended unless you use LoRA)
    ROUTER_LLM_TUNE_MODE = "heads"

    # Prefix-tuning (embedding-prefix) controls (only used when ROUTER_LLM_TUNE_MODE="prefix")
    ROUTER_LLM_PREFIX_LEN = 16
    ROUTER_LLM_PREFIX_INIT_STD = 0.02

    # Actor/Critic adapter head sizes (on top of LLM hidden state)
    ROUTER_LLM_ACTOR_HIDDEN = 512
    ROUTER_LLM_ACTOR_DEPTH = 4
    ROUTER_LLM_ACTOR_DROPOUT = 0.0

    ROUTER_LLM_CRITIC_HIDDEN = 256
    ROUTER_LLM_CRITIC_DEPTH = 4
    ROUTER_LLM_CRITIC_DROPOUT = 0.0

    # LLM policy training controls (important for memory)
    # ROUTER_LLM_FREEZE_BASE = True      # True => train only actor/critic heads
    # ROUTER_LLM_DTYPE = "float16"       # "float16" / "bfloat16" / "float32"
    # ROUTER_LLM_ATTN_IMPL = "flash_attention_2"  # or "sdpa" / "eager"
    # ROUTER_LLM_GRAD_CHECKPOINTING = False
    USE_FLASH_ATTN_2 = True
    FLASH_ATTN_FALLBACK = "sdpa"   # or "eager" or ""(no attn_impl)
    LOCAL_HF_DTYPE = "float16"     # "float16" / "bfloat16" / "float32"


    # QA prompt formatting (applies to QA-style datasets and also safe for Alpaca)
    QA_INCLUDE_CONTEXT = True  # include question context (e.g., passage) in the prompt
    QA_MAX_CONTEXT_DOCS = 10      # For datasets with multiple context documents (e.g., HotpotQA)
    QA_MAX_CONTEXT_CHARS = 2048  # Hard cap to avoid overly long prompts

    # Scoring: extract a final answer span before EM/F1 (prevents explanations from lowering scores)
    EXTRACT_FINAL_ANSWER = True
    FINAL_ANSWER_TAG = "final"

    # Reward function weights - adjusted for better balance
    ALPHA = 1/3   # Quality weight (increased importance)
    BETA = 1/3    # Latency weight
    REWARD_GAMMA = 1/3 # price weight (increased to emphasize cost)

    # =========================================================
    # Per-round (episode) min-max normalization for latency/price
    # =========================================================
    # If enabled, trainer will recompute rewards each episode using
    # min-max normalized latency/price over that episode.
    ROUND_MINMAX_NORM_ENABLE = False
    ROUND_MINMAX_NORM_LATENCY = True
    ROUND_MINMAX_NORM_PRICE = True
    ROUND_MINMAX_NORM_EPS = 1e-8
    ROUND_MINMAX_CLIP_01 = True

    # If True, only completed requests are used to compute min/max.
    ROUND_MINMAX_ONLY_COMPLETED = True

    # When env provides both `price_raw` and `price`, trainer prefers `price_raw`.
    ROUND_MINMAX_USE_PRICE_RAW_IF_AVAILABLE = True

    # =========================================================
    # Environment behavior when per-round min-max is enabled
    # =========================================================
    # If True, environment will NOT apply fixed constant normalization
    # (e.g., latency/MAX_LAT) for its internal reward computation.
    ENV_DISABLE_FIXED_NORM_WHEN_MINMAX = True
    # If True, environment will defer latency/price penalties in reward
    # when ROUND_MINMAX_NORM_ENABLE is on (trainer will recompute reward later).
    ENV_DEFER_LAT_PRICE_REWARD_WHEN_MINMAX = True

    LAMBDA = 5  # Capacity penalty weight (increased to strongly discourage invalid actions)
    # Latency normalizer: reward uses min(lat, MAX_LAT)/MAX_LAT, where lat is
    # END-TO-END (completion_time - arrival_time), i.e. queue wait + service.
    # Set it just past the observed tail: too high and cross-server differences
    # occupy a sliver of the range so the latency term discriminates far less
    # than the quality term (which spans all of [0,1]); too low and the bulk of
    # requests clip to 1.0 and the term stops producing a gradient at all --
    # fatal here, because the queue-wait part of the latency is exactly what
    # routing controls.
    #
    # 40, tracking the operating point rather than a fixed guess. It was 20
    # when the fleet ran a 512-token cap and 2.2 s mean service, then 60 when
    # rho sat at 0.989 and the M/M/1 estimate of end-to-end latency was ~30 s.
    # At the current rho = 0.797 with 2.90 s mean service that estimate is
    # 15.2 s, so 60 would leave the latency term using only the bottom quarter
    # of its range and compress every cross-server difference by 4x. 40 matches
    # the widest SLO threshold below, which is where the tail is expected to
    # sit, so the bulk of the distribution stays unclipped while the slope
    # stays 1.5x steeper than at 60.
    #
    # The tension is real in both directions. Judged on service time alone
    # (spread 1.5-6.9 s at empty queues) the latency term is much weaker than
    # the others -- reward spread 0.024 against 0.054 for quality and 0.179 for
    # price -- and matching quality would want MAX_LAT ~= 26. But that ignores
    # queue wait, which is the part routing actually controls: set it too low
    # and the bulk of requests clip to 1.0 and the term stops producing any
    # gradient at all. Re-set from the realised latency histogram of a run
    # (slo/violation_rate_{10,20,40} bracket it), not from either argument
    # alone, and re-check whenever rho changes.
    MAX_LAT = 40
    # SLO latency thresholds (seconds). Logged as violation rate =
    # fraction of completed requests with end-to-end latency > T.
    # Report a few (tight/moderate/loose); keep all below MAX_LAT.
    SLO_LATENCIES = [10.0, 20.0, 40.0]
    FAIR_REWARD_MIN_FLOOR = False # True the missing server will be set min rewards, False will use the floor reward -Beta-REWARD_GAMMA

    # =========================================================
    # PPO hyperparameters
    # Values below were converged from three real runs:
    #   ACTOR_LR 2e-6  -> never learns  (approx_kl ~0.001, entropy pinned)
    #   ACTOR_LR 1e-4  -> collapses     (KL spike 0.8, entropy 2.24 -> 0.2)
    #   ACTOR_LR 5e-5 + entropy brake -> stable to ~ep20, then needs the
    #   KL clamp + LR annealing below for a stable END of training.
    # =========================================================
    LEARNING_RATE = 1e-4  # legacy fallback, unused when ACTOR/CRITIC set
    GAMMA = 0.99          # discount factor
    GAE_LAMBDA = 0.95     # advantage estimation
    CLIP_EPSILON = 0.2    # PPO clip
    POLICY_COEF = 1       # Policy loss weight
    VALUE_COEF = 1        # Value loss weight
    # Anti-collapse brake: 0 collapses onto few servers; 0.02 only delayed
    # the slide to ~ep20; 0.03 is the current setting.
    ENTROPY_COEF = 0.03
    # 1e-4, lowered from 3e-4 alongside raising ACTOR_DUAL_NORM_TARGET_SPREAD to
    # 0.3. The two multiply: running-norm divides each tower by its measured rms
    # (~0.145) and rescales to tau, so the logits -- and every gradient flowing
    # through them -- are amplified by tau/rms. Going 0.1 -> 0.3 takes that
    # factor from ~1.2x to ~2.1x, which at 3e-4 would put the effective rate
    # near 6e-4, well past the 1e-4 that collapsed a run above. At 1e-4 the
    # effective rate lands around 2e-4 with USE_TARGET_KL_STOP as the backstop.
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 5e-4
    USE_LR_DECAY = False
    LR_DECAY_TYPE = "cosine"
    LR_DECAY_MIN_RATIO = 0.1
    # Spread the cosine over the ACTUAL run length (= MAX_EPISODES).
    # Unset, it falls back to 200 and the decay never bites in a 60-ep run
    # (LR would still be ~97% at ep25) -> no stable end-of-training phase.
    LR_DECAY_EPISODES = 200
    LR_WARMUP_EPISODES = 0
    KL_COEF = 0.00
    MAX_GRAD_NORM = 1
    PPO_EPOCHS = 4   # small interval batch: more epochs overfit noise
    BATCH_SIZE = 1

    PPO_RATIO_AGGREGATION = "per_request_mean"

    # Within-interval contextual-bandit per-request advantage.
    # The frozen interval state makes within-interval routing a contextual
    # bandit (actions don't change the observed context). So on top of the
    # shared interval advantage A^int_t, add a per-request individual advantage
    # A^ind_i = r^task_i - baseline, which sharpens credit from interval- to
    # request-granularity. Baseline here is the leave-one-out (LOO) interval
    # mean of the per-request task reward (unbiased; needs no extra head).
    # Combined per-request advantage: A_i = A^int_t + BANDIT_ADV_WEIGHT * A^ind_i.
    # NOTE: LOO does not control for prompt difficulty (a learned V^task head
    # would; that is a follow-up). Requires USE_PER_INTERVAL_MINIBATCH=False
    # and PPO_RATIO_AGGREGATION="per_request_mean". False = unchanged behavior.
    USE_BANDIT_ADVANTAGE = False
    BANDIT_ADV_WEIGHT = 0.5

    # Use a learned V^task head as the bandit baseline instead of the LOO mean.
    # V^task(o_t, x_i) predicts the expected per-request task reward given state
    # + prompt, trained by regression to the observed task reward. Then
    # A^ind_i = r^task_i - V^task_i. Unlike LOO it CONTROLS FOR PROMPT DIFFICULTY
    # (an easy prompt has high V^task, so it isn't miscredited as good routing),
    # is stable at small N_t, and doubles as a quality/value predictor. Adds a
    # small critic-side head -> needs a fresh model when turned on. Requires
    # USE_BANDIT_ADVANTAGE=True and the dual-tower CLIP path. False = LOO.
    BANDIT_USE_VTASK = False
    BANDIT_VTASK_COEF = 0.5   # weight of the V^task regression loss (critic side)

    # Interval weighting in the PPO policy loss.
    #   False (default): equal weight per INTERVAL (mean of per-interval means)
    #                    -> a request in a sparse interval counts more.
    #   True           : weight each interval by its arrival count N_t
    #                    -> equal weight per REQUEST (grand mean), lower
    #                    variance from small intervals. Reward normalization
    #                    (1/M or 1/N_t) is unaffected; this only reweights loss.
    PPO_LOSS_WEIGHT_BY_ARRIVALS = True

    # The ep20+ slide happened at KL 0.007-0.015 — entirely below the old
    # 0.04 target, so the early stop never fired. 0.012 clamps the late
    # acceleration while passing normal mid-run learning (0.002-0.003).
    TARGET_KL = 0.012
    # Enabled and then turned back off on 2026-07-26. The stop breaks out of
    # the epoch loop BEFORE the optimizer step, so firing on the first epoch
    # discards the whole episode's update -- observed at episode 0, where
    # approx_kl read 0.025 while policy/value/entropy losses all logged exactly
    # 0.0, i.e. an episode of real API calls bought no learning.
    #
    # It fired that early only because approx_kl is inflated: with
    # ACTOR_DUAL_RUNNING_NORM on and the network never switched out of train
    # mode, the rms buffers keep updating on every forward pass INCLUDING the
    # replayed ones inside the update, so pi_new != pi_old even on epoch 1
    # where the ratio should be identically 1. The inflation is worst early,
    # when rms is still travelling from its init of 1.0 down to ~0.1.
    #
    # Turning the stop off removes the symptom, not the cause: the ratio is
    # still computed against a policy that shifts under itself across the 4
    # epochs, so PPO's clipping acts on a moving target and approx_kl remains
    # a conflated measure of (policy change + rms drift). Freezing the rms
    # buffers during the update is the actual fix and is independent of this
    # flag.
    USE_TARGET_KL_STOP = False

    # Full-batch Path A over all intervals: every stability number above
    # (LR / KL / entropy) was measured on this path; minibatching the tiny
    # interval batch only adds gradient noise.
    USE_PER_INTERVAL_MINIBATCH = False
    PPO_INTERVAL_MINIBATCH_SIZE = 2
    PPO_SHUFFLE_INTERVALS = False
    USE_SERVERWISE_MLP = False

    PROMPT_MAX_TOKENS = 1024
    ROUTER_DEBUG_TEXT = False
    LLAVA_FUSION_LAYERS = 2

    USE_CLIP_FUSION_ROUTER = True
    ATTN_D_MODEL  = 256
    ATTN_N_HEADS  = 4
    ATTN_N_LAYERS = 2
    ATTN_FF_MULT  = 4
    ATTN_DROPOUT  = 0
    CLIP_INIT_TEMP = 0.2

    # Additive queue "highway" for the actor: adds a dedicated queue-based
    # term (from raw [util, residual, util/mu] per server) directly to each
    # server's logit, bypassing the prompt-token-dominated fusion attention.
    # Fixes weak queue-state perception without competing with prompt tokens.
    ACTOR_QUEUE_SKIP = False

    # Dual-tower actor: split the per-server logit into two dedicated scores
    #   quality_score = actor_head(prompt x STATIC capability channel)  ("is m capable?")
    #   queue_score   = queue_head([util, residual, util/mu])           ("is m free?")
    #   logit_m = quality_score_m + queue_score_m
    # The two scores are logged separately (dual/quality_spread, dual/queue_spread)
    # so you can see whether the quality tower learns and the queue tower fires.
    # Overrides ACTOR_QUEUE_SKIP when True. Requires a fresh model.
    ACTOR_DUAL_TOWER = True

    # Balance the two towers in logit = s_q*quality + s_k*queue via learnable
    # per-tower scales (exp-parameterized, logged as dual/scale_q, dual/scale_k).
    # Quality grows a ~30x larger spread and mutes the queue tower; queue starts
    # at scale ACTOR_DUAL_QUEUE_INIT_SCALE so it has a comparable voice, then the
    # reward tunes both. NOTE: the imbalance is largely reward-driven — at low
    # load / FAIR=0 the reward may still shrink s_k. To make the queue tower
    # actually matter, pair with a load-relevant regime (higher load / FAIR=1).
    ACTOR_DUAL_LEARN_SCALE = True
    # With ACTOR_DUAL_RUNNING_NORM on, both scores already enter the logit at
    # spread ~1, so this is a genuine prior weight rather than a 30x magnitude
    # patch: 1.0 = start the two towers on equal footing and let the reward
    # decide. (Values like 5 only made sense before normalization, where they
    # were compensating the quality tower's much larger raw spread.)
    # With ACTOR_DUAL_RUNNING_NORM on, both towers already enter the logit at
    # spread ~1, so this is a genuine prior weight, not a magnitude patch. 5.0
    # (which made sense pre-normalization) then hands the queue tower 5x the
    # voice of quality; measured effect was the entropy bonus flattening the
    # now-low-leverage quality tower (quality_spread 0.19 -> 0.05). 1.0 starts
    # the two on equal footing and lets the reward decide.
    ACTOR_DUAL_QUEUE_INIT_SCALE = 1.0

    # Learned FUSION of the two towers instead of a plain (scaled) sum. A small
    # per-server MLP reads [quality_score, queue_score] and outputs a scalar that
    # is ADDED as a correction to the additive base (logit = s_q*q + s_k*k + corr).
    # The correction head is ZERO-initialized, so training starts *identical* to
    # the current additive path and learns a nonlinear/gated fusion on top (e.g.
    # let quality dominate on easy prompts, let queue veto when a server is hot).
    # Cheap, permutation-equivariant (shared across servers), safe to toggle.
    # Enabled 2026-07-26 as a 65-parameter probe, not a capacity upgrade: the
    # additive base logit = s_q*q + s_k*k cannot express "tolerate a longer
    # queue when the quality gap is large", which is a real part of the routing
    # decision. dual/grad_fuse then answers whether that interaction is worth
    # learning at all -- and is the evidence to check before paying for a
    # richer mechanism (e.g. FiLM-conditioning the queue head on quality,
    # which would cost the towers' independent normalisation and the
    # queue_spread diagnostic).
    ACTOR_DUAL_FUSE = True
    ACTOR_DUAL_FUSE_HIDDEN = 16

    # Divide each tower's score by a DETACHED running estimate of its own
    # cross-server spread before combining. Fixes a structural asymmetry: the
    # quality tower has two growth channels (its fusion-transformer
    # representation keeps sharpening AND its head weights grow), so its raw
    # spread drifts upward without bound, whereas the queue tower reads a
    # hand-built z-scored descriptor whose distribution is stationary and can
    # only grow through head weights. Queue influence therefore decays
    # monotonically over a run. Dividing by a slow EMA of each spread closes the
    # magnitude channel, so s_q/s_k become genuine trade-off weights that the
    # reward tunes instead of being outrun by drift. Unlike a per-sample
    # z-score this preserves "no opinion": the divisor is a cross-batch
    # average, so a prompt on which a tower does not discriminate still yields
    # a small spread. Buffers update in train mode only, frozen at eval.
    # dual/quality_spread and dual/queue_spread keep logging the RAW (pre-norm)
    # spreads so they stay diagnostic; dual/rms_q and dual/rms_k log the
    # divisors. Adds two state_dict buffers -> needs a fresh model.
    ACTOR_DUAL_RUNNING_NORM = True
    ACTOR_DUAL_RUNNING_NORM_MOMENTUM = 0.05
    # Target cross-server spread each tower is normalized to (only used when
    # ACTOR_DUAL_RUNNING_NORM=True). Dividing by rms forces spread ~1, which
    # amplifies the logits (rms~0.06 => ~10x) and pulls the initial entropy down
    # to ~1.5 (ceiling ln(8)=2.08). Setting tau<1 keeps the anti-drift
    # normalization but softens the initial policy: tau=0.5 -> entropy ~1.88,
    # tau=0.3 -> ~2.0. Only the START is affected; the learnable tower scales
    # adapt afterward, so this is purely an exploration knob.
    # tau=1.0 makes each normalized tower spread 1, giving initial logit std
    # ~1.41 and initial entropy ~1.48 -- BELOW where runs without running-norm
    # naturally settle (~1.75), i.e. a harsher start than the policy has ever
    # had, on a setup with a documented collapse history. tau=0.7 puts the
    # initial entropy at ~1.73, matching that settling point, while keeping the
    # anti-drift property intact (tau is applied after the rms division, and
    # equally to both towers, so neither balance nor drift-capping changes).
    # 0.3, measured rather than guessed. With the rms buffers seeded to the
    # values a run converges to (rms_q 0.145, rms_k 0.120), tau maps to the
    # initial policy as: 0.10 -> logit std 0.094, entropy 100% of ln(M);
    # 0.30 -> 0.283, 98%; 0.70 -> 0.660, 91%; 1.00 -> 0.944, 85%.
    # Both ends of that range have been run: tau=0.7 gave approx_kl 0.14 and
    # collapsed entropy 1.78 -> 1.25 within two episodes, while tau=0.1 gave
    # approx_kl 0.0005 and left the policy uniform for twelve -- the regime the
    # header notes call "never learns". Extrapolating KL quadratically in the
    # logit scale puts tau=0.3 at approx_kl ~0.010, just under TARGET_KL, with
    # USE_TARGET_KL_STOP catching any overshoot.
    ACTOR_DUAL_NORM_TARGET_SPREAD = 0.3
    # Dedicated (higher) LR for the tower balance scales. Their gradient is
    # ~30x smaller than normal weights (chain rule multiplies by queue_score
    # ~0.02), so at the actor LR they barely move; this lets them adapt.
    ACTOR_DUAL_SCALE_LR = 1e-3

    # Queue tower input scale: feed raw queue LENGTH (util*capacity, e.g.
    # 10 vs 6 vs 9) instead of util (0.20 vs 0.12 vs 0.18). 50x bigger
    # server differences reach the queue head immediately, instead of
    # waiting for its weights to grow 50x at LR 3e-5. State layout is
    # unchanged (util stays at offset 0). drain becomes queue/mu = expected
    # drain time in seconds. Requires a fresh model.
    QUEUE_DESC_RAW_LOAD = True

    # Move price from the semantic (quality) tower into the numeric queue
    # tower. Quality tower then reads a [mu]-only static channel and matches
    # pure capability — dual/quality_spread becomes cost-free and finally
    # answers "does the quality tower learn?" unambiguously. Queue tower
    # becomes the unified numeric tower [load, cap, residual, mu, drain,
    # price_in, price_out] (prices x1e6, from Config.PRICE). State layout,
    # critic, quota decode and all baselines are untouched. Fresh model.
    DUAL_TOWER_PRICE_IN_QUEUE = False

    # Unify numeric-tower feature magnitudes to O(0.1-3): load/10, cap/50,
    # drain/10; residual, mu, prices already O(1). Differences stay full-size
    # (10 vs 6 -> 1.0 vs 0.6, still 5x the old util spread), but the freshly
    # initialized queue head no longer injects +-5-logit noise from raw
    # 0-30-range drains (amplified by scale_k), which would wreck early
    # exploration and conditioning.
    QUEUE_DESC_UNIT_SCALE = False

    # Cross-server z-score for the queue tower's wide-range features. When True,
    # the queue descriptor becomes [z(qload), residual, mu, z(drain)]: cap is
    # DROPPED (constant dead feature), and qload/drain are standardized ACROSS
    # the M servers, (x - mean_m)/(std_m + eps). This is scale-invariant (no
    # blow-up at heavy load), preserves who-is-busier ordering, and keeps mu
    # absolute (fixed capability). Overrides QUEUE_DESC_UNIT_SCALE. Pure
    # forward-side change (state layout / baselines / critic untouched), but the
    # queue_head input dim changes (5->4) so it needs a fresh model.
    QUEUE_DESC_ZSCORE = True

    # Append a per-server frozen, fleet-normalized ALLOCATION-COUNT feature
    # (counts[m] / mean(counts), CUMULATIVE over the episode so far, snapshotted
    # at each interval boundary) to the dual-tower QUEUE head only. Rationale:
    # util is an instantaneous "stock" (current queue, drains on fast servers),
    # while the counts-based quota fairness penalty is on the cumulative "flow"
    # of requests dispatched to each server. Feeding the episode-cumulative
    # alloc ratio closes that state<->reward gap so the policy can see who it
    # has been over-allocating over the long horizon (smoother than 1 interval).
    # Requires the dual tower (ACTOR_DUAL_TOWER) + CLIP fusion; needs a fresh
    # model (state feature dim 5 -> 6). False = unchanged behavior.
    QUEUE_USE_ALLOC = True

    # Capacity of the dual-tower QUEUE head, which reads the ~5-dim descriptor
    # [z(qload), residual, mu, z(drain), alloc] and emits one scalar per server.
    #   QUEUE_HEAD_DEPTH   number of hidden GELU layers. 1 reproduces the
    #                      original Linear->GELU->Linear head exactly.
    #   QUEUE_HEAD_HIDDEN  hidden width; 0 keeps the original ATTN_D_MODEL//4.
    # Changing either needs a fresh model (queue_head shape changes).
    #
    # Depth here is not the usual capacity argument: the input is five explicit
    # hand-built features with the key composition (drain = qload/mu) already
    # precomputed, and P2C balances this fleet with zero parameters, so the
    # target function is close to linear. What the extra layer plausibly buys
    # is a soft threshold -- "veto a server once its drain crosses X" -- that a
    # single hidden layer represents only coarsely.
    #
    # Note what depth CANNOT do: with ACTOR_DUAL_RUNNING_NORM on, the tower's
    # contribution to the logit is pinned at ACTOR_DUAL_NORM_TARGET_SPREAD
    # regardless of raw magnitude, so a bigger head cannot make the queue tower
    # louder, only better shaped at the same volume. Judge it on load/makespan
    # and load/overload_frac; if those do not move, the bottleneck is elsewhere
    # and the extra parameters are just harder to train at E[N_t] ~ 9.
    QUEUE_HEAD_DEPTH = 2
    QUEUE_HEAD_HIDDEN = 0

    # Hard cap on the post-arrival wait for outstanding requests to finish.
    # Without it the drain loop is unbounded: one request that never completes
    # (hung API call, dead collector, a lost completion signal) stalls the run
    # forever, which on an unattended machine costs hours. On timeout the
    # episode proceeds with whatever completed; the shortfall is visible as
    # outcome/incomplete_rate rather than failing silently. The arrival window
    # is INTERVAL_LENGTH*EPISODE_TIME_INTERVAL = 48 s and draining the backlog
    # takes tens of seconds at rho~0.9, so 180 s is several times the expected
    # drain and only fires when something is genuinely stuck.
    EPISODE_COMPLETION_TIMEOUT = 180

    SERVICE_RATE_EMA_ALPHA = 0.1
    SERVICE_RATE_MIN_SAMPLES = 1
    SERVICE_RATE_MIN = 1e-4
    SERVICE_RATE_MAX = 5.0

    # Number of parallel response-collector processes. Each one runs the
    # per-response quality scoring (math_verify can take up to ~10-15s on
    # pathological symbolic answers). A single collector serializes all
    # completions and stalls episode wall-clock; a pool runs math_verify
    # N-way in parallel so one slow response only ties up 1 of N collectors.
    NUM_RESPONSE_COLLECTORS = 4


    # Neural network settings
    HIDDEN_DIM = 512

    # Device settings
    GPU_LIST = [0]
    DEVICE = torch.device("cuda:0")

    # Wandb settings
    WANDB_PROJECT = "router"
    WANDB_ENTITY = None  # Set your wandb entity if needed

    # Logging
    LOG_INTERVAL = 5      # Log every 5 episodes
    SAVE_INTERVAL = 25    # Save every 25 episodes
    EVAL_INTERVAL = 15    # Evaluate every 15 episodes
    PLOT_INTERVAL = 50    # Plot progress every 50 episodes

    # Router QA generation controls (keeps answers short & deterministic)
    # 1024: a throughput/fidelity compromise. At 512 math_hard was truncated
    # 40-70% of the time, so the quality score measured the token cap rather
    # than the endpoint. At 2048 nothing truncates but fleet throughput falls
    # to 1.34 req/s, which caps the arrival rate and starves each interval of
    # requests. Measured math_hard medians on the retained six are 627, 616,
    # 268, 466, 193 and 295 tokens, so 1024 clears every median and only
    # truncates the upper tail -- heaviest on the verbose endpoints
    # (Ministral 3B/14B), negligible on GPT-4.1 mini and Llama 70B.
    # Cost bills actual completion tokens, so the cap never inflates spend for
    # endpoints that answer concisely. Re-measure SERVICE_RATE after changing
    # this: it moves fleet mu by more than any other single setting.
    GEN_MAX_NEW_TOKENS = 1024        # hard cap on answer length
    GEN_MIN_NEW_TOKENS = 0
    GEN_TEMPERATURE = 0.1
    GEN_TOP_P = 1
    GEN_DO_SAMPLE = False


    # Let multiple-choice prompts reason before committing to a letter.
    # MMLU-Pro is constructed to need multi-step reasoning; forcing a bare
    # letter ("Do not explain") put every endpoint at 0.00-0.25 against a
    # 10-way random floor of 0.10, i.e. the task carried no routing signal.
    # With reasoning enabled the fleet spreads 0.30-0.70 and orders by
    # capability. Costs ~275 completion tokens per MMLU request.
    MCQ_COT = True

    # Bound the working shown on math prompts ("at most 3 short steps").
    # Unlike MCQ_COT this is not a quality/throughput trade -- it improves
    # BOTH. Measured across the six endpoints at GEN_MAX_NEW_TOKENS=1024,
    # n=12 prompts: unbounded answers averaged 607 completion tokens and hit
    # the cap 36% of the time, and a truncated answer has no \boxed{} left so
    # math_verify scores it 0. Bounding took fleet mean quality 0.61 -> 0.74,
    # tokens -44%, latency -26%, cap hits 36% -> 15%, and every single
    # endpoint improved (e.g. Ministral 3B 0.50 -> 0.75 as its cap rate went
    # 50% -> 0%). At this token cap truncation dominates reasoning depth.
    # Re-evaluate if GEN_MAX_NEW_TOKENS is raised far above the ~600-token
    # unbounded mean, where the truncation term disappears and the usual
    # "more reasoning helps" trade should reassert itself.
    MATH_BRIEF_REASONING = True

    # Encourage a parseable final answer
    QA_PROMPT_STYLE = "plain"     # "instruction"/"alpaca" or "plain"
    # False: prompts use each dataset's standard output format (\boxed{}
    # for math, option letter for MMLU, short span for QA) instead of the
    # <final> tag; extraction falls back tag -> \boxed -> answer lines.
    QA_FORCE_FINAL_TAG = False
    FINAL_ANSWER_TAG = "final"
    TRUNCATE_AT_FINAL_TAG = True
    OUTPUT_FINAL_ONLY = True           # if True, store only <final>...</final> as response_text
    PRICE_USE_RAW_RESPONSE = True       # price penalty uses raw output length (more realistic)

    MISTRAL_MAX_RETRIES = 2
    API_TRANSIENT_FAIL_PENALTY = 0.0


    # Quality scoring settings. With USE_EM_EXACT_MATCH=True, keep the
    # synthetic capability prior neutral and use observed answer quality.
    MODEL_ELO_SCORES = {
        0: 1150,
        1: 1150,
        2: 1150,
        3: 1150,
        4: 1150,
        5: 1150,
        6: 1150,
        7: 1150,
    }

    USE_ATTN_ROUTER = False

    # ====================================================================
    # [CHANNEL] Dual-channel attention router: split per-server features
    # into dynamic (util only) and static (mu, prices) channels.
    # ====================================================================
    SERVER_DYN_DIM  = 2    # util, residual (in-flight elapsed service time)
    SERVER_STAT_DIM = 3    # mu, price_in, price_out


    # Load factor settings
    MAX_LOAD_FACTOR = 1.5

    # Final evaluation settings
    EVAL_EPISODES = 5         # Number of episodes for evaluation
    FINAL_EVAL_EPISODES = 10  # Number of episodes for final evaluation

    # Poisson prompt generation settings
    # Arrival rate, set against the measured SERVICE_RATE total so the system
    # stays heavily loaded but stable: queueing then dominates end-to-end
    # latency and routing decisions matter, while backlogs stay bounded within
    # an episode instead of diverging.
    #
    # 2.0 against sum(SERVICE_RATE) = 2.510 gives rho = 0.797: heavily loaded
    # but comfortably stable, and it yields 96 requests per 48 s episode with
    # E[N_t] = 12 arrivals per interval.
    #
    # Both ends of that range have been run and neither works. At rho = 0.989
    # (the earlier lambda=1.5 against a 1.517 fleet) arrivals match capacity
    # exactly, so the backlog performs a random walk rather than settling:
    # load/makespan swung 3 to 7.5 across episodes, which is the arrival
    # process, not the policy, and the M/M/1 reasoning used elsewhere in this
    # file stops applying. At the opposite end rho = 0.4 leaves queues near
    # empty, so the dual tower's queue half has nothing to discriminate on and
    # half the architecture is inert.
    #
    # E[N_t] matters here too: at ~8 arrivals over M=6 servers, roughly 40% of
    # the per-interval fairness penalty is irreducible multinomial sampling
    # noise (a perfectly proportional policy scores only 0.588 per-interval
    # Jain). At E[N_t] = 12 that floor rises materially.
    #
    # This is a dependent variable: it has to be re-derived from
    # sum(SERVICE_RATE) every time the fleet, the token cap or the dataset mix
    # changes, never chosen first. An arrival rate left over from an earlier
    # fleet is how this run silently reached rho = 2.47,
    # where every episode ends by hitting EPISODE_COMPLETION_TIMEOUT with a
    # backlog that never drains.
    POISSON_ARRIVAL_RATE = 1.5
    MAX_PROMPT_QUEUE_SIZE = 10000  # Maximum size of the prompt queue
    EPISODE_TIME_INTERVAL = 8 # How many intervals in current episode

    # Training settings
    EPISODE_LENGTH = 100  # Number of prompts per episode (increased for better learning)
    # Length of one telemetry interval, in seconds. This is the TMDP's gap
    # parameter, not a free knob: the policy sees the state frozen at the
    # interval boundary and routes every arrival inside the interval against
    # it, so dt sets how stale that state is allowed to get.
    #
    # 7 balances three things that pull in opposite directions:
    #   * E[N_t] = lambda*dt = 10.5 arrivals per interval. At dt=6 it was 9,
    #     and with M=5 a policy that samples exactly in proportion to capacity
    #     -- the most balanced one that exists -- scores only 0.681 on the
    #     per-interval Jain, so a third of the quota penalty is irreducible
    #     multinomial noise. dt=7 lifts that ceiling to 0.715, dt=8 to 0.740.
    #   * gradient samples per state, and 84 rather than 72 requests/episode.
    #   * staleness. The fleet completes ~2.4 requests/s, so a dt=7 interval
    #     hides ~17 completions; by dt=12 it hides ~28, more than are typically
    #     queued, and the boundary snapshot stops predicting end-of-interval
    #     load at all -- the queue tower would have nothing to read.
    #
    # Episode wall clock is dt * EPISODE_TIME_INTERVAL + drain ~= 60 s, so a
    # 200-episode run takes ~3.3 h. Sweep this ({4, 7, 12}) for the paper: if
    # performance is flat in dt, the telemetry-gap framing is unmotivated.
    INTERVAL_LENGTH = 7 # The length of interval
    MAX_EPISODES = 200   # match LR_DECAY_EPISODES above

    # Queue score settings
    QUEUE_SCORE_FACTOR = 0.2  # Factor to adjust queue score impact
    QUEUE_EPSILON = 0.0001  # Epsilon for queue score stability
    MERGE_ALPHA = 0 # Alpha for merging action probabilities (0.5 for equal weighting)

    # Drop action
    INVALID_ROUTE_PENALTY = 2/3   # try 0.5 ~ 2.0 depending how hard you want to avoid full servers
    FAIL_LATENCY_CAP = 30.0       # just for logging; failed branch uses penalty not latency
    REWARD_CLIP = -2             # optional, set <=0 to disable

    MASK = False

    ROUND_ROBIN = False

    USE_MERGE_TO_TRAIN = False  # Use merge action for training

    ADAPTIVE_EPSILON = False  # Use adaptive epsilon for exploration

    MIX_QUEUE_SCORE = False

    ENTROPY_BASED_EXPLORATION = False  # Use entropy-based exploration

    USE_AVG = False  # Use average reward for training

    RANDOM_SELECT = False

    NAIVE_PPO = False

    ENABLE_QUEUE_PENALTY = False

    # JSQ on the frozen telemetry snapshot = "stale-JSQ" (routes to the
    # shortest raw queue using interval-boundary state).
    JSQ = False

    P2C = False

    # Capacity-weighted JSQ: route to the shortest EXPECTED DRAIN TIME
    # (queue / mu), accounting for heterogeneous service rates.
    CAP_WEIGHTED_JSQ = False

    # Power-of-d-choices: sample d admissible servers, route to the shortest
    # (P2C is d=2). POWER_OF_D_WEIGHTED ranks by queue/mu instead of queue.
    POWER_OF_D = False
    POWER_OF_D_CHOICES = 3
    POWER_OF_D_WEIGHTED = False

    # GREEDY = False

     # Greedy utility baseline (predict next-step reward using queue Q + EMA latency/cost per server)
    GREEDY_UTILITY = False
    GREEDY_MASK = False # True will enable action mask
    # Queue-conditioned predictor for GREEDY_UTILITY.
    #   - "none"   : use a single global EMA per server
    #   - "bins"   : keep EMA latency/cost in coarse bins of queue length q
    #   - "linear" : online fit of latency/cost as a + b*q per server
    UTILITY_QUEUE_MODEL = "linear"
    UTILITY_EMA_ALPHA = 0.10      # EMA update rate for latency/cost (0.05-0.2 typical)
    UTILITY_W_QUAL = 1.0          # weight on predicted quality
    UTILITY_W_LAT = 1.0           # weight on predicted latency (penalty)
    UTILITY_W_COST = 1.0          # weight on predicted cost (penalty)
    UTILITY_W_Q = 0.1             # optional extra queue penalty beyond latency term
    UTILITY_Q_EPS = 1e-6
    UTILITY_INIT_LAT = 0        # seconds (fallback if no history yet)
    UTILITY_INIT_COST = 0       # fallback if no history yet
    # ============================================================
    # Greedy utility exploration (safe defaults)
    # ============================================================

    # Ensure each server is tried at least this many times before pure greedy utility.
    # Set 0 to disable.
    GREEDY_WARMUP_MIN_TRIALS = 1

    # With probability epsilon, pick a random server (exploration).
    # Set 0.0 to disable.
    GREEDY_EPSILON = 0.2

    # If > 0, adds a UCB-style bonus to uncertain servers in greedy utility.
    # Set 0.0 to disable.
    GREEDY_UCB_COEF = 0.0

    # If > 1, sample uniformly among the top-K highest-utility servers (simple exploration).
    # Set 1 to disable.
    GREEDY_TOPK = 1

    T = -2
    # FAIR = 1  # 0..1, it will control how fair you want, 1 max, 0 min
    T_QUEUE = -2
    T_REWARD = -2
    FAIR_WARMUP_EPISODES = 0
    FAIRNESS_MODE = "quota"
    # Quota fairness normalizer:
    #   "service_rate"  : S_m = mu_m * dt  (mu-weighted; but mu=req/s is
    #                     endogenous — harder prompts lower measured mu).
    #   "queue_capacity": S_m = SERVER_CAPACITIES  (exogenous, fixed,
    #                     difficulty-independent). Water-filling equalizes
    #                     post-dispatch occupancy (D+n)/capacity, and
    #                     quota_jain_norm_load becomes queue-capacity-normalized.
    QUOTA_NORMALIZE_BY = "service_rate"
    # Use the FROZEN Config.SERVICE_RATE above as the service budget rather than
    # the online EMA carried in the state snapshot. The online estimate is
    # endogenous -- harder prompts produce longer responses and therefore a
    # lower measured req/s -- so a quota built on it would let the fairness
    # reference drift with the very workload being routed. The frozen values are
    # exogenous and fixed at configuration time, which is what makes the
    # entitlement well defined. False = use the live EMA from the snapshot.
    QUOTA_USE_FROZEN_MU = True
    # Count the in-flight (currently-serving) request in the water-filling
    # backlog D. util = qsize excludes it (it is dequeued while generating),
    # so without this a server busy on a long generation looks empty and the
    # quota over-allocates to it. Derived from residual>0 (single worker).
    #
    # Turned on 2026-07-26 because the blind spot is large at this operating
    # point, not marginal. Measured mean service times against the 6 s interval:
    # Llama 70B takes 6.87 s, i.e. LONGER than an interval, so it is essentially
    # always mid-generation and its in-flight request is invisible to the quota
    # for the whole episode; Ministral 8B (3.28 s) and GPT-4.1 mini (3.04 s) are
    # hidden about half the time. At rho = 0.797 that is ~0.8 invisible requests
    # per server against the ~2 the quota hands out per interval (E[N_t]=12 over
    # M=6), so the backlog it reasons about is understated by roughly 40% -- and
    # understated most for the slowest endpoints, which are exactly the ones it
    # should be steering away from.
    QUOTA_COUNT_INFLIGHT = True

    # History-aware fair quota: fold the episode-cumulative allocation count H_m
    # (requests routed to server m in earlier intervals of THIS episode) into the
    # water-filling backlog, so the quota minimizes
    #     Psi = sum_m (w*H_m + D_m + n_m)^2 / S_m
    # instead of (D_m + n_m)^2 / S_m. D_m (current backlog) drains fast on fast
    # servers, so it gives almost no episode-scale memory: two servers that have
    # both emptied their queues look identical even if one has historically been
    # over-allocated. Adding H_m gives the quota a long-horizon memory, so a
    # server that got more earlier is assigned less now, targeting LONG-TERM
    # (cumulative) load fairness rather than only per-interval fairness. The
    # weight w trades the two off: w=0 is the current per-interval quota; larger
    # w corrects historical imbalance more aggressively (and, since H grows over
    # the episode while D stays bounded, eventually dominates D -- keep w modest).
    # H resets each episode (the fairness horizon is one episode).
    # 1.0, raised from 0 on 2026-07-26. w=0 makes the quota a PER-INTERVAL
    # balance target, and at this scale that target is mostly unreachable
    # noise: with E[N_t] ~ 8 spread over M=6, a policy that samples exactly in
    # proportion to capacity -- the most balanced policy that exists -- scores a
    # per-interval Jain of only 0.588, because allocating eight indivisible
    # requests across six servers cannot be even. The same policy scores 0.917
    # cumulatively, which matches what runs actually measure. So roughly 40% of
    # the per-interval penalty was charging the policy for multinomial sampling
    # noise no policy can remove, injecting that variance straight into the
    # gradient.
    #
    # Worse, per-interval balance forbids the one thing a prompt-aware router
    # has over a queue-only baseline: if an interval happens to deliver mostly
    # math prompts, the right move is to over-serve the math specialist now and
    # compensate later. Only a cumulative target permits that trade across time.
    # This is the deficit-counter idea from DRR -- unserved quantum carries to
    # the next round, so fairness is a long-run property, not a per-round one.
    #
    # w=1.0 puts w*H on par with D around mid-episode (H ~ 5, D ~ 5.7 at
    # rho=0.85) and lets it dominate later, which is what long-run fairness
    # means. If end-to-end latency degrades -- the quota can now favour a server
    # that is behind on cumulative count even while its queue is long -- step
    # back to 0.5 rather than to 0.
    QUOTA_HISTORY_WEIGHT = 1.0
    FAIR_TARGET = 1     # 最终的 FAIR 值
    FAIR = 1           # 起始（trainer 会覆盖）

    # =========================================================
    # Lagrangian (adaptive) fairness — RCPO, Tessler et al. 2018
    # =========================================================
    # FAIR is a PRICE, not a CONSTRAINT: once the quality gain of concentrating
    # exceeds the quota penalty, the policy pays it and Jain slides. FAIR=1 is
    # already the maximum fixed strength (p = max(0, e+f-1) saturates at p=e),
    # so there is no knob left. This makes the penalty strength an adaptive
    # multiplier mu instead:
    #   primal: quota_flair_reward scales its penalty by mu,
    #           rhat = r_floor + (1 - mu*p)(rbar - r_floor)
    #   dual:   mu <- clip(mu + MU_LR*(JAIN_FLOOR - Jain_ema), 0, MU_MAX)
    # mu is UNBOUNDED above, so mu*p can exceed 1 and push an over-allocated
    # server below the floor -- pressure that fixed f can never reach. Jain
    # below the floor -> mu rises until it is pushed back; Jain at or above the
    # floor -> mu decays toward 0 and quality is free to improve. Set the floor
    # to a value the fleet demonstrably reaches (check the natural Jain first);
    # an unreachable floor makes mu grow without bound and crushes task reward.
    # Two-timescale: keep MU_LR small so mu moves slower than the policy.
    USE_LAGRANGIAN_FAIR = False
    LAGRANGIAN_JAIN_FLOOR = 0.85
    # NOTE on scale: the dual gradient is the violation (floor - Jain), which is
    # only ~0.03 in magnitude, so a "normal-looking" LR like 0.03 moves mu by
    # ~1e-3 per episode and does nothing within a 200-episode run. 1.0 reaches
    # the floor in ~100 episodes in simulation without oscillating; raise toward
    # 2.0 for a faster lock-on, lower if mu overshoots and Jain rings.
    LAGRANGIAN_MU_LR = 0.2
    LAGRANGIAN_MU_INIT = 1.0    # 1.0 == current fixed-strength behavior
    LAGRANGIAN_MU_MAX = 10.0
    LAGRANGIAN_JAIN_EMA = 0.3   # EMA smoothing of episode Jain for the mu update

    # =================================================================
    # VISUALIZATION AND LOGGING CONTROL
    # =================================================================

    # Master switches for different types of logging/visualization
    ENABLE_WANDB_LOGGING = True          # Enable/disable all wandb logging
    ENABLE_CONSOLE_LOGGING = True        # Enable/disable console output
    ENABLE_QUEUE_MONITORING = True       # Enable/disable queue state monitoring
    ENABLE_VISUALIZATIONS = True         # Enable/disable all plot generation
    ENABLE_FILE_EXPORTS = True           # Enable/disable file exports

    # Detailed visualization control
    VISUALIZATION_CONFIG = {
        'training_progress_plots': True,     # Episode rewards, moving averages
        'queue_monitoring_plots': True,      # Queue states, utilization
        'final_analysis_plots': True,       # Comprehensive final analysis
        'action_distribution_plots': True,   # Action distribution charts
        'server_utilization_plots': True,   # Server utilization over time
        'real_time_plots': False,           # Real-time plotting (resource intensive)
    }

    # Detailed logging control
    LOGGING_CONFIG = {
        'episode_metrics': False,            # Basic episode metrics (rewards, actions)
        'queue_events': True,               # Individual queue events (add/complete/fail)
        'queue_trends': False,               # Queue trend analysis
        'server_statistics': False,          # Server performance statistics
        'training_metrics': True,           # PPO training loss metrics
        'evaluation_metrics': True,        # Periodic evaluation results
        'real_time_queue_state': True,     # Real-time queue state updates
        'periodic_summaries': True,        # Periodic summary reports
    }

    # File export control
    EXPORT_CONFIG = {
        'queue_events_json': False,          # Export queue events to JSON
        'training_checkpoints': True,       # Save model checkpoints
        'final_reports': True,              # Generate final analysis reports
        'plot_images': True,                # Save plots as image files
        'csv_metrics': False,               # Export metrics to CSV (optional)
        'detailed_logs': False,             # Detailed debug logs (verbose)
    }

    # Wandb specific control
    WANDB_CONFIG = {
        'log_episode_metrics': True,        # Episode-level metrics
        'log_queue_events': True,           # Individual queue events
        'log_training_plots': True,         # Upload training plots
        'log_queue_plots': True,            # Upload queue monitoring plots
        'log_model_artifacts': True,        # Upload model checkpoints
        'log_hyperparameters': True,        # Log all hyperparameters
        'watch_model': True,                # Watch model gradients/weights
    }

    # Console output control
    CONSOLE_CONFIG = {
        'episode_progress': True,           # Episode progress messages
        'queue_events': True,               # Real-time queue event messages
        'training_updates': True,           # Training progress updates
        'evaluation_results': True,        # Evaluation results
        'periodic_summaries': False,        # Periodic queue summaries
        'error_messages': True,             # Error and warning messages
        'debug_messages': False,            # Detailed debug messages
    }

    # Performance and frequency control
    FREQUENCY_CONFIG = {
        'queue_event_logging': 1,           # Log every N queue events (1 = all)
        'queue_summary_interval': 5.0,     # Queue summary every N seconds
        'plot_generation_interval': 25,    # Generate plots every N episodes
        'checkpoint_save_interval': 25,     # Save checkpoints every N episodes
        'wandb_upload_interval': 1,        # Upload to wandb every N log calls
        'file_export_interval': 50,        # Export files every N episodes
    }


    # =================================================================
    # Quality scoring and LLM-as-judge (optional)
    # =================================================================

    # Ground-truth matching for quality (optional)
    # =================================================================
    # If True, compute quality_score by comparing the model
    # response_text against the dataset ground-truth output.
    #
    # NOTE: This metric is computed AFTER generation completes, so you
    # typically should NOT include it in the RL state.
    USE_EM_EXACT_MATCH = True
    USE_LLM_JUDGE = False

    ZERO_QUALITY_USE_JUDGE = False
    ZERO_QUALITY_JUDGE_THRESHOLD = 1e-8

    # If judge is used, final quality = max(EM/F1, judge_score)
    ZERO_QUALITY_JUDGE_USE_MAX = False

    # Judge model
    LLM_JUDGE_MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    LLM_JUDGE_NORMALIZE = "sigmoid"
    LLM_JUDGE_NORM_K = 1.0

    LLM_JUDGE_USE_RAW_RESPONSE = False
    LLM_JUDGE_PRELOAD = True
    LLM_JUDGE_DEVICE = "cuda"
    LLM_JUDGE_DTYPE = "float16"
    LLM_JUDGE_ATTN_IMPL = "flash_attention_2"
    LLM_JUDGE_MAX_LENGTH = 2048
    LLM_JUDGE_CACHE_PATH = "judge_cache_skywork_gt.jsonl"
    LLM_JUDGE_CACHE_IN_MEMORY = True

    # If True, include per-server quality scores in the RL state vector.
    # Recommended False when USE_EM_EXACT_MATCH=True.
    INCLUDE_QUALITY_IN_STATE = False

    # Softer matching metric (less strict than EM):
    #   "f1"       : token-level F1 (default, continuous 0..1)
    #   "ratio"    : character-level similarity ratio (0..1)
    #   "contains" : 1 if one contains the other (after normalization)
    #   "em"       : strict exact match (0/1)
    # Fallback metric, used ONLY when a gold answer arrives as a bare
    # string; per-sample gold dicts carry their own "metric" and override
    # this (MATH samples embed "number"). Aligned with the MATH-only run.
    # EM_METRIC = "mmlu"
    EM_METRIC = "number"

    # Optionally binarise the match score (useful if you want 0/1 reward)
    EM_BINARIZE = False
    EM_THRESHOLD = 0.2

    @classmethod
    def get_config_summary(cls):
        """Get a summary of current configuration"""
        summary = {
            'wandb_logging': cls.ENABLE_WANDB_LOGGING,
            'console_logging': cls.ENABLE_CONSOLE_LOGGING,
            'queue_monitoring': cls.ENABLE_QUEUE_MONITORING,
            'visualizations': cls.ENABLE_VISUALIZATIONS,
            'file_exports': cls.ENABLE_FILE_EXPORTS,
            'active_visualizations': sum(cls.VISUALIZATION_CONFIG.values()),
            'active_logging': sum(cls.LOGGING_CONFIG.values()),
            'active_console': sum(cls.CONSOLE_CONFIG.values()),
        }
        return summary

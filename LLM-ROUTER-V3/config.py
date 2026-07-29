import torch

class Config:
    # ================================================================
    # Fleet: 6 endpoints, non-reasoning (or reasoning verified OFF).
    # Three specialist clusters: math -> Ministrals, MCQ -> 5.4-nano/small,
    # qa -> large/small. math-vs-mmlu rank corr +0.03, oracle headroom +0.051.
    # Router action = list index: keep MODEL_NAMES / PRICE / SERVICE_RATE in
    # the same order. Dropped models stay commented in place.
    # New-endpoint reasoning screen: empty content? billed reasoning_tokens>0?
    # inline <think>? out_tok pinned at the cap?
    # ================================================================
    # 8-fleet (2026-07-30): cheap five + mid trap + premium pair.
    # Roles on v2' (drop/logiqa/math): mini = quality king (drop 0.75, math
    # 0.83); large = logiqa co-crown; codestral = DELIBERATE trap (wins
    # nothing, drop 0.25) -- quality-blind baselines feed it its ~14% quota
    # share, quality-aware methods route around it within the delta slack.
    MODEL_NAMES = [
        "ministral-3b-2512",                                   # 0 cheap/fast floor
        "ministral-8b-2512",                                   # 1 cheap, math 0.917
        "gpt-4.1-nano-2025-04-14",                             # 2 fast all-rounder
        "mistral-small-2506",                                  # 3 value king (qa+mmlu)
        "gpt-5.4-nano",                                        # 4 MCQ/logiqa specialist; reasoning_effort="none", verified reasoning_tokens=0
        "codestral-2508",                                      # 5 trap: mid price, wins nothing on v2'
        "mistral-large-2512",                                  # 6 premium: logiqa co-crown
        "gpt-4.1-mini",                                        # 7 premium king: drop 0.75 / math 0.83
        # ---- dropped, kept for the record ----
        # "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",  # best quality/price flagship, but Together serverless degrades badly under concurrent runs
        # "gpt-4.1-mini",                            # dominated by small (0.669 @ $349 vs 0.764 @ $131)
        # "codestral-2508",                          # wins nothing, 3rd most expensive
        # "gpt-4o-mini",                             # dominated by small, slowest (mu .39)
        # "together/Qwen/Qwen2.5-7B-Instruct-Turbo", # weakest overall (0.550), redundant
        # "together/google/gemma-3n-E4B-it",         # reasoning trap: empty content
        # "ministral-14b-2512",                      # clean but slow (mu .31), mmlu 0.556
        # "mistral-medium-2604",                     # true apex but $1.5/$7.5 -- too expensive
    ]

    PRICE = [   # $(in, out) per token
        (0.00000010, 0.00000010),   # 0 3b        $0.10/$0.10 per 1M
        (0.00000015, 0.00000015),   # 1 8b        $0.15/$0.15
        (0.00000010, 0.00000040),   # 2 4.1-nano  $0.10/$0.40
        (0.00000015, 0.00000060),   # 3 small     $0.15/$0.60
        (0.00000020, 0.00000125),   # 4 5.4-nano  $0.20/$1.25 (confirmed 2026-07-28)
        (0.00000030, 0.00000090),   # 5 codestral $0.30/$0.90
        (0.00000050, 0.00000150),   # 6 large     $0.50/$1.50 (confirmed 2026-07-28)
        (0.00000040, 0.00000160),   # 7 4.1-mini  $0.40/$1.60
        # ---- dropped ----
        # (0.00000104, 0.00000104), # Llama-3.3-70B-Turbo
        # (0.00000040, 0.00000160), # gpt-4.1-mini
        # (0.00000030, 0.00000090), # codestral
        # (0.00000015, 0.00000060), # gpt-4o-mini
        # (0.00000030, 0.00000030), # Qwen2.5-7B-Turbo
    ]

    # 1 / mean serial service time (one single-threaded worker per endpoint).
    # Also the frozen fairness reference (QUOTA_USE_FROZEN_MU). Batch variance
    # is 25-50%: re-measure (bench_fleet_service_rate.py, n=30, one batch)
    # whenever the fleet, prompts or token cap change.
    # SERVICE_RATE = [
    #     0.8320, # 0 3b        (2026-07-28 evening batch, n=30)
    #     0.2845, # 1 8b        (halved vs prior batch -- long math CoT sample)
    #     0.4748, # 2 4.1-nano
    #     0.5876, # 3 small
    #     0.3302, # 4 large
    #     0.4664, # 5 5.4-nano
    #     # 0.2500 70B (probe) | 0.4456 gpt-4.1-mini | 0.6399 codestral | 0.3929 4o-mini (dropped)
    # ]   # total 2.976 req/s -> POISSON_ARRIVAL_RATE=2.5 gives rho=0.84
    # ---- v2-easy pilot (drop/arc/math), superseded by v2' below ----
    # SERVICE_RATE = [0.8590, 0.6881, 0.6646, 0.6732, 0.3786, 0.6871]  # total 3.951
    # ---- v2' mix (drop/logiqa/math L1-5), 2026-07-29 batch, n=30 ----
    # harder mix -> longer outputs -> mu down across the board.
    SERVICE_RATE = [
        0.7251, # 0 3b        (v2' batch)
        0.2952, # 1 8b
        0.5639, # 2 4.1-nano
        0.4935, # 3 small
        0.5484, # 4 5.4-nano
        0.4928, # 5 codestral (v2' probe n=25)
        0.2322, # 6 large     (v2' batch)
        0.2650, # 7 4.1-mini  (v2' probe n=25)
    ]   # total 3.616 req/s -> lambda 2.53 gives rho=0.70

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
            "weight": 0,
            "metric": "f1",
            "task_type": "multihop_qa",
            "max_samples": 20000,
        },
        # Closed-book recall (replaced HotpotQA 2026-07-27: its truncated
        # contexts made scores prompt-driven, and it carried 67% of input
        # tokens). Answers ship alias lists; token F1 handles them.
        {
            "name": "mandarjoshi/trivia_qa",
            "config": "rc.nocontext",
            "split": "train[:20000]",
            "weight": 0,
            "metric": "f1",
            "task_type": "qa",
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
            "weight": 0,
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
        # [v2 mix] Discrete reasoning over paragraphs (counting/arithmetic
        # on text), token F1 over answer spans (answers_spans.spans).
        # v2 recipe = drop 1/3 + arc 1/3 + competition_math 1/3, v1 trio -> 0.
        {
            "name": "ucinlp/drop",
            "config": None,
            "split": "train",
            "weight": 1/3,
            "metric": "f1",
            "task_type": "drop",
            "max_samples": 20000,
        },
        # [v2 mix] Science-reasoning MCQ; reuses the mmlu metric/prompt via
        # the choices-dict + answerKey loader support.
        {
            "name": "allenai/ai2_arc",
            "config": "ARC-Challenge",
            "split": "train",
            "weight": 0,    # DROPPED for v2: fleet ceilinged it (5/6 endpoints ~1.0)
            "metric": "mmlu",
            "task_type": "arc",
            "max_samples": 20000,
        },
        # [v2 mix] Logical-reasoning MCQ (passage + query + 4 options),
        # replaces ARC; loader maps query/context/correct_option.
        {
            "name": "lucasmccabe/logiqa",
            "config": None,
            "split": "train",
            "weight": 1/3,
            "metric": "mmlu",
            "task_type": "logiqa",
            "max_samples": 20000,
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
            "levels": ["Level 1","Level 2","Level 3","Level 4", "Level 5"],
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
    # If enabled, trainer recomputes rewards each episode. Price uses
    # Router-R1's sliding-window percentile normalization (sqrt preprocess,
    # rolling buffer across episodes, 5/95th-percentile bounds), fed with
    # true dollar_cost. Latency stays on absolute /MAX_LAT semantics
    # (NORM_LATENCY=False) so SLO anchoring is preserved.
    ROUND_MINMAX_NORM_ENABLE = True
    ROUND_MINMAX_NORM_LATENCY = False
    ROUND_MINMAX_NORM_PRICE = True
    ROUND_MINMAX_NORM_EPS = 1e-8
    ROUND_MINMAX_CLIP_01 = True
    ROUND_MINMAX_WINDOW = 1000          # rolling buffer size (Router-R1)
    ROUND_MINMAX_PERCENTILES = (5, 95)  # robust min/max bounds (Router-R1)
    # frozen = bounds from the uniform-routing calibration file, shared by
    # every arm (rewards commensurable across methods); rolling = Router-R1's
    # adaptive buffer. Regenerate the file when fleet/mix/token cap change
    # (scratchpad calib_window.py, 1000 uniform calls).
    ROUND_MINMAX_WINDOW_MODE = "frozen"
    ROUND_MINMAX_WINDOW_FILE = "price_window_v2p.json"

    # Reward combiner (applies in the trainer recompute, needs minmax on):
    #   "linear" = ALPHA*q - BETA*lat_norm - REWARD_GAMMA*price_norm (sealed main)
    #   "gated"  = q * (1 - GATED_BETA*lat_norm) * (1 - GATED_GAMMA*price_norm)
    # gated: earned quality discounted by operational cost; q=0 zeroes the
    # request (cheap wrong answers worth nothing; difficulty noise muted).
    # NOTE gated rewards run ~3-4x larger than linear -- for FLAIR arms raise
    # FAIR_MU_MAX (~15) so the fairness dual keeps authority, and do NOT
    # compare reward values across combiners (component metrics stay valid).
    REWARD_COMBINER = "linear"
    REWARD_GATED_BETA = 1.0
    REWARD_GATED_GAMMA = 1.0

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
    # Latency normalizer AND slope: penalty = BETA*min(lat,MAX_LAT)/MAX_LAT.
    # Too high -> cross-server differences compressed; too low -> bulk clips
    # to 1 and the gradient dies. Re-check whenever rho changes.
    MAX_LAT = 30
    # SLO latency thresholds (seconds). Logged as violation rate =
    # fraction of completed requests with end-to-end latency > T.
    # Report a few (tight/moderate/loose); keep all below MAX_LAT.
    SLO_LATENCIES = [5.0, 10.0, 20.0]
    FAIR_REWARD_MIN_FLOOR = False # True the missing server will be set min rewards, False will use the floor reward -Beta-REWARD_GAMMA

    # ---- PPO ----
    LEARNING_RATE = 1e-4  # legacy fallback, unused when ACTOR/CRITIC set
    GAMMA = 0.99          # discount factor
    GAE_LAMBDA = 0.95     # advantage estimation
    CLIP_EPSILON = 0.2    # PPO clip
    POLICY_COEF = 1       # Policy loss weight
    VALUE_COEF = 1        # Value loss weight
    # Exploration bonus; also fights determinism the fairness constraint wants.
    ENTROPY_COEF = 0.01
    # Note: running-norm multiplies effective actor LR by tau/rms (<=6x with the floor).
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 5e-4
    USE_LR_DECAY = False
    LR_DECAY_TYPE = "cosine"
    LR_DECAY_MIN_RATIO = 0.1
    # Spread the cosine over the ACTUAL run length (= MAX_EPISODES).
    # Unset, it falls back to 200 and the decay never bites in a 60-ep run
    # (LR would still be ~97% at ep25) -> no stable end-of-training phase.
    LR_DECAY_EPISODES = 200
    LR_WARMUP_EPISODES = 3
    KL_COEF = 0.00
    MAX_GRAD_NORM = 1
    PPO_EPOCHS = 4   # small interval batch: more epochs overfit noise
    BATCH_SIZE = 1

    PPO_RATIO_AGGREGATION = "per_request_mean"

    # Optional per-request LOO advantage on top of the shared interval one
    # (A_i = A_t + w*(r_i - LOO mean)). Off = estimator the theory covers.
    USE_BANDIT_ADVANTAGE = False
    BANDIT_ADV_WEIGHT = 0.5

    # Bandit baseline variant: V^task instead of LOO. Needs USE_BANDIT_ADVANTAGE.
    BANDIT_USE_VTASK = False

    # b_t = mean_i V^task(s_t,p_i), subtracted from the interval advantage
    # BEFORE normalization. Action-free -> adds no bias; removes the prompt-
    # difficulty component (~65% of per-request variance) the critic cannot
    # see (s_t predates the arrivals). Needs a fresh model.
    VTASK_INTERVAL_BASELINE = True

    # Own net (not a head on the actor trunk): the trunk is trained to RANK
    # servers and discards absolute difficulty; sharing would also train the
    # actor's representation and confound the ablation.
    VTASK_INDEPENDENT_NET = True
    VTASK_NET_HIDDEN = 256
    VTASK_NET_DEPTH = 2
    BANDIT_VTASK_COEF = 0.5   # unused: V^task no longer rides in critic_loss

    # Fitted AFTER the PPO loop on its own optimizer + replay (unbiased for
    # any fitting scheme). 8 steps @5e-4: 64 @1e-3 overfit the replay in sim.
    VTASK_TRAIN_STEPS = 8     # gradient steps per episode (was effectively <=4)
    VTASK_BATCH_SIZE = 256
    VTASK_LEARNING_RATE = 5e-4   # its own optimizer; matches the critic LR
    # Replay of (input, realized reward); one episode alone is too few samples.
    VTASK_REPLAY_SIZE = 20000
    # Huber knee well below target std (~0.17), else smooth_l1 = scaled MSE.
    VTASK_HUBER_BETA = 0.1

    # True: flat 1/sum(N_t) over requests -- matches the factorized joint-
    # action gradient. False: equal weight per interval (legacy).
    PPO_LOSS_WEIGHT_BY_ARRIVALS = True

    # Early-stop threshold on mean per-request KL (joint interval KL = SUM of
    # per-request KLs; log both when comparing across delta_t).
    TARGET_KL = 0.012
    # Epoch-0 parity is exact now (frozen rms), so the stop only trims late
    # epochs. Off by choice; watch train/mean_request_kl <= ~0.05 instead.
    USE_TARGET_KL_STOP = False

    # Full-batch path A; minibatching the tiny interval batch adds noise.
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

    # Legacy queue-highway (superseded by the dual tower).
    ACTOR_QUEUE_SKIP = False

    # Dual tower: logit_m = s_q*quality(prompt x capability) + s_k*queue(load).
    # Diagnostics: dual/quality_spread, dual/queue_spread, dual/scale_*.
    ACTOR_DUAL_TOWER = True

    # Learnable per-tower scales -- with running-norm this is the ONLY channel
    # for tower reweighting (measured: scale_k +30% over a run). Keep on.
    ACTOR_DUAL_LEARN_SCALE = True
    # Prior tower weight at init; 1.0 = equal footing (post-normalization).
    ACTOR_DUAL_QUEUE_INIT_SCALE = 1.0

    # Zero-init nonlinear correction on [q,k]. Measured dual/grad_fuse stayed
    # ~0.02-0.03 for 200 eps (towers: 0.09-0.33) -- it learns nothing.
    ACTOR_DUAL_FUSE = False
    ACTOR_DUAL_FUSE_HIDDEN = 16

    # Divide each tower by a detached EMA of its cross-server spread, rescale
    # to tau: kills unbounded quality-spread drift so s_q/s_k stay meaningful.
    # Spreads are logged RAW; divisors as dual/rms_*.
    ACTOR_DUAL_RUNNING_NORM = True
    ACTOR_DUAL_RUNNING_NORM_MOMENTUM = 0.05
    # tau: initial logit softness (0.3 ~ 98% of max entropy). Learnable scales
    # take over from there. tau/rms also multiplies gradients -- see RMS_FLOOR.
    ACTOR_DUAL_NORM_TARGET_SPREAD = 0.3
    # Higher LR for the 2 scale params (their gradients are chain-rule tiny).
    ACTOR_DUAL_SCALE_LR = 1e-3

    # Queue tower reads raw queue length (not util); drain = queue/mu seconds.
    QUEUE_DESC_RAW_LOAD = True

    # Move price channel into the queue tower (quality tower becomes pure capability).
    DUAL_TOWER_PRICE_IN_QUEUE = False

    # Unit-scale queue features (superseded by ZSCORE below).
    QUEUE_DESC_UNIT_SCALE = False

    # Queue descriptor = [z(qload), residual, mu, z(drain)]: scale-invariant,
    # preserves who-is-busier. Fresh model (input dim changes).
    QUEUE_DESC_ZSCORE = True

    # Adds episode-cumulative alloc ratio to the queue head: the fairness
    # penalty is on cumulative flow, util is only the instantaneous stock.
    QUEUE_USE_ALLOC = True

    # Queue head size. Note running-norm pins the tower's output spread, so
    # capacity shapes opinions but cannot raise volume.
    QUEUE_HEAD_DEPTH = 2
    QUEUE_HEAD_HIDDEN = 0

    # Drain cap: episode proceeds with what completed; shortfall shows up as
    # outcome/incomplete_rate instead of hanging the run.
    EPISODE_COMPLETION_TIMEOUT = 360

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

    # Output cap. Moves fleet mu more than any other knob: re-measure
    # SERVICE_RATE after changing. Router QA generation controls below.
    GEN_MAX_NEW_TOKENS = 512         # hard cap on answer length
    GEN_MIN_NEW_TOKENS = 0
    GEN_TEMPERATURE = 0.1
    GEN_TOP_P = 1
    GEN_DO_SAMPLE = False


    # MCQ chain-of-thought. OFF puts the whole fleet at the random floor on
    # MMLU-Pro (0.00-0.25, measured) -- the task then carries no signal.
    MCQ_COT = True

    # OFF, measured: bounding MCQ reasoning cost mean mmlu 0.633->0.517 and
    # collapsed nano 0.708->0.333 (no truncation existed to be saved).
    MCQ_BRIEF_REASONING = False

    # ON, measured: brief math raised quality AND throughput (0.61->0.74;
    # truncation, not reasoning depth, dominated at the old cap).
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

    # Dependent variable: re-derive from sum(SERVICE_RATE) whenever fleet /
    # cap / mix change. Targets: overall rho ~0.5-0.7 (queues alive, not
    # drowning) AND rho_cheap = lambda/mu(cheapest-4) ~0.7-0.8 so cost-
    # seeking CAN concentrate (the FAIR-off ablation needs that room).
    POISSON_ARRIVAL_RATE = 2.5   # 8-fleet v2 mix: rho=0.69 (total mu 3.616)
    # POISSON_ARRIVAL_RATE = 2.65  # v2-easy pilot value (mu 3.951)
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
    # 5s x 10 intervals (was 7x8): same rho, 25% more interval-level PPO
    # samples per episode (the binding sample count -- the policy loss sees T
    # advantages, not T*N_t), ~10% fewer requests, shorter wall clock.
    # delta_t = 5s is still 2-3.5 mean service times (1.4-2.5s), so telemetry
    # staleness -- the TMDP premise -- remains material. The v-calibration is
    # delta_t-invariant by construction, so the delta_t ablation (e.g. {3,5,8}
    # or including 7) needs no fairness re-tuning.
    INTERVAL_LENGTH = 5 # The length of interval
    MAX_EPISODES = 150   # match LR_DECAY_EPISODES above

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
    FAIRNESS_MODE = "wf_dual"   # "wf_dual" | "quota" (legacy per-server ReLU) | "legacy"
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

    # The same choice applied to the mu the POLICY sees in its state, rather
    # than to the quota. Without it the state carried the online EMA, so the
    # value sitting in what the dual tower calls its STATIC capability channel
    # drifted throughout training -- and drifted in a way the policy itself
    # caused, since mu_hat = completions / service time and service time depends
    # on which task types were routed there. Three consequences: the static
    # channel was not static; dual/quality_spread mixed capability learning with
    # mu drift, so it stopped being a clean diagnostic; and re-measuring
    # SERVICE_RATE between runs shifted the input distribution the tower had
    # learned against, which this file has now done five times. The EMA is still
    # estimated and logged under service_rate/*, it just no longer feeds the
    # policy. False restores the old behaviour.
    STATE_USE_FROZEN_MU = True

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
    # 0: the water-filling quota targets the interval-start backlog only.
    # The backlog ALREADY carries the history physically -- a server
    # over-allocated in interval t drains slower and enters t+1 with larger
    # D_m, so water filling automatically compensates past imbalance. Adding
    # H_cum on top double-counts that memory, and under wf_dual it would also
    # distort the regret reference (v is computed against the same
    # d_for_quota). Keep 0 unless deliberately studying long-horizon
    # entitlement, which is a different objective than per-interval regret.
    QUOTA_HISTORY_WEIGHT = 0.0
    # Legacy knobs, IGNORED in wf_dual mode (strength lives in delta + nu).
    FAIR_TARGET = 1     # 最终的 FAIR 值 (legacy modes only)
    FAIR = 1           # 起始（trainer 会覆盖; legacy modes only）

    # =========================================================
    # Lagrangian (adaptive) fairness — RCPO, Tessler et al. 2018
    # =========================================================
    # FAIR is a PRICE, not a CONSTRAINT: once the quality gain of concentrating
    # Legacy Jain-floor dual (quota mode only; wf_dual has its own dual).
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

    # ---- WF-dual fairness (constrained, RCPO) ----
    # base_t = softmin_beta over request rewards; v = WF regret / iid quota-
    # sampling reference (E[v]=1 for that reference; NOT a floor -- min is 0).
    # tr_t = base_t - nu*(v - (1+delta)); dual: nu tracks the Rao-Blackwell
    # vhat, complementary slackness makes the penalty (and its noise) vanish
    # when slack. FAIR-off ablation: FAIR_DUAL_ENABLE=False, FAIR_MU_INIT=0.
    # delta floor: stochastic policies pay v_sampling ~1.02, so delta=0 is
    # infeasible while exploring (measured: nu saturates); 0.25 = strict.
    FAIR_DELTA = 0.25          # constraint level: E[v] <= 1 + delta
    FAIR_DUAL_ENABLE = True   # False => mu_lag frozen at FAIR_MU_INIT
    FAIR_MU_INIT = 0.0
    FAIR_DUAL_LR = 0.05       # eta (two-timescale: slower than the policy)
    FAIR_DUAL_EMA = 0.1       # alpha for the vbar EMA
    FAIR_MU_MAX = 5.0
    WF_TILT_BETA = -2.0       # softmin tilt over request rewards
    WF_EPS_DBAR = 1e-6        # T_fair guard: exclude intervals with Dbar below
    QUOTA_TIEBREAK = "lex"    # deterministic quota tie-break ("closest" = legacy)
    # gamma for INTERVAL returns/GAE. 1.0 aligns the actor objective with the
    # undiscounted per-episode constraint statistic the dual uses (finite
    # horizon T, so 1.0 is safe). Request-level diagnostics keep Config.GAMMA.
    INTERVAL_GAMMA = 1.0
    # Critic learns the residual return Y = R - b (b = V^task prompt baseline)
    # instead of R, so the s-conditional mean is not subtracted twice; the
    # advantage A = Y - V is unchanged in value, only the critic target moves.
    CRITIC_RESIDUAL_TARGET = True
    # Additive critic conditioning on mu_lag/mu_max: mu shifts the augmented
    # return of the same state (tr = base - mu*v), so a state-only critic sees
    # a nonstationary target as the dual moves. V^task does NOT get mu (its
    # target r_i contains no fairness penalty).
    CRITIC_INPUT_MU = True
    ROUTE_DIST_LOG = True     # per-task routing distribution + pairwise JS
    # Freeze rms per episode: rollout and replay share one divisor (exact
    # epoch-0 log-prob parity); live EMA commits after PPO for next episode.
    ACTOR_DUAL_RMS_FREEZE = True
    # Rate-limit each commit (unclamped first commit once amplified logits
    # 10x overnight and collapsed the policy).
    ACTOR_DUAL_RMS_COMMIT_RATIO = 1.5
    # Seed rms from the first forward's measured spread (constants can't be
    # right: untrained queue spread varies 6x across inits).
    ACTOR_DUAL_RMS_SELF_SEED = True
    # Floor on the divisors: tau/rms multiplies logits AND gradients, so this
    # caps amplification at 6x (0.01 floor meant 30x -> one-update collapse).
    ACTOR_DUAL_RMS_FLOOR = 0.05
    # Cap on the divisors, symmetric to the floor. With RESCALE below, floor
    # and cap are dormant fuses (frozen rms is pinned at 1.0).
    ACTOR_DUAL_RMS_CAP = 1.0
    # Rescale-reset at commit (forced weight normalization, EDM2-style):
    # divide each tower's output Linear by its live rms, reset rms to 1.
    # Policy-invariant (no KL kick). Kills both observed failure modes:
    # rms chasing spread growth decays effective LR (0.17->12, KL 1e-5,
    # frozen policy); capping rms lets normalized spread grow unbounded
    # (max_share 0.88 collapse onto 3b). Sharpening now only through the
    # learnable tower scales.
    ACTOR_DUAL_RMS_RESCALE = True

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

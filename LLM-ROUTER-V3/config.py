import torch

class Config:
    MODEL_NAMES = [
        # ===== Tier 1: 最便宜 =====
        "ministral-3b-2512",

        # ===== Tier 2: 便宜 weak baseline =====
        # "together/Qwen/Qwen3.5-9B",
        # "ministral-8b-2512",
        "gpt-4.1-nano-2025-04-14",

        # ===== Tier 3: mid-tier =====
        # "together/Qwen/Qwen2.5-7B-Instruct-Turbo",
        "mistral-small-2603",
        "gpt-4.1-mini-2025-04-14",

        # ===== Tier 4: 强模型 =====
        "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        # "mistral-large-2512",
        # "gpt-4.1-2025-04-14",
        "together/openai/gpt-oss-120b",
    ]
    # MODEL_NAMES = [
    #     "ministral-3b-2512",                            # 0 Mistral
    #     "gpt-4.1-nano-2025-04-14",                      # 1 OpenAI
    #     "mistral-small-2603",                           # 2 Mistral
    #     "gpt-4.1-mini-2025-04-14",                      # 3 OpenAI
    #     "novita/meta-llama/llama-3.3-70b-instruct",     # 4 Novita
    #     "novita/deepseek/deepseek-v3.2",                # 5 Novita  ⚠️核对ID
    # ]

    PRICE = [
        # ===== Tier 1: 最便宜 =====
        (0.00000010, 0.00000010),  # ministral-3b-2512

        # ===== Tier 2: 便宜 weak baseline =====
        # (0.00000017, 0.00000025),  # novita/meta-llama/llama-3.1-8b-instruct
        # (0.00000015, 0.00000015),  # ministral-8b-2512
        (0.00000010, 0.00000040),  # gpt-4.1-nano-2025-04-14

        # ===== Tier 3: mid-tier =====
        # (0.00000030, 0.00000030),  # together/Qwen/Qwen2.5-7B-Instruct-Turbo
        (0.00000015, 0.00000060),  # mistral-small-2506
        (0.00000040, 0.00000160),  # gpt-4.1-mini-2025-04-14

        # ===== Tier 4: 强模型 =====
        (0.00000104, 0.00000104),  # together/meta-llama/Llama-3.3-70B-Instruct-Turbo
        # (0.00000050, 0.00000150),  # mistral-large-2512
        # (0.00000200, 0.00000800),  # gpt-4.1-2025-04-14
        (0.00000015, 0.00000060),  # gpt-oss-120b

    ]

    # PRICE = [
    #     (0.00000010, 0.00000010),   # 0 ministral-3b
    #     (0.00000010, 0.00000040),   # 1 gpt-4.1-nano
    #     (0.00000015, 0.00000060),   # 2 mistral-small
    #     (0.00000040, 0.00000160),   # 3 gpt-4.1-mini
    #     (0.000000135, 0.00000040),  # 4 novita llama-3.3-70b
    #     (0.000000269, 0.00000040),  # 5 novita deepseek-v3.2
    # ]

    SERVICE_RATE = [
        0.663,   # ministral-3b
        # 0.5605,   # llama-3.1-8b (novita)
        # 0.4924,   # ministral-8b
        0.745,   # gpt-4.1-nano
        # 0.4137,   # Qwen2.5-7B
        0.639,   # mistral-small
        0.565,   # gpt-4.1-mini
        0.471,   # Llama-3.3-70B
        # 0.2065,   # mistral-large
        0.313,   # gpt-oss-120b
    ]
    SERVER_CAPACITIES = [50] * 6

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
            "levels": None,
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
    MAX_LAT = 60
    # SLO latency thresholds (seconds). Logged as violation rate =
    # fraction of completed requests with end-to-end latency > T.
    # Report a few (tight/moderate/loose); keep all below MAX_LAT.
    SLO_LATENCIES = [5.0, 10.0, 20.0]
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
    ENTROPY_COEF = 0.01
    ACTOR_LEARNING_RATE = 5e-5
    CRITIC_LEARNING_RATE = 3e-4
    USE_LR_DECAY = True
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

    # Interval weighting in the PPO policy loss.
    #   False (default): equal weight per INTERVAL (mean of per-interval means)
    #                    -> a request in a sparse interval counts more.
    #   True           : weight each interval by its arrival count N_t
    #                    -> equal weight per REQUEST (grand mean), lower
    #                    variance from small intervals. Reward normalization
    #                    (1/M or 1/N_t) is unaffected; this only reweights loss.
    PPO_LOSS_WEIGHT_BY_ARRIVALS = False

    # The ep20+ slide happened at KL 0.007-0.015 — entirely below the old
    # 0.04 target, so the early stop never fired. 0.012 clamps the late
    # acceleration while passing normal mid-run learning (0.002-0.003).
    TARGET_KL = 0.012
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
    ACTOR_DUAL_TOWER = False

    # (dead config, nothing reads it; the episode-completion wait loop in
    # trainer.py is unbounded — API-client timeouts/retries bound it in
    # practice)
    # EPISODE_COMPLETION_TIMEOUT = 180

    SERVICE_RATE_EMA_ALPHA = 0.1
    SERVICE_RATE_MIN_SAMPLES = 1
    SERVICE_RATE_MIN = 1e-4
    SERVICE_RATE_MAX = 5.0


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
    GEN_MAX_NEW_TOKENS = 768         # hard cap on answer length
    GEN_MIN_NEW_TOKENS = 0
    GEN_TEMPERATURE = 0.1
    GEN_TOP_P = 1
    GEN_DO_SAMPLE = False


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


    # Quality scoring settings
    # Realigned to the 6-server list (the old dict was keyed for 10
    # servers; after the cut, index 4 = Llama-70B would have read 700).
    MODEL_ELO_SCORES = {
        0: 700,   # ministral-3b
        1: 1200,  # gpt-4.1-nano
        2: 1100,  # mistral-small
        3: 1300,  # gpt-4.1-mini
        4: 1500,  # Llama-3.3-70B
        5: 1500,  # gpt-oss-120b
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
    POISSON_ARRIVAL_RATE = 3  # Average arrival rate of prompts per second
    MAX_PROMPT_QUEUE_SIZE = 10000  # Maximum size of the prompt queue
    EPISODE_TIME_INTERVAL = 8 # How many intervals in current episode

    # Training settings
    EPISODE_LENGTH = 100  # Number of prompts per episode (increased for better learning)
    INTERVAL_LENGTH = 5 # The length of interval
    MAX_EPISODES = 200   # match LR_DECAY_EPISODES above

    # Queue score settings
    QUEUE_SCORE_FACTOR = 0.2  # Factor to adjust queue score impact
    QUEUE_EPSILON = 0.0001  # Epsilon for queue score stability
    MERGE_ALPHA = 0 # Alpha for merging action probabilities (0.5 for equal weighting)

    # Drop action
    INVALID_ROUTE_PENALTY = 2/3   # try 0.5 ~ 2.0 depending how hard you want to avoid full servers
    FAIL_LATENCY_CAP = 60.0       # just for logging; failed branch uses penalty not latency
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
    FAIR_TARGET = 1      # 最终的 FAIR 值
    FAIR = 1            # 起始（trainer 会覆盖）

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

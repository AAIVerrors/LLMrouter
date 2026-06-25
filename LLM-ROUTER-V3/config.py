import torch

class Config:
    MODEL_NAMES = [
        # ===== Tier 1: 最便宜 =====
        "ministral-3b-2512",

        # ===== Tier 2: 便宜 weak baseline =====
        "together/meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        "ministral-8b-2512",
        "gpt-4.1-nano-2025-04-14",

        # ===== Tier 3: mid-tier =====
        "together/Qwen/Qwen2.5-7B-Instruct-Turbo",
        "mistral-small-2603",
        "gpt-4.1-mini-2025-04-14",

        # ===== Tier 4: 强模型 =====
        "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "mistral-large-2512",
        # "gpt-4.1-2025-04-14",
        "together/openai/gpt-oss-120b",
    ]

    PRICE = [
        # ===== Tier 1: 最便宜 =====
        (0.00000010, 0.00000010),  # ministral-3b-2512

        # ===== Tier 2: 便宜 weak baseline =====
        (0.00000014, 0.00000014),  # together/meta-llama/Meta-Llama-3-8B-Instruct-Lite
        (0.00000015, 0.00000015),  # ministral-8b-2512
        (0.00000010, 0.00000040),  # gpt-4.1-nano-2025-04-14

        # ===== Tier 3: mid-tier =====
        (0.00000030, 0.00000030),  # together/Qwen/Qwen2.5-7B-Instruct-Turbo
        (0.00000015, 0.00000060),  # mistral-small-2506
        (0.00000040, 0.00000160),  # gpt-4.1-mini-2025-04-14

        # ===== Tier 4: 强模型 =====
        (0.00000104, 0.00000104),  # together/meta-llama/Llama-3.3-70B-Instruct-Turbo
        (0.00000050, 0.00000150),  # mistral-large-2512
        # (0.00000200, 0.00000800),  # gpt-4.1-2025-04-14
        (0.00000015, 0.00000060),  # 9 gpt-oss-120b
        
    ]

    SERVICE_RATE = [
        0.5147,   # 0 ministral-3b
        0.5605,   # 1 Meta-Llama-3-8B-Lite
        0.4924,   # 2 ministral-8b
        0.7042,   # 3 gpt-4.1-nano
        0.4137,   # 4 Qwen2.5-7B
        0.3352,   # 5 mistral-small
        0.5391,   # 6 gpt-4.1-mini
        0.1351,   # 7 Llama-3.3-70B
        0.2065,   # 8 mistral-large
        0.2036,   # 9 gpt-oss-120b
    ]
    SERVER_CAPACITIES = [50] * 10

    USE_UTIL = True  # in the state use load/capability or load + capability

    # Dataset settings (ONE dataset per run)
    # Examples:
    #   - "tatsu-lab/alpaca"
    #   - "hotpotqa/hotpot_qa"   (set DATASET_CONFIG to "distractor" or "fullwiki")
    #   - "squad"
    #   - "cais/mmlu" - DATASET_CONFIG = "all" - DATASET_SPLIT = "auxiliary_train[:20000]"
    #   - "mixed" - "None" - "Train"
    DATASET_NAME = "hotpotqa/hotpot_qa"
    DATASET_CONFIG = "distractor"    # Optional HF config name (e.g., HotpotQA: "distractor" / "fullwiki")
    DATASET_SPLIT = "train"     # "train" / "validation" / "test" (must exist in the dataset)
    MAX_SAMPLES = 50000         # Optional cap for faster experiments
    SHUFFLE_DATASET = True
    DATASET_SEED = 42

    # Add this inside class Config
    USE_MIXED_DATASET = False  # If True, use a mixture of datasets instead of a single one.

    MIXED_DATASETS = [
        # Multi-hop QA, needs context; metric uses token F1
        {
            "name": "hotpotqa/hotpot_qa",
            "config": "distractor",
            "split": "train",
            "weight": 0.25,
            "metric": "f1",
            "task_type": "multihop_qa",
            "max_samples": 20000,
        },
        # Reading comprehension / easier QA, metric uses token F1
        {
            "name": "squad",
            "config": None,
            "split": "train[:20000]",
            "weight": 0.25,
            "metric": "f1",
            "task_type": "qa",
        },
        # Multiple-choice knowledge/reasoning, metric checks A/B/C/D
        {
            "name": "cais/mmlu",
            "config": "all",
            "split": "auxiliary_train[:20000]",
            "weight": 0.25,
            "metric": "mmlu",
            "task_type": "mmlu",
        },
        # Math reasoning, metric extracts exact final number
        {
            "name": "openai/gsm8k",
            "config": "main",
            "split": "train[:20000]",
            "weight": 0.25,
            "metric": "number",
            "task_type": "math",
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
    MAX_LAT = 30
    FAIR_REWARD_MIN_FLOOR = False # True the missing server will be set min rewards, False will use the floor reward -Beta-REWARD_GAMMA
    
    # PPO hyperparameters - tuned for the routing problem
    LEARNING_RATE = 1e-4 # Reduced for more stable learning
    GAMMA = 0.99          # Slightly reduced discount factor
    GAE_LAMBDA = 0.95      # Reduced for less variance in advantage estimation
    CLIP_EPSILON = 0.2    # Slightly reduced for more conservative updates
    POLICY_COEF = 1       # Policy loss weight
    VALUE_COEF = 1      # Reduced value function weight
    ENTROPY_COEF = 0.0   # Increased entropy for more exploration
    ACTOR_LEARNING_RATE = 5e-6
    CRITIC_LEARNING_RATE = 1e-5
    USE_LR_DECAY = True
    LR_DECAY_TYPE = "cosine"
    LR_DECAY_MIN_RATIO = 0.1
    LR_WARMUP_EPISODES = 0
    KL_COEF = 0.00
    MAX_GRAD_NORM = 1 # Reduced for more stable training
    PPO_EPOCHS = 4   # Increased for more thorough updates
    BATCH_SIZE = 1      # Increased batch size

    TARGET_KL = 0.04
    USE_TARGET_KL_STOP = False

    USE_PER_INTERVAL_MINIBATCH = True
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
    
    EPISODE_COMPLETION_TIMEOUT = 180
    
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
    GEN_MAX_NEW_TOKENS = 256         # hard cap on answer length
    GEN_MIN_NEW_TOKENS = 0
    GEN_TEMPERATURE = 0.1         
    GEN_TOP_P = 1
    GEN_DO_SAMPLE = False

    
    # Encourage a parseable final answer
    QA_PROMPT_STYLE = "plain"     # "instruction"/"alpaca" or "plain"
    QA_FORCE_FINAL_TAG = True 
    FINAL_ANSWER_TAG = "final"
    TRUNCATE_AT_FINAL_TAG = True
    OUTPUT_FINAL_ONLY = True           # if True, store only <final>...</final> as response_text
    PRICE_USE_RAW_RESPONSE = True       # price penalty uses raw output length (more realistic)

    MISTRAL_MAX_RETRIES = 2
    API_TRANSIENT_FAIL_PENALTY = 0.0

    
    # Quality scoring settings
    MODEL_ELO_SCORES = {
        0: 700,  # GPT-2 base ELO
        1: 1100,  # Qwen base ELO
        2: 700,  # GPT-2 base ELO
        3: 1500,
        4: 700,  # GPT-2 base ELO
        5: 1100,
        6: 700,  # GPT-2 base ELO
        7: 1500,
        8: 700,  # GPT-2 base ELO
        9: 1100,
    }
    
    USE_ATTN_ROUTER = False

    # ====================================================================
    # [CHANNEL] Dual-channel attention router: split per-server features
    # into dynamic (util only) and static (mu, prices) channels.
    # ====================================================================
    SERVER_DYN_DIM  = 1    # util, slot_count, interval_norm, time_remaining_norm  
    SERVER_STAT_DIM = 3    # mu, price_in, price_out                         

    
    # Load factor settings
    MAX_LOAD_FACTOR = 1.5

    # Final evaluation settings
    EVAL_EPISODES = 5         # Number of episodes for evaluation
    FINAL_EVAL_EPISODES = 10  # Number of episodes for final evaluation
    
    # Poisson prompt generation settings
    POISSON_ARRIVAL_RATE = 5  # Average arrival rate of prompts per second
    MAX_PROMPT_QUEUE_SIZE = 10000  # Maximum size of the prompt queue
    EPISODE_TIME_INTERVAL = 8 # How many intervals in current episode
    
    # Training settings
    EPISODE_LENGTH = 100  # Number of prompts per episode (increased for better learning)
    INTERVAL_LENGTH = 3 # The length of interval
    MAX_EPISODES = 200   # Increased for more training
    
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
    
    USE_AVG = False 
    
    RANDOM_SELECT = False

    NAIVE_PPO = False

    ENABLE_QUEUE_PENALTY = False

    JSQ = False

    P2C = False

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
    FAIR_TARGET = 1      # 最终的 FAIR 值
    FAIR = 1             # 起始（trainer 会覆盖）

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
    # EM_METRIC = "mmlu"
    EM_METRIC = "f1"

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
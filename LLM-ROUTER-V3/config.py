import torch
import time

class Config:
    # Model settings
    # MODEL_NAMES = [
    #     # "lmsys/fastchat-t5-3b-v1.0",
    #     "google/gemma-1.1-2b-it",
    #     # "google/gemma-2-2b-it",
    #     "google/gemma-2b-it",
    #     "ibm-granite/granite-3.0-2b-instruct",
    #     "ibm-granite/granite-3.1-2b-instruct",
    #     "meta-llama/Llama-3.2-1B-Instruct",
    #     # "meta-llama/Llama-3.2-3B-Instruct",
    #     # "microsoft/Phi-3-mini-128k-instruct",
    #     # "microsoft/Phi-3-mini-4k-instruct",
    # ]


    MODEL_NAMES = [
        # "gpt-5-2025-08-07",
        # "gpt-5-mini-2025-08-07",
        # "gpt-5-nano-2025-08-07",
        # 'gpt-4o-mini-2024-07-18',
        # 'gpt-5-nano-2025-08-07',
        'gpt-4.1-2025-04-14',
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
        "ministral-8b-2410",
        # "mistralai/Ministral-8B-Instruct-2410",
        "mistral-7b-instruct-v0.2", # open-mistral-7b
        # "mistralai/Mistral-7B-Instruct-v0.3",
        "mistral-medium",  # mistral-medium-2508
        "mixtral-8x7b-instruct-v0.1", # open-mixtral-8x7b
        
        "mistral-large-2512",
        "mistral-large-2411",
        "labs-mistral-small-creative",
        "mistral-medium-2505",
        
        "mistral-small-2506",
        # "magistral-medium-2509",
        "ministral-14b-2512",
        # "mistralai/Ministral-3-14B-Instruct-2512",
        "ministral-3b-2512",
        # "mistralai/Ministral-3-3B-Instruct-2512",
        # "magistral-small-2509"
    ]
    

    SERVICE_RATE = [
        0.318878,  # gpt-4.1-2025-04-14
        0.275670,  # gpt-4.1-mini-2025-04-14
        0.634155,  # gpt-4.1-nano-2025-04-14
        0.711388,  # ministral-8b-2410
        0.359749,  # mistral-7b-instruct-v0.2
        0.181186,  # mistral-medium
        0.563756,  # mixtral-8x7b-instruct-v0.1
        0.241330,  # mistral-large-2512
        0.298681,  # mistral-large-2411
        0.574042,  # labs-mistral-small-creative
        0.215799,  # mistral-medium-2505
        0.485690,  # mistral-small-2506
        # 0.101311,  # magistral-medium-2509
        0.541000,  # ministral-14b-2512
        0.810045,  # ministral-3b-2512
        # 0.291055,  # magistral-small-2509
    ]


    PRICE = [
        (0.000002, 0.000008), # gpt-4.1-2025-04-14
        (0.0000004, 0.0000016),# "gpt-4.1-mini-2025-04-14" 
        (0.0000001, 0.0000004), # gpt-4.1-nano-2025-04-14
        # (0.00000125, 0.00001), # gpt-5-2025-08-07
        # (0.00000025, 0.000002), # gpt-5-mini-2025-08-07
        # (0.00000005, 0.0000004), # gpt-5-nano-2025-08-07
        # (0.00000015, 0.0000006), # gpt-4o-mini-2024-07-18
        # (0.00000005, 0.0000004), # gpt-5-nano-2025-08-07
        (0.00000015, 0.00000015), # ministral-8b-2410
        (0.00000025, 0.00000025), # mistral-7b-instruct
        (0.0000004, 0.000002), # mistral-medium
        (0.0000007, 0.0000007), # mixtral-8x7b-instruct-v0.1
        (0.0000005, 0.0000015), # "mistral-large-2512"
        (0.000002, 0.000006),  # mistral-large-2411
        (0.0000001,0.0000003),  # labs-mistral-small-creative
        (0.0000004,0.000002),  # mistral-medium-2505
        (0.0000001, 0.0000003), # "mistral-small-2506"
        # (0.000002, 0.000005), # "magistral-medium-2509"
        (0.0000002, 0.0000002),# "ministral-14b-2512"
        (0.0000001, 0.0000001),# "ministral-3b-2512"
        # (0.0000005, 0.0000015),# "magistral-small-2509"
    ]
    
    # Model settings
    # MODEL_NAMES = [
    #     # # 'meta-llama/Llama-2-13b-chat-hf',
    #     # # 'meta-llama/Meta-Llama-3-8B-Instruct',
    #     # 'allenai/Llama-3.1-Tulu-3-8B',
    #     # # 'meta-llama/Llama-3.1-8B-Instruct',
    #     # # 'mistralai/Ministral-8B-Instruct-2410',
    #     # # 'mistralai/Mistral-7B-Instruct-v0.2',
    #     # # # "lmsys/fastchat-t5-3b-v1.0",
    #     # "google/gemma-1.1-2b-it",
    #     # # "google/gemma-7b",
    #     # # "google/gemma-2-2b-it",
    #     # # "google/gemma-2b-it",
    #     # # "ibm-granite/granite-3.0-2b-instruct",
    #     # # "ibm-granite/granite-3.1-2b-instruct",
    #     # # "meta-llama/Llama-3.2-1B-Instruct",
    #     # # "ibm-granite/granite-3.0-2b-instruct",
    #     # # "ibm-granite/granite-3.1-2b-instruct",
    #     # "meta-llama/Llama-3.2-1B-Instruct",
    #     # "meta-llama/Llama-3.2-3B-Instruct",
    #     # "microsoft/Phi-3-mini-128k-instruct",
    #     # "microsoft/Phi-3-mini-4k-instruct",
    #     # 'gpt-3.5-turbo-0125',
    #     # # 'gpt-3.5-turbo-1106',
    #     # 'gpt-4o-2024-08-06',
    #     # 'gpt-4o-mini-2024-07-18',
    #     # 'gpt-5-nano-2025-08-07',
    #     # 'gpt-4.1-nano-2025-04-14',
    #     # 'o3-mini',
    #     # 'o1-mini',
    #     # 'gemini-2.0-flash-001',
    #     # 'gemini-2.0-flash-exp',
    #     # 'gemini-1.5-flash-001',
    #     # 'gemini-1.5-flash-002',
    #     # 'gemini-1.5-flash-8b-001',
    #     # 'claude-3-5-haiku-20241022',
    #     # 'claude-3-haiku-20240307',
    #     # 'claude-3-7-sonnet-20250219',
    #     # 'claude-3-5-sonnet-20240620',
    #     "ministral-8b-2410",
    #     # "mistral-7b-instruct-v0.2", # open-mistral-7b
    #     "mistral-medium",  # mistral-medium-2508
    #     # "mistral-small-24b-instruct-2501", # mistral-small-2501
    #     # "mixtral-8x22b-instruct-v0.1", # open-mixtral-8x22b
    #     "mixtral-8x7b-instruct-v0.1", # open-mixtral-8x7b
    #     # new 
    #     # "gpt-4.1-mini-2025-04-14",
    #     "mistral-large-2512",
    #     # "mistral-small-2506",
    #     # "magistral-medium-2509",
    #     # "ministral-14b-2512",
    #     "ministral-3b-2512",
    #     "magistral-small-2509"
    # ]

    # SERVICE_RATE = [
    #     # 0.266312,
    #     # 0.460004,
    #     # 0.736517,
    #     0.675391,
    #     # 0.800053,
    #     0.218149,
    #     0.655419,
    #     # 0.642034,
    #     0.137048,
    #     # 0.275566,
    #     # 0.198817,
    #     # 0.594973,
    #     0.404235,
    #     0.285589,
    # ]
    
    # PRICE = [
    #     # (0.00000025, 0.00000025),
    #     # (0.00000025, 0.00000025),
    #     # (0.0000001, 0.0000001),
    #     # (0.0000001, 0.0000001),
    #     # (0.0000001, 0.0000001), # allenai/Llama-3.1-Tulu-3-8B
    #     # # (0.0000001, 0.0000001),
    #     # (0.000000025, 0.000000025), # "google/gemma-1.1-2b-it"
    #     # # (0.0000001, 0.0000001),
    #     # (0.0000000125, 0.0000000125), # meta-llama/Llama-3.2-1B-Instruct
    #     # (0.0000000375, 0.0000000375), # meta-llama/Llama-3.2-3B-Instruct
    #     # (0.00000015, 0.0000006), # gpt-4o-mini-2024-07-18
    #     # (0.00000005, 0.0000004), # gpt-5-nano-2025-08-07
    #     # (0.0000001, 0.0000004), # gpt-4.1-nano-2025-04-14
    #     # (0.00000015, 0.0000006), # genmi-2.0-flash-exp
    #     # (0.000000075, 0.0000003),  # gemini-1.5-flash-001
    #     # (0.000000075, 0.0000003),  # gemini-1.5-flash-002
    #     # (0.0000000375, 0.00000015),  # gemini-1.5-flash-8b-001
    #     # (0.0000008, 0.000004), # claude-3-5-haiku-20241022
    #     # (0.00000025, 0.00000125), # claude-3-haiku-20240307
    #     (0.00000015, 0.00000015), # ministral-8b-2410
    #     # (0.00000025, 0.00000025), # mistral-7b-instruct
    #     (0.0000004,0.000002), # mistral-medium
    #     # (0.0000001, 0.0000003), # mistral-small-24b-instruct-2501
    #     (0.0000007, 0.0000007), # mixtral-8x7b-instruct-v0.1
    #     # (0.0000004, 0.0000016),# "gpt-4.1-mini-2025-04-14" 
    #     (0.0000005, 0.0000015),# "mistral-large-2512"
    #     # (0.0000001, 0.0000003),# "mistral-small-2506"
    #     # (0.000002, 0.000005),# "magistral-medium-2509"
    #     # (0.0000002, 0.0000002),# "ministral-14b-2512"
    #     (0.0000001, 0.0000001),# "ministral-3b-2512"
    #     (0.0000005, 0.0000015),# "magistral-small-2509"
    # ]
    
    # Server capabilities (max concurrent requests)
    SERVER_CAPACITIES = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,]  # Capacity for each model

    USE_UTIL = True # in the state use load/capability or load + capability
    
    # Dataset settings (ONE dataset per run)
    # Examples:
    #   - "tatsu-lab/alpaca"
    #   - "hotpotqa/hotpot_qa"   (set DATASET_CONFIG to "distractor" or "fullwiki")
    #   - "squad"
    DATASET_NAME = "hotpotqa/hotpot_qa"
    DATASET_CONFIG = "distractor"     # Optional HF config name (e.g., HotpotQA: "distractor" / "fullwiki")
    DATASET_SPLIT = "train"     # "train" / "validation" / "test" (must exist in the dataset)
    MAX_SAMPLES = 20000         # Optional cap for faster experiments

    # Prompt encoder settings (RouterNetwork)
    # Any SentenceTransformer model name, e.g., 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', etc.
    PROMPT_MODEL = "all-MiniLM-L6-v2"

    # If False: use raw SentenceTransformer embedding directly (no projection).
    # If True: learn a small projection emb_dim -> PROMPT_DIM.
    USE_PROMPT_PROJECTION = False
    PROMPT_DIM = 64


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

    ROUTER_LLM_CRITIC_HIDDEN = 521
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
    # "instruction":  Instruction/Input/Response (matches your previous Alpaca-style prompt)
    # "plain":        Question/Context/Answer
    # QA_PROMPT_STYLE = "plain"
    QA_INCLUDE_CONTEXT = False
    QA_MAX_CONTEXT_DOCS = 8      # For datasets with multiple context documents (e.g., HotpotQA)
    QA_MAX_CONTEXT_CHARS = 2500  # Hard cap to avoid overly long prompts

    # Scoring: extract a final answer span before EM/F1 (prevents explanations from lowering scores)
    EXTRACT_FINAL_ANSWER = True
    FINAL_ANSWER_TAG = "final"
    
    # Start time for frequent requests
    START_TIME = time.time()  # Start time for the first request
    
    # Training settings
    EPISODE_LENGTH = 100  # Number of prompts per episode (increased for better learning)
    INTERVAL_LENGTH = 1
    MAX_EPISODES = 200   # Increased for more training
    
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
    FAIR_REWARD_MIN_FLOOR = True # True the missing server will be set min rewards, False will use the floor reward -Beta-REWARD_GAMMA
    
    # PPO hyperparameters - tuned for the routing problem
    LEARNING_RATE =1e-5  # Reduced for more stable learning
    GAMMA = 0.99          # Slightly reduced discount factor
    GAE_LAMBDA = 0.95      # Reduced for less variance in advantage estimation
    CLIP_EPSILON = 0.2    # Slightly reduced for more conservative updates
    POLICY_COEF = 1       # Policy loss weight
    VALUE_COEF = 0.5      # Reduced value function weight
    ENTROPY_COEF = 0.1   # Increased entropy for more exploration
    KL_COEF = 0.01
    MAX_GRAD_NORM = 0.5
    PPO_EPOCHS = 3        # Increased for more thorough updates
    BATCH_SIZE = 1      # Increased batch size
    # WEIGHT_DECAY = 1e-5
    USE_SERVERWISE_MLP = True

    
    # Neural network settings
    HIDDEN_DIM = 512      # Increased capacity
    INPUT_DIM = 256       # Embedding dimension for prompts
    ATTENTION_HEADS = 8  # Number of attention heads
    
    # Environment settings
    TIME_STEP = 0.1       # Time increment per environment step
    COMPLETION_CHECK_STEPS = 20  # Steps to wait for request completion at episode end
    
    # Device settings
    GPU_LIST = [0]
    DEVICE = torch.device("cuda:0")
    
    # Wandb settings
    WANDB_PROJECT = "enhanced-llm-router-ppo"
    WANDB_ENTITY = None  # Set your wandb entity if needed
    
    # Logging
    LOG_INTERVAL = 5      # Log every 5 episodes
    SAVE_INTERVAL = 25    # Save every 25 episodes
    EVAL_INTERVAL = 15    # Evaluate every 15 episodes
    PLOT_INTERVAL = 50    # Plot progress every 50 episodes
    
    # # Text generation settings (for actual LLM inference)
    # MAX_LENGTH = 512
    # TEMPERATURE = 0.7

    # Router QA generation controls (keeps answers short & deterministic)
    # NOTE: MAX_LENGTH is kept for backward compatibility; use GEN_* for routing QA.
    GEN_MAX_NEW_TOKENS = 1024         # hard cap on answer length
    GEN_MIN_NEW_TOKENS = 0
    GEN_TEMPERATURE = 0.7           
    GEN_TOP_P = 0.95
    GEN_DO_SAMPLE = False

    
    # Encourage a parseable final answer
    QA_PROMPT_STYLE = "plain"     # "instruction"/"alpaca" or "plain"
    QA_FORCE_FINAL_TAG = True
    FINAL_ANSWER_TAG = "final"
    TRUNCATE_AT_FINAL_TAG = True
    OUTPUT_FINAL_ONLY = True           # if True, store only <final>...</final> as response_text
    PRICE_USE_RAW_RESPONSE = True       # price penalty uses raw output length (more realistic)

    
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
    
    # Latency simulation settings
    MODEL_LATENCY_RANGES = {
        "gpt2": (0.5, 1.0),      # GPT-2 latency range in seconds
        "qwen": (1, 2),      # Qwen latency range in seconds
        "default": (0.6, 1.1)    # Default latency range
    }

    USE_ATTN_ROUTER = True
    ATTN_D_MODEL = 512
    ATTN_N_HEADS = 8
    ATTN_N_LAYERS = 4
    ATTN_FF_MULT = 4
    ATTN_DROPOUT = 0.1
    ATTN_USE_GLOBAL_TOKEN = True

    
    # Load factor settings
    MAX_LOAD_FACTOR = 1.5     # Maximum latency increase due to server load
    COMPLETION_PROBABILITY = 0.3  # Probability of request completion per step
    
    # Final evaluation settings
    EVAL_EPISODES = 5         # Number of episodes for evaluation
    FINAL_EVAL_EPISODES = 10  # Number of episodes for final evaluation
    
    # Poisson prompt generation settings
    POISSON_ARRIVAL_RATE = 2  # Average arrival rate of prompts per second
    MAX_PROMPT_QUEUE_SIZE = 10000  # Maximum size of the prompt queue
    EPISODE_TIME_INTERVAL = 30  # Time interval for each episode in seconds
    
    # Queue score settings
    QUEUE_SCORE_FACTOR = 0.2  # Factor to adjust queue score impact
    QUEUE_EPSILON = 0.0001  # Epsilon for queue score stability
    MERGE_ALPHA = 0 # Alpha for merging action probabilities (0.5 for equal weighting)

    # Drop action
    INVALID_ROUTE_PENALTY = 0.7   # try 0.5 ~ 2.0 depending how hard you want to avoid full servers
    FAIL_LATENCY_CAP = 30.0       # just for logging; failed branch uses penalty not latency
    REWARD_CLIP = 2.0             # optional, set <=0 to disable

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
    UTILITY_W_Q = 0.0             # optional extra queue penalty beyond latency term
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
    GREEDY_EPSILON = 0.05
    
    # If > 0, adds a UCB-style bonus to uncertain servers in greedy utility.
    # Set 0.0 to disable.
    GREEDY_UCB_COEF = 0.0
    
    # If > 1, sample uniformly among the top-K highest-utility servers (simple exploration).
    # Set 1 to disable.
    GREEDY_TOPK = 1

    T = -0.5
    FAIR = 0  # 0..1, it will control how fair you want, 1 max, 0 min
    MEAN_IN_FAIR_REWARD = False
    T_QUEUE = -0.5
    T_REWARD = -0.5

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
    # =================================================================
    # LLM-as-a-judge quality evaluation (optional)
    # =================================================================
    # If True, compute quality_score using a reward model / judge AFTER generation.
    # This is typically more expensive than EM/F1 and should NOT be included in the RL state.
    USE_LLM_JUDGE = False

    # Reward-model judge (Hugging Face)
    LLM_JUDGE_MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

    # Inference settings for judge
    LLM_JUDGE_DEVICE = "cuda"          # "cuda" or "cpu"
    LLM_JUDGE_DTYPE = "float16"        # "float16" / "bfloat16" / "float32"
    LLM_JUDGE_ATTN_IMPL = "flash_attention_2"  # "flash_attention_2" / "eager" / "sdpa"
    LLM_JUDGE_MAX_LENGTH = 2048        # prompt+answer token budget for judge input
    LLM_JUDGE_BATCH_SIZE = 4

    # Preload judge model at worker start (recommended)
    LLM_JUDGE_PRELOAD = True


    # Normalize raw reward-model logit to [0, 1]
    #   - "sigmoid": 1 / (1 + exp(-k * raw))
    #   - "tanh":    0.5 * (tanh(k * raw) + 1)
    #   - "none":    return raw (unbounded)
    LLM_JUDGE_NORMALIZE = "none"
    LLM_JUDGE_NORM_K = 1.0

    # Judge scoring uses the (post-processed) response by default.
    # If True, uses response_text_raw instead.
    LLM_JUDGE_USE_RAW_RESPONSE = True

    # Caching to reduce repeated judge calls
    LLM_JUDGE_CACHE_PATH = "judge_cache_skywork.jsonl"  # set None to disable file cache
    LLM_JUDGE_CACHE_IN_MEMORY = False

    # Ground-truth matching for quality (optional)
    # =================================================================
    # If True, compute quality_score by comparing the model
    # response_text against the dataset ground-truth output.
    #
    # NOTE: This metric is computed AFTER generation completes, so you
    # typically should NOT include it in the RL state.
    USE_EM_EXACT_MATCH = True

    # If True, include per-server quality scores in the RL state vector.
    # Recommended False when USE_EM_EXACT_MATCH=True.
    INCLUDE_QUALITY_IN_STATE = False

    # Softer matching metric (less strict than EM):
    #   "f1"       : token-level F1 (default, continuous 0..1)
    #   "ratio"    : character-level similarity ratio (0..1)
    #   "contains" : 1 if one contains the other (after normalization)
    #   "em"       : strict exact match (0/1)
    EM_METRIC = "em"

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
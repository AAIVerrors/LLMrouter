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
    
    # Model settings
    MODEL_NAMES = [
        # 'meta-llama/Llama-2-13b-chat-hf',
        # 'meta-llama/Meta-Llama-3-8B-Instruct',
        'allenai/Llama-3.1-Tulu-3-8B',
        'meta-llama/Llama-3.1-8B-Instruct',
        # 'mistralai/Ministral-8B-Instruct-2410',
        # 'mistralai/Mistral-7B-Instruct-v0.2',
        # # "lmsys/fastchat-t5-3b-v1.0",
        "google/gemma-1.1-2b-it",
        # "google/gemma-2-2b-it",
        # "google/gemma-2b-it",
        # "ibm-granite/granite-3.0-2b-instruct",
        # "ibm-granite/granite-3.1-2b-instruct",
        # "meta-llama/Llama-3.2-1B-Instruct",
        # "ibm-granite/granite-3.0-2b-instruct",
        # "ibm-granite/granite-3.1-2b-instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        # "microsoft/Phi-3-mini-128k-instruct",
        # "microsoft/Phi-3-mini-4k-instruct",
        # 'gpt-3.5-turbo-0125',
        # # 'gpt-3.5-turbo-1106',
        # 'gpt-4o-2024-08-06',
        'gpt-4o-mini-2024-07-18',
        # 'o3-mini',
        # 'o1-mini',
        # 'gemini-2.0-flash-001',
        # 'gemini-2.0-flash-exp',
        # 'gemini-1.5-flash-001',
        # 'gemini-1.5-flash-002',
        # 'gemini-1.5-flash-8b-001',
        'claude-3-5-haiku-20241022',
        'claude-3-haiku-20240307',
        # 'claude-3-7-sonnet-20250219',
        # 'claude-3-5-sonnet-20240620',
        "ministral-8b-2410",
        "mistral-7b-instruct-v0.2", # open-mistral-7b
        "mistral-medium",  # mistral-medium-2508
        # "mistral-small-24b-instruct-2501", # mistral-small-2501
        # "mixtral-8x22b-instruct-v0.1", # open-mixtral-8x22b
        "mixtral-8x7b-instruct-v0.1" # open-mixtral-8x7b
    ]

    SERVICE_RATE = [
        28,
        28,
        137,
        163,
        92,
        40,
        39,
        105,
        116,
        136,
        52,
        102,  
    ]
    
    PRICE = [
        # (0.00000025, 0.00000025),
        # (0.00000025, 0.00000025),
        # (0.0000001, 0.0000001),
        # (0.0000001, 0.0000001),
        (0.0000001, 0.0000001),
        (0.0000001, 0.0000001),
        (0.0000001, 0.0000001),
        (0.0000001, 0.0000001),
        (0.0000001, 0.0000001),
        (0.00000015, 0.0000006), # gpt-4o-mini-2024-07-18
        # (0.00000015, 0.0000006), # genmi-2.0-flash-exp
        # (0.000000075, 0.0000003),  # gemini-1.5-flash-001
        # (0.000000075, 0.0000003),  # gemini-1.5-flash-002
        # (0.0000000375, 0.00000015),  # gemini-1.5-flash-8b-001
        (0.0000008, 0.000004), # claude-3-5-haiku-20241022
        (0.00000025, 0.00000125), # claude-3-haiku-20240307
        (0.0000001, 0.0000001), # ministral-8b-2410
        (0.00000025, 0.00000025), # mistral-7b-instruct
        (0.0000004,0.000002), # mistral-medium
        # (0.0000001, 0.0000003), # mistral-small-24b-instruct-2501
        (0.0000007, 0.0000007) # mixtral-8x7b-instruct-v0.1
    ]
    
    # Server capabilities (max concurrent requests)
    SERVER_CAPACITIES = [32, 32, 30, 32, 32, 30, 32, 32, 30, 32, 30, 32]  # Capacity for each model
    
    # Dataset settings
    DATASET_NAME = "tatsu-lab/alpaca"
    MAX_SAMPLES = 10000  # Limit for testing
    
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
    LAMBDA = 5  # Capacity penalty weight (increased to strongly discourage invalid actions)
    
    # PPO hyperparameters - tuned for the routing problem
    LEARNING_RATE =1e-4  # Reduced for more stable learning
    GAMMA = 0.99          # Slightly reduced discount factor
    GAE_LAMBDA = 0.95      # Reduced for less variance in advantage estimation
    CLIP_EPSILON = 0.2    # Slightly reduced for more conservative updates
    POLICY_COEF = 1       # Policy loss weight
    VALUE_COEF = 0.5      # Reduced value function weight
    ENTROPY_COEF = 0.05   # Increased entropy for more exploration
    KL_COEF = 0.02
    MAX_GRAD_NORM = 0.5
    PPO_EPOCHS = 4        # Increased for more thorough updates
    BATCH_SIZE = 1      # Increased batch size
    
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
    
    # Text generation settings (for actual LLM inference)
    MAX_LENGTH = 100
    TEMPERATURE = 0.7
    
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
    
    # Load factor settings
    MAX_LOAD_FACTOR = 1.5     # Maximum latency increase due to server load
    COMPLETION_PROBABILITY = 0.3  # Probability of request completion per step
    
    # Final evaluation settings
    EVAL_EPISODES = 5         # Number of episodes for evaluation
    FINAL_EVAL_EPISODES = 10  # Number of episodes for final evaluation
    
    # Poisson prompt generation settings
    POISSON_ARRIVAL_RATE = 5  # Average arrival rate of prompts per second
    MAX_PROMPT_QUEUE_SIZE = 10000  # Maximum size of the prompt queue
    EPISODE_TIME_INTERVAL = 60  # Time interval for each episode in seconds
    
    # Queue score settings
    QUEUE_SCORE_FACTOR = 0.2  # Factor to adjust queue score impact
    QUEUE_EPSILON = 0.0001  # Epsilon for queue score stability
    MERGE_ALPHA = 0 # Alpha for merging action probabilities (0.5 for equal weighting)

    ROUND_ROBIN = False
    
    USE_MERGE_TO_TRAIN = False  # Use merge action for training 
    
    ADAPTIVE_EPSILON = False  # Use adaptive epsilon for exploration
    
    MIX_QUEUE_SCORE = False
    
    ENTROPY_BASED_EXPLORATION = False  # Use entropy-based exploration
    
    USE_AVG = False 
    
    RANDOM_SELECT = False
    
    T = -2

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
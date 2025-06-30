import torch
import time

class Config:
    # Model settings
    MODEL_NAMES = [
        "openai-community/gpt2",
        "Qwen/Qwen2.5-0.5B"
    ]
    
    # Server capabilities (max concurrent requests)
    SERVER_CAPACITIES = [10, 20]  # Capacity for each model
    
    # Dataset settings
    DATASET_NAME = "tatsu-lab/alpaca"
    MAX_SAMPLES = 10000  # Limit for testing
    
    # Start time for frequent requests
    START_TIME = time.time()  # Start time for the first request
    
    # Training settings
    EPISODE_LENGTH = 100  # Number of prompts per episode (increased for better learning)
    MAX_EPISODES = 100   # Increased for more training
    
    # Reward function weights - adjusted for better balance
    ALPHA = 2.0   # Quality weight (increased importance)
    BETA = 1.0    # Latency weight
    LAMBDA = 5.0  # Capacity penalty weight (increased to strongly discourage invalid actions)
    
    # PPO hyperparameters - tuned for the routing problem
    LEARNING_RATE = 1e-4  # Reduced for more stable learning
    GAMMA = 0.95          # Slightly reduced discount factor
    GAE_LAMBDA = 0.9      # Reduced for less variance in advantage estimation
    CLIP_EPSILON = 0.15   # Slightly reduced for more conservative updates
    VALUE_COEF = 0.25     # Reduced value function weight
    ENTROPY_COEF = 0.02   # Increased entropy for more exploration
    MAX_GRAD_NORM = 0.5
    PPO_EPOCHS = 6        # Increased for more thorough updates
    BATCH_SIZE = 128      # Increased batch size
    
    # Neural network settings
    HIDDEN_DIM = 512      # Increased capacity
    INPUT_DIM = 512       # Embedding dimension for prompts
    
    # Environment settings
    TIME_STEP = 0.1       # Time increment per environment step
    COMPLETION_CHECK_STEPS = 20  # Steps to wait for request completion at episode end
    
    # Device settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        0: 1200,  # GPT-2 base ELO
        1: 1100,  # Qwen base ELO
    }
    
    # Latency simulation settings
    MODEL_LATENCY_RANGES = {
        "gpt2": (0.5, 1.0),      # GPT-2 latency range in seconds
        "qwen": (0.7, 1.2),      # Qwen latency range in seconds
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
    MAX_PROMPT_QUEUE_SIZE = 10000  # Maximum size of the
    EPISODE_TIME_INTERVAL = 60  # Time interval for each episode in seconds
    
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
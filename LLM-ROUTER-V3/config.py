import torch

class Config:
    # Model settings
    MODEL_NAMES = [
        "openai-community/gpt2",
        "Qwen/Qwen2.5-0.5B"
    ]
    
    # Server capabilities (max concurrent requests)
    SERVER_CAPACITIES = [8, 6]  # Capacity for each model
    
    # Dataset settings
    DATASET_NAME = "tatsu-lab/alpaca"
    MAX_SAMPLES = 10000  # Limit for testing
    
    # Training settings
    EPISODE_LENGTH = 50  # Number of prompts per episode (increased for better learning)
    MAX_EPISODES = 200   # Increased for more training
    
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
    
    # Evaluation settings
    EVAL_EPISODES = 5         # Number of episodes for evaluation
    FINAL_EVAL_EPISODES = 10  # Number of episodes for final evaluation
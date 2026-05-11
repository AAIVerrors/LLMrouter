#!/usr/bin/env python3
"""
Main training script for Enhanced LLM Router.
Clean implementation without emojis, separated into multiple files for maintainability.
"""

import random
import numpy as np
import torch
import wandb
from datetime import datetime

from config import Config
from trainer import EnhancedLLMRouterTrainer

import torch.multiprocessing as mp
from dotenv import load_dotenv
import os


def setup_environment():
    """Setup training environment with reproducible seeds"""
    print("Setting up training environment...")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU")

    print(f"Random seeds set to 42 for reproducibility")

def print_configuration():
    """Print training configuration"""
    print("\nTraining Configuration:")
    print("-" * 40)
    print(f"Models: {Config.MODEL_NAMES}")
    print(f"Server Capacities: {Config.SERVER_CAPACITIES}")
    print(f"Episode Length: {Config.EPISODE_LENGTH}")
    print(f"Max Episodes: {Config.MAX_EPISODES}")
    print(f"Reward Weights: α={Config.ALPHA}, β={Config.BETA}, λ={Config.REWARD_GAMMA}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"PPO Epochs: {Config.PPO_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print("-" * 40)

def main():
    """Main function"""
    print("Enhanced LLM Router Training")
    print("=" * 50)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    load_dotenv()
    
    # Setup environment
    setup_environment()
    print_configuration()
    
    # Create and run trainer
    try:
        mp.set_start_method('spawn', force=True)
        
        print("Enhanced LLM Router Training")
        print("=" * 50)
        
        trainer = EnhancedLLMRouterTrainer()
        trainer.train()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C)")
        try:
            wandb.finish()
        except:
            pass
        print("Cleanup completed")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        try:
            wandb.finish()
        except:
            pass
        
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
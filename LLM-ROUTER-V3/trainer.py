import os
import numpy as np
import wandb
import torch
from datetime import datetime

from config import Config
from environment import EnhancedRouterEnvironment
from router_network import PPOAgent
from data_loader import EpisodeBuffer  # Only import EpisodeBuffer
from plotter import TrainingPlotter
from logger import MetricsLogger
from PoissonPromptGenerator import PoissonPromptGenerator

class EnhancedLLMRouterTrainer:
    def __init__(self):
        # Initialize components
        self.env = EnhancedRouterEnvironment(enable_monitoring=Config.ENABLE_QUEUE_MONITORING)
        self.buffer = EpisodeBuffer()
        
        # Initialize episode tracking
        self.current_episode = 0  
        
        # Initialize PPO agent
        state_dim = len(Config.SERVER_CAPACITIES)  # load per server
        action_dim = len(Config.SERVER_CAPACITIES)  # Number of servers
        self.agent = PPOAgent(state_dim, action_dim)
        
        # Initialize wandb based on config
        self.wandb_available = False
        if Config.ENABLE_WANDB_LOGGING:
            self.wandb_available = self.init_wandb()
        
        # Set wandb availability for queue monitor
        if hasattr(self.env, 'queue_monitor') and self.env.queue_monitor:
            self.env.queue_monitor.wandb_available = self.wandb_available
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_stats = []
        self.training_metrics = []
        self.last_service_rate = [1] * len(Config.SERVER_CAPACITIES)  # Default service rate
        
        self.training_in_progress = False

        # Print configuration summary
        if Config.ENABLE_CONSOLE_LOGGING:
            self.print_config_summary()
        
    def print_config_summary(self):
        """Print current configuration summary"""
        if not Config.CONSOLE_CONFIG.get('episode_progress', True):
            return
            
        print("\n Training Configuration Summary:")
        print("=" * 50)
        summary = Config.get_config_summary()
        
        print(f" Core Settings:")
        print(f"   Wandb Logging: {'✅' if summary['wandb_logging'] else '❌'}")
        print(f"   Console Logging: {'✅' if summary['console_logging'] else '❌'}")
        print(f"   Queue Monitoring: {'✅' if summary['queue_monitoring'] else '❌'}")
        print(f"   Visualizations: {'✅' if summary['visualizations'] else '❌'}")
        print(f"   File Exports: {'✅' if summary['file_exports'] else '❌'}")
        
        print(f"\n Active Features:")
        print(f"   Visualizations: {summary['active_visualizations']}/{len(Config.VISUALIZATION_CONFIG)}")
        print(f"   Logging Types: {summary['active_logging']}/{len(Config.LOGGING_CONFIG)}")
        print(f"   Console Types: {summary['active_console']}/{len(Config.CONSOLE_CONFIG)}")
        
        print("=" * 50)
    
    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        if not Config.ENABLE_WANDB_LOGGING:
            print("Wandb logging disabled by config")
            return False
            
        try:
            # Convert Config class to dictionary for wandb
            config_dict = {}
            for attr in dir(Config):
                if not attr.startswith('_'):
                    value = getattr(Config, attr)
                    if not callable(value) and not isinstance(value, dict):
                        config_dict[attr] = value
            
            wandb.init(
                project=Config.WANDB_PROJECT,
                entity=Config.WANDB_ENTITY,
                config=config_dict,
                name=f"enhanced_llm_router_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                reinit=True
            )
            
            if Config.WANDB_CONFIG.get('watch_model', True):
                wandb.watch(self.agent.network, log='all', log_freq=100)
            
            if Config.CONSOLE_CONFIG.get('episode_progress', True):
                print(" Wandb initialized successfully!")
                print(f" Project: {Config.WANDB_PROJECT}")
            return True
            
        except Exception as e:
            if Config.CONSOLE_CONFIG.get('error_messages', True):
                print(f" Warning: Could not initialize wandb: {e}")
                print("Continuing without wandb logging...")
            return False
        
    def get_episode_data(self):
        record = self.run_episode()  # record is now a list of Request objects
        
        

        episode_info = {
            'rewards': [],
            'quality_scores': [],
            'latencies': [],
            'capacity_penalties': [],
            'invalid_actions': 0,
            'valid_actions': 0,
            'service_rate': [],
        }
        
        # Count processed prompts per server
        num_servers = len(Config.SERVER_CAPACITIES)
        num_processed = [0] * num_servers
        for req in record:
            server_id = req['server_id']
            if server_id is not None and req['status'] == 'completed' and req['episode'] == self.current_episode:
                num_processed[server_id] += 1

        # Compute per-server service rate
        duration = Config.EPISODE_TIME_INTERVAL
        service_rate = [n / duration for n in num_processed]
        
        print(service_rate)

        # Store service_rate for use in agent
        self.last_service_rate = service_rate

        episode_reward = 0

        for req in record:
            episode_info['rewards'].append(req['reward'])
            episode_info['quality_scores'].append(req['quality_score'])
            episode_info['latencies'].append(req['processing_latency'])
            # If you have capacity_penalty, add logic here

            if req['status'] == 'completed' and req['episode'] == self.current_episode:
                episode_info['valid_actions'] += 1
            else:
                episode_info['invalid_actions'] += 1

            episode_reward += req['reward']

        episode_info['total_reward'] = episode_reward
        episode_info['episode_length'] = len(record)
        episode_info['service_rate'] = service_rate

        return episode_info

    def run_episode(self) -> dict:
        """Run a single episode using Poisson prompt generator"""
        
        state = self.env.reset()
        
        import time
        start = time.time()
        
        while True:
            if time.time() - start > Config.EPISODE_TIME_INTERVAL:
                print(f"Episode {self.current_episode} timed out after {Config.EPISODE_TIME_INTERVAL} seconds")
                break
            
            # Get prompt from Poisson generator (environment handles this)
            action_mask = self.env.get_action_mask()
            
            prompt = self.env.get_next_prompt()
            if not prompt:
                # print("No more prompts available")
                time.sleep(0.1)
                continue
            prompt = prompt['prompt']  # Extract the actual prompt text
            print(prompt)
            # Use a dummy prompt for action selection (actual prompt comes from environment)
            action, log_prob, value = self.agent.get_action(state, prompt, action_mask, service_rate=self.last_service_rate)
            
            next_state, done = self.env.step(action, prompt)
            
            state = next_state
            
            if done:
                break
            
            # Add step to buffer (reward will be filled in later)
            self.buffer.add_step(
                state=state,
                prompt=prompt,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=0,  # Placeholder, will be updated after episode
                action_mask=action_mask
            )
            
        # --- Pause and clean servers before training ---
        self.env.pause_all_servers()
        time.sleep(1)
        
        episode_record = self.env.get_episode_data()
        
        time.sleep(2)
        self.env.clean_all_queues()
        time.sleep(2)
        
        # Update buffer rewards with actual episode rewards
        for i, req in enumerate(episode_record):
            if i < len(self.buffer.current_episode):
                self.buffer.current_episode[i]['reward'] = req['reward']
        
        return episode_record
    
    def train_step(self):
        """Perform one training step with collected trajectories"""
        trajectories = self.buffer.get_current_episode()
        
        if len(trajectories) == 0:
            return {}
        
        training_metrics = self.agent.update(trajectories)
        self.buffer.finish_episode()
        return training_metrics
    
    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/enhanced_router_model_ep_{episode}.pt'
        self.agent.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("Starting Enhanced LLM Router Training with PPO")
        print("=" * 60)
        print(f"Models: {Config.MODEL_NAMES}")
        print(f"Server Capacities: {Config.SERVER_CAPACITIES}")
        print(f"Episode Length: {Config.EPISODE_LENGTH}")
        print(f"Max Episodes: {Config.MAX_EPISODES}")
        print(f"Wandb Available: {self.wandb_available}")
        print("=" * 60)
        
        for episode in range(Config.MAX_EPISODES):
            self.current_episode = episode  # Update current episode
            self.env.set_episode(episode)  # Update environment episode tracking
            
            print(f"\nRunning Episode {episode}...")
 
            # Run episode
            episode_info = self.get_episode_data()
            self.episode_rewards.append(episode_info['rewards'])
            self.episode_stats.append(episode_info)
            
            # print(episode_info)
            
            # Train every episode
            training_metrics = None
            print(f"Training agent (episode {episode})...")
            training_metrics = self.train_step()
            if training_metrics:
                print(f"   Policy Loss: {training_metrics['policy_loss']:.6f}")
                print(f"   Value Loss: {training_metrics['value_loss']:.6f}")
            
            if self.wandb_available:
                wandb.log({
                    "episode": episode,
                    "total_reward": episode_info['total_reward'],
                    "mean_reward": np.mean(episode_info['rewards']),
                    "std_reward": np.std(episode_info['rewards']),
                    "policy_loss": training_metrics['policy_loss'] if training_metrics else None,
                    "value_loss": training_metrics['value_loss'] if training_metrics else None,
                    "entropy_loss": training_metrics['entropy_loss'] if training_metrics else None,
                    "throughput_per_episode/requests_completed": episode_info['valid_actions'],
                }, step=episode)
            
            if self.wandb_available and hasattr(self.env, 'queue_monitor'):
                self.env.queue_monitor.log_throughput_to_wandb(episode)
            
            # Plot progress and queue monitoring
            plot_interval = 25 if episode < 100 else 50
            if episode > 0 and episode % plot_interval == 0:
                print(f"Creating training progress plots...")
                plotter = TrainingPlotter(self.wandb_available)
                plotter.plot_training_progress(self.episode_rewards, self.episode_stats)
                
                # Create queue monitoring plots
                if hasattr(self.env, 'queue_monitor'):
                    self.env.queue_monitor.create_queue_visualization()
                    print("Queue state summary:")
                    self.env.queue_monitor.print_queue_summary()
            
            # Save periodic checkpoints
            if episode > 0 and episode % Config.SAVE_INTERVAL == 0:
                self.save_checkpoint(episode)
                
            # --- Resume servers after training ---
            self.env.resume_all_servers()
        
        print(f"\nTraining completed!")
        
        try:
            self.env.pause_all_servers()
            self.env.clean_all_queues()
            if self.wandb_available:
                wandb.finish()
            print("Training cleanup completed successfully")
        except Exception as e:
            print(f"Cleanup failed: {e}")
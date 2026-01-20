import os
import numpy as np
import wandb
import torch
import json
from datetime import datetime

from config import Config
from environment import EnhancedRouterEnvironment
from router_network import PPOAgent
from data_loader import EpisodeBuffer  # Only import EpisodeBuffer
from plotter import TrainingPlotter
from logger import MetricsLogger
from PoissonPromptGenerator import PoissonPromptGenerator

class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert arrays to lists
            return super(NumpyEncoder, self).default(obj)

class EnhancedLLMRouterTrainer:
    def __init__(self):
        self.trajectory_dir = f"trajectories/run-{Config.T}-{Config.EPISODE_TIME_INTERVAL}-{Config.MAX_EPISODES}-{Config.USE_AVG}-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{Config.INTERVAL_LENGTH}-{Config.POISSON_ARRIVAL_RATE}-{Config.EPISODE_TIME_INTERVAL}"
        os.makedirs(self.trajectory_dir, exist_ok=True)
        
        # Initialize components
        self.env = EnhancedRouterEnvironment(enable_monitoring=Config.ENABLE_QUEUE_MONITORING)
        self.buffer = EpisodeBuffer()
        
        # Initialize episode tracking
        self.current_episode = 0  
        # Initialize PPO agent
        M = len(Config.SERVER_CAPACITIES)
        include_quality_state = bool(getattr(Config, 'INCLUDE_QUALITY_IN_STATE', True)) and not bool(getattr(Config, 'USE_EM_EXACT_MATCH', False))
        state_dim = M * (4 if include_quality_state else 3)
        action_dim = M  # Number of servers
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
            'prices': [],
        }
        
        # Count processed prompts per server
        num_servers = len(Config.SERVER_CAPACITIES)
        num_processed = [0] * num_servers
        counter = 0
        for index,req in enumerate(record):
            server_id = req['server_id']
            if server_id is not None and req['status'] == 'completed' and req['episode'] == self.current_episode:
                num_processed[server_id] += 1 
                counter += 1
            # if index == len(record) - 1:
            #     req['reward'] += counter # Add bonus for last request in episode


        # Compute per-server service rate
        duration = Config.EPISODE_TIME_INTERVAL
        service_rate = [n / duration + self.last_service_rate[index] for index,n in enumerate(num_processed)]
        
        print(service_rate)

        # Store service_rate for use in agent
        self.last_service_rate = service_rate

        episode_reward = 0

        for req in record:
            episode_info['rewards'].append(req['reward'])
            if req['status'] == 'completed' and req['episode'] == self.current_episode:
                episode_info['quality_scores'].append(req['quality_score'])
                episode_info['latencies'].append(req['processing_latency'])
                episode_info['prices'].append(req['price'])
                
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
        
        price = [((a[0] + a[1]) / 2) for a in Config.PRICE]  # avg price per server

        M = len(Config.SERVER_CAPACITIES)
        include_quality_state = bool(getattr(Config, 'INCLUDE_QUALITY_IN_STATE', True)) and not bool(getattr(Config, 'USE_EM_EXACT_MATCH', False))

        def build_state(loads):
            # loads: per-server queue lengths (from env)
            util = [float(loads[i]) / float(c) for i, c in enumerate(Config.SERVER_CAPACITIES)]
            s = util + list(Config.SERVICE_RATE)
            if include_quality_state:
                s += [1.0] * M
            s += price
            return s

        state = build_state(self.env.reset())
        # state = [self.env.reset()[index]/c for index,c in enumerate(Config.SERVER_CAPACITIES)] + [1] * len(Config.SERVER_CAPACITIES) + price

        # state = [self.env.reset()[index]/c for index,c in enumerate(Config.SERVER_CAPACITIES)] + self.last_service_rate + price

        import time
        start = time.time()
        robin_counter = 0  # Initialize round-robin counter
        
        current_time_slot = 0
        time_slot_buffer = []
        current = -1
        
        while True:
            routing_start = time.time()
            current_time_slot = (routing_start - start) // Config.INTERVAL_LENGTH
            if current_time_slot >= Config.EPISODE_TIME_INTERVAL :
                print(f"Episode {self.current_episode} timed out after {Config.EPISODE_TIME_INTERVAL} seconds")
                break

            # Get prompt from Poisson generator (environment handles this)
            action_mask = self.env.get_action_mask()

            prompt_entry = self.env.get_next_prompt()
            if not prompt_entry:
                time.sleep(0.1)
                continue

            if isinstance(prompt_entry, dict):
                prompt = prompt_entry.get('prompt', '')
                ground_truth = (prompt_entry.get('output') or prompt_entry.get('answer') or prompt_entry.get('target'))
            else:
                prompt = str(prompt_entry)
                ground_truth = None

            action, log_prob, value, next_counter = self.agent.get_action(state, prompt, action_mask, service_rate=self.last_service_rate, round_robin_counter=robin_counter)
            print("log_prob:", log_prob)
            robin_counter = next_counter
            
            next_state, done = self.env.step(action, prompt, ground_truth)

            if Config.NAIVE_PPO:
                state = build_state(next_state)
            else:
            
                if current != current_time_slot:
                    state = build_state(next_state)
                    # state = [next_state[index]/c for index,c in enumerate(Config.SERVER_CAPACITIES)] + [1]* len(Config.SERVER_CAPACITIES) + price
                    # state = [next_state[index]/c for index,c in enumerate(Config.SERVER_CAPACITIES)] + self.last_service_rate +  price
                    current = current_time_slot
             
            
            if done:
                break
            
            
            # Add step to buffer (reward will be filled in later)
            self.buffer.add_step(
                time_slot = int(current_time_slot),
                route_time= current_time_slot,
                state=state,
                prompt=prompt,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=0,  # Placeholder, will be updated after episode
                action_mask=action_mask,
                service_rate=Config.SERVICE_RATE
            )
        
        # Wait for all prompts to be processed
        self.env.pause_prompt_generator()
        time.sleep(2)
        
        # --- Pause and clean servers before training ---
        while self.env.check_get_episode_completed() == False:
            print("Waiting for all prompts to be processed...")
            time.sleep(1)
        
        episode_record = self.env.get_episode_data()
        
        time.sleep(2)
        
        self.env.pause_all_servers()
        time.sleep(2)
        
        self.env.clean_all_queues()
        time.sleep(2)
        
        # Update buffer rewards with actual episode rewards
        for i, req in enumerate(episode_record):
            if i < len(self.buffer.current_episode):
                self.buffer.current_episode[i]['reward'] = req['reward']
                
        # normalize latency, price and then recompute rewards
        # if len(episode_record) > 0:
        #     latencies = np.array([req['processing_latency'] for req in episode_record])
        #     prices = np.array([req['price'] for req in episode_record])
        #     if len(latencies) > 1:
        #         lat_mean, lat_std = latencies.mean(), latencies.std()
        #         prices_mean, prices_std = prices.mean(), prices.std()
        #         for i, req in enumerate(episode_record):
        #             norm_latency = (req['processing_latency'] - lat_mean) / (lat_std + 1e-8) if lat_std > 0 else 0
        #             req['processing_latency'] = norm_latency
        #             norm_price = (req['price'] - prices_mean) / (prices_std + 1e-8) if prices_std > 0 else 0
        #             req['price'] = norm_price
        #             # Recompute reward with normalized latency and price
        #             req['reward'] = Config.ALPHA * req['quality_score'] - Config.BETA * norm_latency - Config.REWARD_GAMMA * norm_price
        #             # Update buffer as well
        #             if i < len(self.buffer.current_episode):
        #                 self.buffer.current_episode[i]['reward'] = req['reward']
        
        # Update greedy utility history (EMA latency/cost, optional queue-conditioned stats).
        # Safe no-op if the agent does not implement update_server_stats.
        try:
            self.agent.update_server_stats(episode_record)
        except Exception as e:
            print(f"[WARN] update_server_stats failed: {e}")

        return episode_record

    def tensor_to_python(self, obj):
        if isinstance(obj, torch.Tensor):
            if obj.dim() == 0:
                return obj.item()
            else:
                return obj.cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.tensor_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.tensor_to_python(v) for v in obj]
        else:
            return obj

    
    
    def save_trajectories_json(self, trajectories, episode, trajectory_dir):
        serializable_trajectories = [self.tensor_to_python(t) for t in trajectories]
        filename = os.path.join(trajectory_dir, f"episode_{episode:04d}.json")
        with open(filename, 'w') as f:
            json.dump(serializable_trajectories, f, indent=2, cls=NumpyEncoder)
    
    def train_step(self, episode=None, trajectory_dir=None):
        """Perform one training step with collected trajectories"""
        trajectories = self.buffer.get_current_episode()
        
        if len(trajectories) == 0:
            return {}

        if trajectory_dir is not None:
            self.save_trajectories_json(trajectories, episode, trajectory_dir)

        training_metrics = self.agent.update_new(trajectories)
        self.buffer.finish_episode()
        return training_metrics
    
    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/enhanced_router_model_ep_{episode}.pt'
        self.agent.save(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def safe_percentile(self, x, q):
        """Returns percentile q of list x, or None if empty/invalid."""
        if x is None:
            return None
        arr = np.array([v for v in x if v is not None and np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return None
        return float(np.percentile(arr, q))

    def jains_fairness(self, x):
        arr = np.asarray(list(x.values()), dtype=float)
        n = arr.size
        if n == 0:
            return 0.0
    
        s1 = arr.sum()
        s2 = np.square(arr).sum()
        if s2 == 0.0:
            return 1.0  # all zeros => perfectly equal
    
        return float((s1 * s1) / (n * s2))

    
    # def jains_fairness(self, x):
    #     """
    #     Jain's fairness index: (sum x)^2 / (n * sum x^2)
    #     x should be non-negative (e.g., per-server completed counts).
    #     Returns None if invalid/empty.
    #     """
    #     if x is None:
    #         return None
    #     arr = np.array(x, dtype=float)
    #     if arr.size == 0:
    #         return None
    #     arr = np.clip(arr, 0.0, None)
    #     s1 = arr.sum()
    #     s2 = np.square(arr).sum()
    #     if s2 <= 1e-12:
    #         return None
    #     return float((s1 * s1) / (arr.size * s2))

    
    def train(self):
        """Main training loop"""
        print("Starting Enhanced LLM Router Training with PPO")
        print("=" * 60)
        print(f"Models: {Config.MODEL_NAMES}")
        print(f"Server Capacities: {Config.SERVER_CAPACITIES}")
        print(f"Episode Length: {Config.EPISODE_TIME_INTERVAL}")
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
            
            print(episode_info)
            
            # Train every episode
            # if not Config.ROUND_ROBIN:
            if True:
                training_metrics = None
                print(f"Training agent (episode {episode})...")
                training_metrics = self.train_step(episode, self.trajectory_dir)
                
                if training_metrics:
                    print(f"   Policy Loss: {training_metrics['policy_loss']:.6f}")
                    print(f"   Value Loss: {training_metrics['value_loss']:.6f}")
                
                if self.wandb_available:
                    
                    lat_list = episode_info['latencies']
                    p50 = self.safe_percentile(lat_list, 50)
                    p90 = self.safe_percentile(lat_list, 90)
                    p99 = self.safe_percentile(lat_list, 99)
                    
                    # Fairness: prefer COMPLETED per-server counts if you have them
                    # (best), otherwise fall back to attempted actions.
                    per_server_counts = None
                    
                    fairness = self.jains_fairness(training_metrics['route distribution'])
                    
                    # Fix price mean
                    price_mean = float(np.mean(episode_info['prices'])) if episode_info.get('prices') else None

                    wandb.log({
                        "episode": episode,
                        "total_reward": float(np.sum(episode_info['rewards'])) if episode_info.get('rewards') else None,
                        "mean_reward": float(np.mean(episode_info['rewards'])) if episode_info.get('rewards') else None,
                        "std_reward": float(np.std(episode_info['rewards'])) if episode_info.get('rewards') else None,
                    
                        "policy_loss": training_metrics['policy_loss'] if training_metrics else None,
                        "value_loss": training_metrics['value_loss'] if training_metrics else None,
                        "entropy_loss": training_metrics['entropy_loss'] if training_metrics else None,
                    
                        "quality_scores": float(np.mean(episode_info['quality_scores'])) if episode_info.get('quality_scores') else None,
                        "latencies": float(np.mean(episode_info['latencies'])) if episode_info.get('latencies') else None,
                    
                        "p50": p50,
                        "p90": p90,
                        "p99": p99,
                    
                        "Jain_fairness_index": fairness,

                        'valid_total_request_ratio': episode_info.get('valid_actions', None)/episode_info.get('episode_length', None),
                    
                        "price": price_mean,
                    
                        "throughput_per_episode/requests_completed": episode_info.get('valid_actions', None),
                        "request_total_numbers": episode_info.get('episode_length', None),
                    
                        "returns": training_metrics['rewards_returns'] if training_metrics else None,
                        "term_returns": training_metrics['term_rewards_returns'] if training_metrics else None,
                        "min_rewards": training_metrics['min_rewards'] if training_metrics else None,
                        "cumulated_avg_rewards_return": training_metrics['cumulated_avg_rewards'] if training_metrics else None,
                        "entropy of route distribution": training_metrics['entropy of route distribution'] if training_metrics else None,
                        "approx_kl": training_metrics['approx_kl'] if training_metrics else None,
                    }, step=episode)


                # Add server usage percentage if available
                if training_metrics and 'server_usage_percentage' in training_metrics:
                    for server_id, usage in training_metrics['server_usage_percentage'].items():
                        wandb.log({f"server_{server_id}_usage": usage}, step=episode)

                # Add server scores if available
                # if training_metrics and 'each_server_score' in training_metrics:
                #     for server_id, score in training_metrics['each_server_score'].items():
                #         wandb.log({f"server_{server_id}_score": score}, step=episode)
                        
                # # Add min score server 
                # if training_metrics and 'min_score_server' in training_metrics:
                #     wandb.log({"min_score_server": training_metrics['min_score_server']}, step=episode)
                    
                # # Add mean reward per server if available
                # if training_metrics and 'mean_reward_per_server' in training_metrics:
                #     for server_id, mean_reward in training_metrics['mean_reward_per_server'].items():
                #         wandb.log({f"server_{server_id}_reward": mean_reward}, step=episode)
                        
                # # Add min mean reward server
                # if training_metrics and 'min_mean_reward_server' in training_metrics:
                #     wandb.log({"min_mean_reward_server": training_metrics['min_mean_reward_server']}, step=episode)
                    
                # # Add return of each server
                # if training_metrics and 'each_server_returns' in training_metrics:
                #     for server_id, ret in training_metrics['each_server_returns'].items():
                #         wandb.log({f"server_{server_id}_trajectory_return": ret}, step=episode)
                
                # # Add min return server
                # if training_metrics and 'min_return_server' in training_metrics:
                #     wandb.log({"min_return_server": training_metrics['min_return_server']}, step=episode)

                if self.wandb_available and hasattr(self.env, 'queue_monitor'):
                    self.env.queue_monitor.log_throughput_to_wandb(episode)
            else:   
                # # min rewards of each time slot
                def cumulated_return(self, rewards):
                    """Compute cumulated returns"""
                    returns = torch.zeros_like(rewards)
                    returns[-1] = rewards[-1]
                    for t in reversed(range(len(rewards) - 1)):
                        returns[t] = rewards[t] + Config.GAMMA * returns[t + 1]
                    return returns
                
                # term_rewards = []
                
                # min_rewards = []
                # term_returns = []
                # for t in range(Config.EPISODE_TIME_INTERVAL):
                #     rewards_t = [step['reward'] for step in self.buffer.current_episode if step['time_slot'] == t]
                #     if rewards_t:
                #         term_rewards.append(np.mean(rewards_t))
                #     else:
                #         term_rewards.append(0)
                # min_rewards = []
                # term_returns = []
                # for t in range(Config.EPISODE_TIME_INTERVAL):
                #     rewards_t = [step['reward'] for step in self.buffer.current_episode if step['time_slot'] == t]
                #     if rewards_t:
                #         min_rewards.append(min(rewards_t))
                #     else:
                #         min_rewards.append(0)
                
                min_rewards_per_time_slot = []
                for t in range(Config.EPISODE_TIME_INTERVAL):
                    rewards_t = [step['reward'] for step in self.buffer.current_episode if step['time_slot'] == t]
                    if rewards_t:
                        min_rewards_per_time_slot.append(min(rewards_t))
                    else:
                        min_rewards_per_time_slot.append(0)
                min_rewards = np.array(min_rewards_per_time_slot)
                
                returns = self.cumulated_return(self, torch.tensor([step['reward'] for step in self.buffer.current_episode], dtype=torch.float32))[0]
                
                
                        
                # # return 
                # rewards_returns = []

                # returns = self.cumulated_return(self, torch.tensor([step['reward'] for step in self.buffer.current_episode], dtype=torch.float32))[0]

                # term_returns = []
                # G = 0
                # for r in reversed(self.buffer.current_episode):
                #     if r['done']:
                #         G = 0
                #     G = r['reward'] + Config.GAMMA * G
                #     term_returns.insert(0, G)
                # term_returns = np.array(term_returns)
                # term_returns = (term_returns - term_returns.mean()) / (term_returns.std() + 1e-8)
                
                # cumulated_avg_rewards = []
                # cum_sum = 0
                # for i, r in enumerate(self.buffer.current_episode):
                #     cum_sum += r['reward']
                #     cumulated_avg_rewards.append(cum_sum / (i + 1)) 
                # cumulated_avg_rewards = np.array(cumulated_avg_rewards) 
                # cumulated_avg_rewards = (cumulated_avg_rewards - cumulated_avg_rewards.mean()) / (cumulated_avg_rewards.std() + 1e-8)
                # training_metrics = {
                #     'min_rewards': min_rewards,
                #     'rewards_returns': returns,
                #     'term_rewards_returns': term_returns,
                #     'cumulated_avg_rewards': cumulated_avg_rewards
                # }
                
                if self.wandb_available:
                        wandb.log({
                            "episode": episode,
                            "total_reward": episode_info['total_reward'],
                            "mean_reward": np.mean(episode_info['rewards']),
                            "std_reward": np.std(episode_info['rewards']),
                            "quality_scores": np.mean(episode_info['quality_scores']),
                            "latencies": np.mean(episode_info['latencies']),
                            "price": np.mean([episode_info['prices']]),
                            "throughput_per_episode/requests_completed": episode_info['valid_actions'],
                            "min_rewards": np.mean(min_rewards),
                            "returns": returns,
                            # "returns": training_metrics['rewards_returns'],
                            # "term_returns": training_metrics['term_rewards_returns'],
                            # 'min_rewards': training_metrics['min_rewards'],
                            # 'cumulated_avg_rewards_return': training_metrics['cumulated_avg_rewards'],
                        }, step=episode)
                
                # server usage percentage
                server_usage = {i: 0 for i in range(len(Config.SERVER_CAPACITIES))}
                for step in self.buffer.current_episode:
                    server_usage[step['action']] += 1
                total_actions = sum(server_usage.values())
                if total_actions > 0:
                    server_usage = {k: v / total_actions for k, v in server_usage.items()}
                if self.wandb_available:
                    for server_id, usage in server_usage.items():
                        wandb.log({f"server_{server_id}_usage": usage}, step=episode)
                    
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

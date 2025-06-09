import numpy as np
import wandb
from config import Config

class MetricsLogger:
    def __init__(self, wandb_available=False):
        self.wandb_available = wandb_available
    
    def log_metrics(self, episode, episode_info, episode_rewards, training_metrics=None, eval_info=None):
        """Log metrics to wandb and console"""
        # Console logging
        self._log_to_console(episode, episode_info, episode_rewards, training_metrics, eval_info)
        
        # Wandb logging
        if self.wandb_available:
            self._log_to_wandb(episode, episode_info, episode_rewards, training_metrics, eval_info)
    
    def _log_to_console(self, episode, episode_info, episode_rewards, training_metrics=None, eval_info=None):
        """Log metrics to console"""
        print(f"\n{'='*60}")
        print(f"EPISODE {episode} RESULTS")
        print(f"{'='*60}")
        print(f"Total Reward: {episode_info['total_reward']:.3f}")
        print(f"Completed Requests: {episode_info['completed_requests']}")
        print(f"Valid Actions: {episode_info['valid_actions']}")
        print(f"Invalid Actions: {episode_info['invalid_actions']}")
        print(f"Quality Score (avg): {np.mean(episode_info['quality_scores']):.3f}")
        
        valid_latencies = [l for l in episode_info['latencies'] if l > 0]
        if valid_latencies:
            print(f"Latency (avg): {np.mean(valid_latencies):.3f}s")
        else:
            print(f"Latency: No completed requests yet")
            
        print(f"Action Distribution: {episode_info['action_distribution']}")
        
        # Show recent performance trend
        if len(episode_rewards) >= 10:
            recent_rewards = episode_rewards[-10:]
            recent_avg = np.mean(recent_rewards)
            print(f"Last 10 episodes avg: {recent_avg:.3f}")
        
        if 'env_stats' in episode_info:
            env_stats = episode_info['env_stats']
            print(f"Environment Stats:")
            print(f"   Total Requests Created: {env_stats.get('total_requests_created', 0)}")
            print(f"   Total Completed: {env_stats.get('total_requests_completed', 0)}")
            if 'avg_latency' in env_stats:
                print(f"   Avg End-to-End Latency: {env_stats['avg_latency']:.3f}s")
        
        if training_metrics:
            print(f"Training Metrics:")
            print(f"   Policy Loss: {training_metrics['policy_loss']:.6f}")
            print(f"   Value Loss: {training_metrics['value_loss']:.6f}")
            print(f"   Entropy Loss: {training_metrics['entropy_loss']:.6f}")
        
        if eval_info:
            print(f"Evaluation Results:")
            print(f"   Mean Reward: {eval_info['mean_reward']:.3f} ± {eval_info['std_reward']:.3f}")
            print(f"   Avg Quality: {eval_info['avg_quality_score']:.3f}")
            print(f"   Avg Latency: {eval_info['avg_latency']:.3f}s")
            print(f"   Invalid Rate: {eval_info['invalid_action_rate']:.3f}")
            print(f"   Completed Requests: {eval_info['total_completed_requests']}")
        
        print(f"{'='*60}")
    
    def _log_to_wandb(self, episode, episode_info, episode_rewards, training_metrics=None, eval_info=None):
        """Log metrics to wandb"""
        try:
            log_dict = {
                'episode': episode,
                'episode_reward': episode_info['total_reward'],
                'episode_length': episode_info['episode_length'],
                'completed_requests': episode_info['completed_requests'],
                'avg_quality_score': np.mean(episode_info['quality_scores']),
                'avg_immediate_reward': np.mean(episode_info['immediate_rewards']),
                'avg_step_reward': np.mean(episode_info['step_rewards']),
                'invalid_actions': episode_info['invalid_actions'],
                'valid_actions': episode_info['valid_actions'],
                'invalid_action_rate': episode_info['invalid_actions'] / Config.EPISODE_LENGTH,
                'valid_action_rate': episode_info['valid_actions'] / Config.EPISODE_LENGTH,
                'total_requests_created': episode_info.get('env_stats', {}).get('total_requests_created', 0),
                'total_requests_completed': episode_info.get('env_stats', {}).get('total_requests_completed', 0)
            }
            
            # Add cumulative metrics
            if len(episode_rewards) >= 1:
                log_dict['cumulative_reward'] = sum(episode_rewards)
                log_dict['avg_reward_so_far'] = np.mean(episode_rewards)
                
            if len(episode_rewards) >= 10:
                log_dict['recent_10_avg_reward'] = np.mean(episode_rewards[-10:])
                
            if len(episode_rewards) >= 20:
                log_dict['recent_20_avg_reward'] = np.mean(episode_rewards[-20:])

            # Add latency stats
            valid_latencies = [l for l in episode_info['latencies'] if l > 0]
            if valid_latencies:
                log_dict['avg_latency'] = np.mean(valid_latencies)
                log_dict['std_latency'] = np.std(valid_latencies)
                log_dict['min_latency'] = np.min(valid_latencies)
                log_dict['max_latency'] = np.max(valid_latencies)

            # Server utilization and action distribution
            for i in range(len(Config.SERVER_CAPACITIES)):
                log_dict[f'server_{i}_actions'] = episode_info['action_distribution'][i]
                log_dict[f'server_{i}_action_rate'] = episode_info['action_distribution'][i] / Config.EPISODE_LENGTH

            # Environment stats
            if 'env_stats' in episode_info:
                env_stats = episode_info['env_stats']
                if 'avg_latency' in env_stats:
                    log_dict['env_avg_latency'] = env_stats['avg_latency']
                    log_dict['env_std_latency'] = env_stats.get('std_latency', 0)

            if training_metrics:
                log_dict.update({
                    'policy_loss': training_metrics['policy_loss'],
                    'value_loss': training_metrics['value_loss'],
                    'entropy_loss': training_metrics['entropy_loss']
                })

            if eval_info:
                log_dict.update({
                    'eval_mean_reward': eval_info['mean_reward'],
                    'eval_std_reward': eval_info['std_reward'],
                    'eval_avg_quality': eval_info['avg_quality_score'],
                    'eval_avg_latency': eval_info['avg_latency'],
                    'eval_invalid_rate': eval_info['invalid_action_rate'],
                    'eval_completed_requests': eval_info['total_completed_requests']
                })
                
                # Evaluation server utilization
                for i in range(len(Config.SERVER_CAPACITIES)):
                    log_dict[f'eval_server_{i}_utilization'] = eval_info['avg_server_utilization'][i]

            wandb.log(log_dict)
            print(f"  Logged to wandb: {len(log_dict)} metrics")
            
        except Exception as e:
            print(f"  Wandb logging failed: {e}")
            self.wandb_available = False
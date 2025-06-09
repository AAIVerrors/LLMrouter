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
            
            # Add queue monitoring data if available
            if 'queue_details' in episode_info and episode_info['queue_details']:
                self._add_queue_metrics_to_log(log_dict, episode_info['queue_details'])
            
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
    
    def _add_queue_metrics_to_log(self, log_dict, queue_details):
        """Add detailed queue metrics to wandb log"""
        try:
            # Aggregate queue metrics
            total_queue_length = 0
            total_capacity = 0
            total_pending = 0
            server_utilizations = []
            
            for i, server_queue in enumerate(queue_details):
                if server_queue:
                    server_id = i
                    queue_length = server_queue.get('current_load', 0)
                    capacity = server_queue.get('capacity', 1)
                    utilization = server_queue.get('utilization', 0)
                    pending = server_queue.get('pending_completions', 0)
                    avg_proc_time = server_queue.get('avg_processing_time', 0)
                    completed_total = server_queue.get('completed_requests_total', 0)
                    recent_requests = server_queue.get('recent_requests_per_minute', 0)
                    
                    # Individual server metrics
                    log_dict.update({
                        f'queue_server_{server_id}_length': queue_length,
                        f'queue_server_{server_id}_utilization': utilization,
                        f'queue_server_{server_id}_pending': pending,
                        f'queue_server_{server_id}_avg_proc_time': avg_proc_time,
                        f'queue_server_{server_id}_completed_total': completed_total,
                        f'queue_server_{server_id}_recent_requests': recent_requests,
                    })
                    
                    # Detailed request information
                    processing_details = server_queue.get('processing_request_details', [])
                    if processing_details:
                        remaining_times = []
                        for req_detail in processing_details:
                            if 'estimated_completion' in req_detail and 'start_time' in req_detail:
                                remaining_time = req_detail['estimated_completion'] - req_detail.get('current_time', req_detail['start_time'])
                                remaining_times.append(max(0, remaining_time))
                        
                        if remaining_times:
                            log_dict[f'queue_server_{server_id}_avg_remaining_time'] = np.mean(remaining_times)
                            log_dict[f'queue_server_{server_id}_max_remaining_time'] = np.max(remaining_times)
                    
                    # Aggregate for system-wide metrics
                    total_queue_length += queue_length
                    total_capacity += capacity
                    total_pending += pending
                    server_utilizations.append(utilization)
            
            # System-wide queue metrics
            if total_capacity > 0:
                log_dict.update({
                    'queue_system_total_length': total_queue_length,
                    'queue_system_total_capacity': total_capacity,
                    'queue_system_load_factor': total_queue_length / total_capacity,
                    'queue_system_avg_utilization': np.mean(server_utilizations) if server_utilizations else 0,
                    'queue_system_max_utilization': np.max(server_utilizations) if server_utilizations else 0,
                    'queue_system_total_pending': total_pending,
                })
                
        except Exception as e:
            print(f"Failed to add queue metrics to log: {e}")
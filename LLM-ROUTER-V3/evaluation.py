import numpy as np
import torch
from enhanced_config import Config

class ModelEvaluator:
    """Utilities for evaluating trained models"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.env = trainer.env
        self.agent = trainer.agent
        self.data_loader = trainer.data_loader
    
    def evaluate_policy(self, num_episodes=10, verbose=True):
        """Comprehensive policy evaluation"""
        if verbose:
            print(f"Evaluating policy over {num_episodes} episodes...")
        
        results = {
            'episode_rewards': [],
            'quality_scores': [],
            'latencies': [],
            'invalid_actions': [],
            'action_distributions': [],
            'server_utilizations': [],
            'completion_rates': []
        }
        
        for episode in range(num_episodes):
            episode_result = self._run_evaluation_episode()
            
            results['episode_rewards'].append(episode_result['total_reward'])
            results['quality_scores'].extend(episode_result['quality_scores'])
            results['latencies'].extend([l for l in episode_result['latencies'] if l > 0])
            results['invalid_actions'].append(episode_result['invalid_actions'])
            results['action_distributions'].append(episode_result['action_distribution'])
            results['completion_rates'].append(episode_result['completed_requests'])
            
            if episode_result['server_utilizations']:
                avg_util = np.mean(episode_result['server_utilizations'], axis=0)
                results['server_utilizations'].append(avg_util)
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results)
        
        if verbose:
            self._print_evaluation_results(summary)
        
        return summary
    
    def _run_evaluation_episode(self):
        """Run a single evaluation episode"""
        state = self.env.reset()
        episode_info = {
            'rewards': [],
            'quality_scores': [],
            'latencies': [],
            'action_distribution': np.zeros(len(Config.SERVER_CAPACITIES)),
            'invalid_actions': 0,
            'completed_requests': 0,
            'server_utilizations': []
        }
        
        for step in range(Config.EPISODE_LENGTH):
            prompt = self.data_loader.get_random_prompt()
            action_mask = self.env.get_action_mask()
            
            # Use deterministic policy for evaluation
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(Config.DEVICE)
                action_mask_tensor = torch.FloatTensor(action_mask).to(Config.DEVICE) if action_mask is not None else None
                
                logits, _ = self.agent.network.forward(state_tensor, prompt, action_mask_tensor)
                action = torch.argmax(logits).cpu().item()
            
            next_state, reward, done, info = self.env.step(action, prompt)
            
            # Update metrics
            episode_info['rewards'].append(reward)
            episode_info['quality_scores'].append(info.get('quality_score', 0))
            episode_info['latencies'].append(info.get('estimated_latency', 0))
            episode_info['action_distribution'][action] += 1
            episode_info['completed_requests'] += info.get('completed_requests', 0)
            
            if not info.get('valid_action', True):
                episode_info['invalid_actions'] += 1
            
            # Track server utilizations
            server_utils = info.get('server_utilizations', [])
            if isinstance(server_utils, list) and len(server_utils) == len(Config.SERVER_CAPACITIES):
                episode_info['server_utilizations'].append(server_utils)
            
            state = next_state
            
            if done:
                break
        
        episode_info['total_reward'] = sum(episode_info['rewards'])
        return episode_info
    
    def _calculate_summary_stats(self, results):
        """Calculate summary statistics from evaluation results"""
        summary = {
            'mean_reward': np.mean(results['episode_rewards']),
            'std_reward': np.std(results['episode_rewards']),
            'min_reward': np.min(results['episode_rewards']),
            'max_reward': np.max(results['episode_rewards']),
            
            'mean_quality': np.mean(results['quality_scores']) if results['quality_scores'] else 0,
            'std_quality': np.std(results['quality_scores']) if results['quality_scores'] else 0,
            
            'mean_latency': np.mean(results['latencies']) if results['latencies'] else 0,
            'std_latency': np.std(results['latencies']) if results['latencies'] else 0,
            
            'total_invalid_actions': sum(results['invalid_actions']),
            'invalid_action_rate': sum(results['invalid_actions']) / (len(results['invalid_actions']) * Config.EPISODE_LENGTH),
            
            'mean_completion_rate': np.mean(results['completion_rates']),
            'total_completions': sum(results['completion_rates']),
            
            'action_distribution': np.mean(results['action_distributions'], axis=0) if results['action_distributions'] else np.zeros(len(Config.SERVER_CAPACITIES)),
            
            'server_utilization': np.mean(results['server_utilizations'], axis=0) if results['server_utilizations'] else np.zeros(len(Config.SERVER_CAPACITIES))
        }
        
        return summary
    
    def _print_evaluation_results(self, summary):
        """Print formatted evaluation results"""
        print("\nEvaluation Results:")
        print("=" * 50)
        
        print("Reward Statistics:")
        print(f"  Mean: {summary['mean_reward']:.3f} ± {summary['std_reward']:.3f}")
        print(f"  Range: [{summary['min_reward']:.3f}, {summary['max_reward']:.3f}]")
        
        print("\nQuality and Latency:")
        print(f"  Quality Score: {summary['mean_quality']:.3f} ± {summary['std_quality']:.3f}")
        print(f"  Latency: {summary['mean_latency']:.3f} ± {summary['std_latency']:.3f} seconds")
        
        print("\nAction Statistics:")
        print(f"  Invalid Action Rate: {summary['invalid_action_rate']:.3f}")
        print(f"  Total Invalid Actions: {summary['total_invalid_actions']}")
        print(f"  Mean Completion Rate: {summary['mean_completion_rate']:.1f} requests/episode")
        
        print("\nAction Distribution:")
        for i, count in enumerate(summary['action_distribution']):
            print(f"  Server {i}: {count:.1f} actions/episode ({count/Config.EPISODE_LENGTH:.1%})")
        
        print("\nServer Utilization:")
        for i, util in enumerate(summary['server_utilization']):
            print(f"  Server {i}: {util:.1%} average utilization")
        
        print("=" * 50)
    
    def compare_with_baseline(self, baseline_policy="random", num_episodes=10):
        """Compare current policy with baseline"""
        print(f"Comparing with {baseline_policy} baseline...")
        
        # Evaluate current policy
        current_results = self.evaluate_policy(num_episodes, verbose=False)
        
        # Evaluate baseline policy
        baseline_results = self._evaluate_baseline(baseline_policy, num_episodes)
        
        # Print comparison
        self._print_comparison(current_results, baseline_results, baseline_policy)
        
        return current_results, baseline_results
    
    def _evaluate_baseline(self, policy_type, num_episodes):
        """Evaluate baseline policy"""
        if policy_type == "random":
            return self._evaluate_random_policy(num_episodes)
        elif policy_type == "round_robin":
            return self._evaluate_round_robin_policy(num_episodes)
        else:
            raise ValueError(f"Unknown baseline policy: {policy_type}")
    
    def _evaluate_random_policy(self, num_episodes):
        """Evaluate random action policy"""
        results = {
            'episode_rewards': [],
            'quality_scores': [],
            'latencies': [],
            'invalid_actions': [],
            'completion_rates': []
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_quality = []
            episode_latency = []
            invalid_count = 0
            completed = 0
            
            for step in range(Config.EPISODE_LENGTH):
                prompt = self.data_loader.get_random_prompt()
                action_mask = self.env.get_action_mask()
                
                # Random valid action
                valid_actions = np.where(action_mask > 0)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = np.random.randint(0, len(Config.SERVER_CAPACITIES))
                
                next_state, reward, done, info = self.env.step(action, prompt)
                
                episode_reward += reward
                episode_quality.append(info.get('quality_score', 0))
                episode_latency.append(info.get('estimated_latency', 0))
                completed += info.get('completed_requests', 0)
                
                if not info.get('valid_action', True):
                    invalid_count += 1
                
                state = next_state
                
                if done:
                    break
            
            results['episode_rewards'].append(episode_reward)
            results['quality_scores'].extend(episode_quality)
            results['latencies'].extend([l for l in episode_latency if l > 0])
            results['invalid_actions'].append(invalid_count)
            results['completion_rates'].append(completed)
        
        return self._calculate_summary_stats(results)
    
    def _evaluate_round_robin_policy(self, num_episodes):
        """Evaluate round-robin action policy"""
        results = {
            'episode_rewards': [],
            'quality_scores': [],
            'latencies': [],
            'invalid_actions': [],
            'completion_rates': []
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_quality = []
            episode_latency = []
            invalid_count = 0
            completed = 0
            current_server = 0
            
            for step in range(Config.EPISODE_LENGTH):
                prompt = self.data_loader.get_random_prompt()
                action_mask = self.env.get_action_mask()
                
                # Round-robin with fallback to next available server
                action = current_server
                attempts = 0
                while action_mask[action] == 0 and attempts < len(Config.SERVER_CAPACITIES):
                    action = (action + 1) % len(Config.SERVER_CAPACITIES)
                    attempts += 1
                
                current_server = (current_server + 1) % len(Config.SERVER_CAPACITIES)
                
                next_state, reward, done, info = self.env.step(action, prompt)
                
                episode_reward += reward
                episode_quality.append(info.get('quality_score', 0))
                episode_latency.append(info.get('estimated_latency', 0))
                completed += info.get('completed_requests', 0)
                
                if not info.get('valid_action', True):
                    invalid_count += 1
                
                state = next_state
                
                if done:
                    break
            
            results['episode_rewards'].append(episode_reward)
            results['quality_scores'].extend(episode_quality)
            results['latencies'].extend([l for l in episode_latency if l > 0])
            results['invalid_actions'].append(invalid_count)
            results['completion_rates'].append(completed)
        
        return self._calculate_summary_stats(results)
    
    def _print_comparison(self, current, baseline, baseline_name):
        """Print comparison between current and baseline policy"""
        print(f"\nPolicy Comparison: Trained vs {baseline_name.title()}")
        print("=" * 60)
        
        print(f"{'Metric':<25} {'Trained':<15} {baseline_name.title():<15} {'Improvement':<15}")
        print("-" * 60)
        
        metrics = [
            ('Mean Reward', 'mean_reward'),
            ('Quality Score', 'mean_quality'),
            ('Latency (s)', 'mean_latency'),
            ('Invalid Rate', 'invalid_action_rate'),
            ('Completions/Ep', 'mean_completion_rate')
        ]
        
        for metric_name, metric_key in metrics:
            current_val = current[metric_key]
            baseline_val = baseline[metric_key]
            
            if metric_key in ['mean_latency', 'invalid_action_rate']:
                # Lower is better
                improvement = (baseline_val - current_val) / baseline_val * 100 if baseline_val != 0 else 0
                improvement_str = f"{improvement:+.1f}%"
            else:
                # Higher is better
                improvement = (current_val - baseline_val) / baseline_val * 100 if baseline_val != 0 else 0
                improvement_str = f"{improvement:+.1f}%"
            
            print(f"{metric_name:<25} {current_val:<15.3f} {baseline_val:<15.3f} {improvement_str:<15}")
        
        print("=" * 60)
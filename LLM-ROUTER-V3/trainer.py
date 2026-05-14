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

import time

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

        include_quality_state = bool(getattr(Config, "INCLUDE_QUALITY_IN_STATE", True)) and not bool(
            getattr(Config, "USE_EM_EXACT_MATCH", False)
        )
        
        # queue part length depends on USE_UTIL
        # USE_UTIL=True  -> util(M)
        # USE_UTIL=False -> load(M) + capacity(M) = 2M
        # ---- State dimension for flat *interleaved* per-server features ----
        # Per-server features: util, slot_count, mu, (optional q), price_in, price_out
        per_server_dim = 2 + 1 + (1 if include_quality_state else 0) + 2
        state_dim = M * per_server_dim
        action_dim = M
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
        self.last_service_rate = list(Config.SERVICE_RATE)  # Default service rate
        
        self.training_in_progress = False

        # Print configuration summary
        if Config.ENABLE_CONSOLE_LOGGING:
            self.print_config_summary()
    
    def update_service_rate_from_episode(self, episode_record):
        """
        Update per-server service rate after each trajectory/episode.

        Estimate:
            mu_i = completed_requests_i / sum(actual_service_time_i)

        Prefer response['decode_time'] because it excludes queue waiting more than
        processing_latency does.
        """
        M = len(Config.SERVER_CAPACITIES)

        old_mu = np.asarray(self.last_service_rate, dtype=np.float64)
        new_mu = old_mu.copy()

        alpha = float(getattr(Config, "SERVICE_RATE_EMA_ALPHA", 1))
        min_samples = int(getattr(Config, "SERVICE_RATE_MIN_SAMPLES", 1))
        min_mu = float(getattr(Config, "SERVICE_RATE_MIN", 1e-4))
        max_mu = float(getattr(Config, "SERVICE_RATE_MAX", 5.0))

        counts = [0 for _ in range(M)]
        service_times = [[] for _ in range(M)]

        for req in episode_record:
            if req.get("status") != "completed":
                continue
            if req.get("episode") != self.current_episode:
                continue

            sid = req.get("server_id", None)
            if sid is None:
                continue

            try:
                sid = int(sid)
            except Exception:
                continue

            if sid < 0 or sid >= M:
                continue

            resp = req.get("response", {}) or {}

            # Prefer actual decoding/service time
            service_time = None
            if isinstance(resp, dict):
                service_time = resp.get("decode_time", None)

            # Fallback if decode_time is missing
            if service_time is None:
                service_time = req.get("processing_latency_raw", None)
            if service_time is None:
                service_time = req.get("processing_latency", None)

            try:
                service_time = float(service_time)
            except Exception:
                continue

            if not np.isfinite(service_time) or service_time <= 0:
                continue

            counts[sid] += 1
            service_times[sid].append(service_time)

        instant_mu = [None for _ in range(M)]

        for i in range(M):
            if counts[i] < min_samples or len(service_times[i]) == 0:
                continue

            total_service_time = float(np.sum(service_times[i]))
            if total_service_time <= 0:
                continue

            mu_hat = counts[i] / total_service_time
            mu_hat = float(np.clip(mu_hat, min_mu, max_mu))

            instant_mu[i] = mu_hat
            new_mu[i] = (1.0 - alpha) * old_mu[i] + alpha * mu_hat

        self.last_service_rate = new_mu.tolist()

        return {
            "service_rate_updated": self.last_service_rate,
            "service_rate_instant": instant_mu,
            "service_rate_counts": counts,
        }
        
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

        num_servers = len(Config.SERVER_CAPACITIES)
        episode_info = {
            'rewards': [],
            'quality_scores': [],
            'latencies': [],
            'latencies_clipped': [],
            'latencies_norm': [],
            'capacity_penalties': [],
            'invalid_actions': 0,
            'valid_actions': 0,
            'service_rate': [],
            'prices': [],
            'prices_norm': [],
            'queue_length': [],
            'reward_sum_per_server': {i: 0.0 for i in range(num_servers)},
            'reward_count_per_server': {i: 0 for i in range(num_servers)},
            'mean_reward_per_server': {i: None for i in range(num_servers)},
            'active_server_ids_with_completed_requests': [],
            'active_server_avg_rewards': [],
            'server_avg_reward_tilted_ConfigT': None,
        }
        
        # Count processed prompts per server
        num_processed = [0] * num_servers
        counter = 0
        for index, req in enumerate(record):
            server_id = req['server_id']
            if server_id is not None and req['status'] == 'completed' and req['episode'] == self.current_episode:
                num_processed[server_id] += 1
                counter += 1
            # if index == len(record) - 1:
            #     req['reward'] += counter # Add bonus for last request in episode

        # Compute per-server service rate
        duration = Config.EPISODE_TIME_INTERVAL * Config.INTERVAL_LENGTH
        # service_rate = [n / duration + self.last_service_rate[index] for index, n in enumerate(num_processed)]
        service_rate = list(self.last_service_rate)
        
        # print(service_rate)

        # Store service_rate for use in agent
        # self.last_service_rate = service_rate

        episode_reward = 0

        for req in record:
            server_id = req.get('server_id')
            reward = float(req.get('reward', 0.0))

            episode_info['rewards'].append(reward)
            episode_info['queue_length'].append(req['queue_length'])

            if server_id is not None and req['status'] == 'completed' and req['episode'] == self.current_episode:
                episode_info['reward_sum_per_server'][server_id] += reward
                episode_info['reward_count_per_server'][server_id] += 1

            if req['status'] == 'completed' and req['episode'] == self.current_episode:
                episode_info['quality_scores'].append(req['quality_score'])
                episode_info['latencies'].append(req.get('processing_latency_raw', req.get('processing_latency')))
                episode_info['latencies_clipped'].append(req.get('processing_latency_clipped', req.get('processing_latency')))
                if req.get('processing_latency_norm') is not None:
                    episode_info['latencies_norm'].append(req.get('processing_latency_norm'))
                episode_info['prices'].append(req.get('price_raw', req.get('price')))
                if req.get('price_norm') is not None:
                    episode_info['prices_norm'].append(req.get('price_norm'))
                
            if req['status'] == 'completed' and req['episode'] == self.current_episode:
                episode_info['valid_actions'] += 1
            else:
                episode_info['invalid_actions'] += 1

            episode_reward += reward

        for server_id in range(num_servers):
            count = episode_info['reward_count_per_server'][server_id]
            if count > 0:
                episode_info['mean_reward_per_server'][server_id] = (
                    episode_info['reward_sum_per_server'][server_id] / count
                )

        active_server_ids = []
        active_server_avg_rewards = []
        for server_id, mean_reward in episode_info['mean_reward_per_server'].items():
            if mean_reward is not None:
                active_server_ids.append(server_id)
                active_server_avg_rewards.append(float(mean_reward))

        episode_info['active_server_ids_with_completed_requests'] = active_server_ids
        episode_info['active_server_avg_rewards'] = active_server_avg_rewards
        episode_info['server_avg_reward_tilted_ConfigT'] = self.tilted_log_mean_exp_value(
            active_server_avg_rewards,
            beta=Config.T,
        )

        episode_info['total_reward'] = episode_reward
        episode_info['episode_length'] = len(record)
        episode_info['service_rate'] = service_rate

        return episode_info

    def run_episode(self) -> dict:
        """Run a single episode using Poisson prompt generator"""
        
        M = len(Config.SERVER_CAPACITIES)
        include_quality_state = bool(getattr(Config, 'INCLUDE_QUALITY_IN_STATE', True)) and not bool(getattr(Config, 'USE_EM_EXACT_MATCH', False))

        _slot_count_norm = max(
            float(Config.POISSON_ARRIVAL_RATE) * float(Config.INTERVAL_LENGTH), 1.0
        )

        def build_state(loads, slot_counts=None):
            """Build **flat** state vector with per-server interleaved features.

            Per-server layout (concatenated for i=0..M-1):
              [ util_i, slot_count_i, mu_i, price_in_i, price_out_i ]   (F = 5)

            slot_count_i: number of times server i was selected in the current
                          time slot, normalized by expected arrivals per slot
                          (POISSON_ARRIVAL_RATE * INTERVAL_LENGTH).

            Returns:
              np.ndarray of shape [M * 5]
            """
            M = len(Config.SERVER_CAPACITIES)

            PRICE_SCALE = 1e6

            # prices are tuples: (input_price, output_price)
            price_in = [float(a[0]) * PRICE_SCALE for a in Config.PRICE]
            price_out = [float(a[1]) * PRICE_SCALE for a in Config.PRICE]

            feats_flat = []
            for i in range(M):
                cap = float(Config.SERVER_CAPACITIES[i])
                load = float(loads[i])
                util = load / max(cap, 1.0)
                sc = float(slot_counts[i]) / _slot_count_norm if slot_counts is not None else 0.0
                mu = float(self.last_service_rate[i])

                feats_flat.extend([util, sc, mu, price_in[i], price_out[i]])

            # sanity check
            expected_len = M * 5
            if len(feats_flat) != expected_len:
                raise ValueError(f"build_state length mismatch: got {len(feats_flat)}, expected {expected_len} (M={M}, F=5).")

            return np.array(feats_flat, dtype=np.float32)   # [M * 5]

        M = len(Config.SERVER_CAPACITIES)
        slot_counts = np.zeros(M, dtype=np.float32)
        state = build_state(self.env.reset(), slot_counts)
        self.env.clean_prompt_queue()
        
        # state = [self.env.reset()[index]/c for index,c in enumerate(Config.SERVER_CAPACITIES)] + [1] * len(Config.SERVER_CAPACITIES) + price

        # state = [self.env.reset()[index]/c for index,c in enumerate(Config.SERVER_CAPACITIES)] + self.last_service_rate + price


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

            prompt_entry = self.env.get_next_prompt()
            if not prompt_entry:
                time.sleep(0.1)
                continue
            arrival_time = time.time()
            prompt_time_slot = int((arrival_time - start) // Config.INTERVAL_LENGTH)

            if isinstance(prompt_entry, dict):
                prompt = prompt_entry.get('prompt', '')
                ground_truth = (
                    prompt_entry.get('output')
                    or prompt_entry.get('answer')
                    or prompt_entry.get('target')
                )
            else:
                prompt = str(prompt_entry)
                ground_truth = None


            if Config.NAIVE_PPO:
                current_loads = self.env.get_state()
                state = build_state(current_loads, slot_counts)
            else:
                if current != current_time_slot:
                    current_loads = self.env.get_state()
                    slot_counts[:] = 0.0
                    state = build_state(current_loads, slot_counts)
                    current = current_time_slot
                else:
                    # Same slot: refresh only the dynamic slot_count column
                    state = build_state(
                        [state[i * 5] * float(Config.SERVER_CAPACITIES[i]) for i in range(M)],
                        slot_counts,
                    )

            action_mask = self.env.get_action_mask()

            state_for_action = state.copy()

            action, log_prob, value, next_counter = self.agent.get_action(
                state_for_action,
                prompt,
                action_mask,
                service_rate=self.last_service_rate,
                round_robin_counter=robin_counter,
            )

            robin_counter = next_counter
            slot_counts[action] += 1.0

            next_state, done = self.env.step(action, prompt, ground_truth)
            queue_length = next_state.tolist()

            self.buffer.add_step(
                time_slot=prompt_time_slot,
                route_time=current_time_slot,
                state=state_for_action,
                prompt=prompt,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=0,
                action_mask=action_mask,
                service_rate=list(self.last_service_rate),
                queue_length=queue_length,
            )

            if done:
                break
            

        
        # Wait for all prompts to be processed
        self.env.pause_prompt_generator()
        # time.sleep(2)
        
        # --- Pause and clean servers before training ---
        while self.env.check_get_episode_completed() == False:
            # print("Waiting for all prompts to be processed...")
            time.sleep(1)
        
        episode_record = self.env.get_episode_data()
        
        time.sleep(1)
        
        self.env.pause_all_servers()
        time.sleep(1)
        
        self.env.clean_all_queues()
        time.sleep(1)
        
        # Update buffer rewards with actual episode rewards
        for i, req in enumerate(episode_record):
            if i < len(self.buffer.current_episode):
                self.buffer.current_episode[i]['reward'] = req['reward']
                
        # Optional: per-round min-max normalization for latency and price, then recompute rewards.
        # This keeps raw fields (processing_latency / price) unchanged for logging, and only overwrites req['reward'].
        if getattr(Config, "ROUND_MINMAX_NORM_ENABLE", False) and len(episode_record) > 0:
            eps = float(getattr(Config, "ROUND_MINMAX_NORM_EPS", 1e-8))
            clip01 = bool(getattr(Config, "ROUND_MINMAX_CLIP_01", True))
            norm_lat = bool(getattr(Config, "ROUND_MINMAX_NORM_LATENCY", True))
            norm_price = bool(getattr(Config, "ROUND_MINMAX_NORM_PRICE", True))
            only_completed = bool(getattr(Config, "ROUND_MINMAX_ONLY_COMPLETED", True))
            use_price_raw = bool(getattr(Config, "ROUND_MINMAX_USE_PRICE_RAW_IF_AVAILABLE", True))

            idxs = []
            lats = []
            prices = []

            for i, req in enumerate(episode_record):
                if only_completed and req.get("status") != "completed":
                    continue

                lat = req.get("processing_latency", None)
                if lat is None or (not np.isfinite(lat)):
                    continue

                # Prefer unscaled price if provided by the environment.
                if use_price_raw and ("price_raw" in req) and (req.get("price_raw") is not None):
                    pr = req.get("price_raw", 0.0)
                else:
                    pr = req.get("price", 0.0)
                if pr is None or (not np.isfinite(pr)):
                    pr = 0.0

                pr_base = float(pr)

                idxs.append(i)
                lats.append(float(lat))
                prices.append(float(pr_base))

            def _minmax(arr):
                if len(arr) == 0:
                    return []
                a = np.asarray(arr, dtype=np.float64)
                mn = float(np.min(a))
                mx = float(np.max(a))
                if (mx - mn) < eps:
                    out = np.zeros_like(a)
                else:
                    out = (a - mn) / (mx - mn + eps)
                if clip01:
                    out = np.clip(out, 0.0, 1.0)
                return out.tolist()

            lat_norms = _minmax(lats) if norm_lat else [0.0] * len(lats)
            price_norms = _minmax(prices) if norm_price else [0.0] * len(prices)

            for k, i in enumerate(idxs):
                req = episode_record[i]
                q = float(req.get("quality_score", 0.0) or 0.0)
                reward = float(getattr(Config, "ALPHA", 1.0)) * q

                if norm_lat:
                    req["processing_latency_norm"] = float(lat_norms[k])
                    reward -= float(getattr(Config, "BETA", 0.0)) * float(lat_norms[k])

                if norm_price:
                    req["price_norm"] = float(price_norms[k])
                    reward -= float(getattr(Config, "REWARD_GAMMA", 0.0)) * float(price_norms[k])

                clip = getattr(Config, "REWARD_CLIP", None)
                if clip is not None and clip > 0:
                    reward = float(max(min(reward, clip), -clip))

                req["reward"] = float(reward)
                if i < len(self.buffer.current_episode):
                    self.buffer.current_episode[i]["reward"] = float(reward)

        # Update greedy utility history (EMA latency/cost, optional queue-conditioned stats).
        # Safe no-op if the agent does not implement update_server_stats.
        try:
            self.agent.update_server_stats(episode_record)
        except Exception as e:
            print(f"[WARN] update_server_stats failed: {e}")
            
        service_rate_info = self.update_service_rate_from_episode(episode_record)

        if self.wandb_available:
            import wandb
            log_dict = {}
            for i, mu in enumerate(service_rate_info["service_rate_updated"]):
                log_dict[f"service_rate/ema_server_{i}"] = mu

            for i, mu in enumerate(service_rate_info["service_rate_instant"]):
                if mu is not None:
                    log_dict[f"service_rate/instant_server_{i}"] = mu

            for i, c in enumerate(service_rate_info["service_rate_counts"]):
                log_dict[f"service_rate/count_server_{i}"] = c

            wandb.log(log_dict, step=self.current_episode)

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

    @staticmethod

    def softmax_value(x, tau=Config.T_QUEUE):
        """
        softmax_tau(x) = tau * log( mean_i exp(x_i / tau) )
        As tau -> 0, softmax_tau(x) -> max(x)
        """
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
    
        tau = max(float(tau), 1e-12)
        z = x / tau
        m = np.max(z)  # stability
        return float(tau * (m + np.log(np.mean(np.exp(z - m)))))
    
    @staticmethod
    def softmin_value(x, tau=Config.T_REWARD):
        """
        TERM-compatible tilted aggregation:
            value = log(mean_i exp(beta * x_i)) / beta

        For backward compatibility, the argument name remains ``tau``, but it is
        interpreted as the tilt parameter ``beta`` here.

        beta < 0  -> softmin-like (emphasises smaller values)
        beta -> 0 -> mean
        beta > 0  -> softmax-like (emphasises larger values)
        """
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None

        beta = float(tau)
        if abs(beta) < 1e-8:
            return float(np.mean(x))

        z = beta * x
        m = np.max(z)  # stability
        return float((m + np.log(np.mean(np.exp(z - m)))) / beta)

    @staticmethod
    def tilted_log_mean_exp_value(x, beta=Config.T):
        """
        Same aggregation form used in PPO training in router_network:
            agg(x) = log(mean_i exp(beta * x_i)) / beta

        - beta < 0: softmin-like aggregation
        - beta > 0: softmax-like aggregation
        - beta ~= 0: arithmetic mean

        This helper ignores NaN/None values and returns None when no valid values exist.
        """
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None

        beta = float(beta)
        if abs(beta) < 1e-8:
            return float(np.mean(x))

        z = beta * x
        m = np.max(z)
        return float((m + np.log(np.mean(np.exp(z - m)))) / beta)

    
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
                if self.agent.scheduler is not None:
                    self.agent.scheduler.step()

                if training_metrics:
                    print(f"   Policy Loss: {training_metrics['policy_loss']:.6f}")
                    print(f"   Value Loss: {training_metrics['value_loss']:.6f}")
                
                if self.wandb_available:
                    
                    lat_list = episode_info['latencies']
                    p50 = self.safe_percentile(lat_list, 50)
                    p90 = self.safe_percentile(lat_list, 90)
                    p99 = self.safe_percentile(lat_list, 99)
                    p95 = self.safe_percentile(lat_list, 95)
                    
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
                        "softmin_reward": self.softmin_value(episode_info['rewards']) if episode_info.get('rewards') else None,
                        "reward/server_avg_reward_tilted_ConfigT": episode_info.get('server_avg_reward_tilted_ConfigT'),
                        "reward/server_avg_reward_active_server_count": len(episode_info.get('active_server_avg_rewards', [])),
                    
                        "policy_loss": training_metrics['policy_loss'] if training_metrics else None,
                        "value_loss": training_metrics['value_loss'] if training_metrics else None,
                        "entropy_loss": training_metrics['entropy_loss'] if training_metrics else None,
                    
                        "quality_scores": float(np.mean(episode_info['quality_scores'])) if episode_info.get('quality_scores') else None,
                        "latencies": float(np.mean(episode_info['latencies'])) if episode_info.get('latencies') else None,
                    
                        "p50": p50,
                        "p90": p90,
                        "p99": p99,
                        'p95': p95,
                    
                        "latencies_clipped": float(np.mean(episode_info['latencies_clipped'])) if episode_info.get('latencies_clipped') else None,
                        "latencies_norm": float(np.mean(episode_info['latencies_norm'])) if episode_info.get('latencies_norm') else None,
                        "price_norm": float(np.mean(episode_info['prices_norm'])) if episode_info.get('prices_norm') else None,
                        "Jain_fairness_index": fairness,

                        'valid_total_request_ratio': episode_info.get('valid_actions', None)/episode_info.get('episode_length', None),
                    
                        "price": price_mean,
                    
                        "throughput_per_episode/requests_completed": episode_info.get('valid_actions', None),
                        "request_total_numbers": episode_info.get('episode_length', None),
                    
                        "returns": training_metrics['rewards_returns'] if training_metrics else None,
                        "term_returns": training_metrics['term_rewards_returns'] if training_metrics else None,
                        "min_rewards": training_metrics['min_rewards'] if training_metrics else None,
                        "gap_rewards": float(np.max(episode_info['rewards']) - np.min(episode_info['rewards'])) if episode_info.get('rewards') else None,
                        "cumulated_avg_rewards_return": training_metrics['cumulated_avg_rewards'] if training_metrics else None,
                        "entropy of route distribution": training_metrics['entropy of route distribution'] if training_metrics else None,
                        "approx_kl": training_metrics['approx_kl'] if training_metrics else None,
                    }, step=episode)


                # Add server usage percentage if available
                if self.wandb_available and training_metrics and 'server_usage_percentage' in training_metrics:
                    for server_id, usage in training_metrics['server_usage_percentage'].items():
                        wandb.log({f"server_{server_id}_usage": usage}, step=episode)

                # Add avg reward per server
                if self.wandb_available and episode_info and 'mean_reward_per_server' in episode_info:
                    for server_id, mean_reward in episode_info['mean_reward_per_server'].items():
                        if mean_reward is not None:
                            wandb.log({f"reward/server_{server_id}_avg_reward": float(mean_reward)}, step=episode)
                        wandb.log({f"reward/server_{server_id}_count": int(episode_info['reward_count_per_server'][server_id])}, step=episode)
                        wandb.log({f"reward/server_{server_id}_sum": float(episode_info['reward_sum_per_server'][server_id])}, step=episode)
                    agg_server_avg_reward = episode_info.get('server_avg_reward_tilted_ConfigT')
                    if agg_server_avg_reward is not None:
                        wandb.log({"reward/server_avg_reward_softmin_ConfigT": float(agg_server_avg_reward)}, step=episode)

                if self.wandb_available and episode_info and episode_info.get("queue_length"):
                    M = len(Config.SERVER_CAPACITIES)
                    caps = np.asarray(Config.SERVER_CAPACITIES, dtype=np.float32)
                    caps = np.maximum(caps, 1.0)  # avoid divide-by-zero
                
                    q_raw = episode_info["queue_length"]
                
                    # q_vec should be length-M (one value per server)
                    # If q_raw is a history (T snapshots, each length-M), use the LAST snapshot.
                    if isinstance(q_raw, (list, tuple, np.ndarray)) and len(q_raw) > 0 and isinstance(q_raw[0], (list, tuple, np.ndarray)):
                        # history case: q_raw = [ [q1..qM], [q1..qM], ... ]
                        q_vec = np.asarray(q_raw[-1], dtype=np.float32)
                    else:
                        # already per-server vector case: q_raw = [q1..qM]
                        q_vec = np.asarray(q_raw, dtype=np.float32)
                
                    # Safety: ensure correct shape
                    if q_vec.shape[0] == M:
                        ratio_vec = q_vec / caps
                        ratio_list = ratio_vec.tolist()
                
                        # ---- Per-server: ONLY length and ratio ----
                        for server_id in range(M):
                            wandb.log({f"queue_length/server_{server_id}_queue_length_ratio": float(ratio_vec[server_id])}, step=episode)
                            wandb.log({f"queue_length/server_{server_id}_queue_length": float(q_vec[server_id])}, step=episode)
                
                        # ---- Others remain (summary) ----
                        wandb.log({"queue_length/server_queue_length_ratio_p90": self.safe_percentile(ratio_list, 90)}, step=episode)
                        wandb.log({"queue_length/server_queue_length_ratio_p50": self.safe_percentile(ratio_list, 50)}, step=episode)
                        wandb.log({"queue_length/server_queue_length_ratio_p95": self.safe_percentile(ratio_list, 95)}, step=episode)
                        wandb.log({"queue_length/server_queue_length_ratio_p99": self.safe_percentile(ratio_list, 99)}, step=episode)
                
                        max_ratio = float(np.max(ratio_vec))
                        min_ratio = float(np.min(ratio_vec))
                
                        wandb.log({"queue_length/server_queue_length_ratio_mean": float(np.mean(ratio_vec))}, step=episode)
                        wandb.log({"queue_length/server_queue_length_ratio_min": min_ratio}, step=episode)  # FIXED
                        wandb.log({"queue_length/server_queue_length_ratio_max": max_ratio}, step=episode)  # FIXED
                        wandb.log({"queue_length/server_queue_length_ratio_gap": max_ratio - min_ratio}, step=episode)
                        wandb.log({"queue_length/server_queue_length_ratio_softmax": self.softmax_value(ratio_list)}, step=episode)

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
                            "reward/server_avg_reward_tilted_ConfigT": episode_info.get('server_avg_reward_tilted_ConfigT'),
                            "reward/server_avg_reward_active_server_count": len(episode_info.get('active_server_avg_rewards', [])),
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
                    if episode_info and 'mean_reward_per_server' in episode_info:
                        for server_id, mean_reward in episode_info['mean_reward_per_server'].items():
                            if mean_reward is not None:
                                wandb.log({f"reward/server_{server_id}_avg_reward": float(mean_reward)}, step=episode)
                            wandb.log({f"reward/server_{server_id}_count": int(episode_info['reward_count_per_server'][server_id])}, step=episode)
                            wandb.log({f"reward/server_{server_id}_sum": float(episode_info['reward_sum_per_server'][server_id])}, step=episode)
                        agg_server_avg_reward = episode_info.get('server_avg_reward_tilted_ConfigT')
                        if agg_server_avg_reward is not None:
                            wandb.log({"reward/server_avg_reward_softmin_ConfigT": float(agg_server_avg_reward)}, step=episode)
                    
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
            time.sleep(2)
        
        print(f"\nTraining completed!")
        
        try:
            self.env.pause_all_servers()
            self.env.clean_all_queues()
            if self.wandb_available:
                wandb.finish()
            print("Training cleanup completed successfully")
        except Exception as e:
            print(f"Cleanup failed: {e}")
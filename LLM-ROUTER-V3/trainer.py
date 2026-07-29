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
        per_server_dim = 4
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
    
    def compute_route_distribution(self, episode_record):
        """Per-task routing distribution P(endpoint | task) and pairwise JS.

        The marginal fairness panels (Jain, max_endpoint_share, ...) cannot
        distinguish uniform routing from per-task specialization whose mixture
        happens to be uniform -- and the latter is precisely success. The JS
        divergence between the tasks' conditional routing distributions is the
        quantitative definition of "prompt-aware": structurally 0 for every
        prompt-blind baseline, > 0 iff different tasks route differently.
        (Mechanism evidence only; the fairness calibration is checked by the
        analytic vhat, not by JS.)
        """
        M = len(Config.SERVER_CAPACITIES)
        counts = {}
        for req in episode_record:
            if req.get("episode") != self.current_episode:
                continue
            sid = req.get("server_id")
            task = str(req.get("task_type", "") or "")
            if sid is None or not task:
                continue
            sid = int(sid)
            if 0 <= sid < M:
                counts.setdefault(task, np.zeros(M, dtype=np.float64))[sid] += 1

        dists = {t: c / c.sum() for t, c in counts.items() if c.sum() > 0}
        if len(dists) < 2:
            return dists, None

        def _js(p, q):
            m = 0.5 * (p + q)
            def _kl(a, b):
                mask = a > 0
                return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))))
            return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

        tasks = sorted(dists.keys())
        pairs = [
            _js(dists[tasks[i]], dists[tasks[j]])
            for i in range(len(tasks)) for j in range(i + 1, len(tasks))
        ]
        return dists, float(np.mean(pairs))

    def update_service_rate_from_episode(self, episode_record):
        """
        Update per-server service rate after each trajectory/episode.

        Estimate:
            mu_i = completed_requests_i / sum(service_time_i)

        service_time is PURE generation time (completion_time - start_time,
        with start_time stamped at dequeue), which excludes queue waiting.
        End-to-end latency is intentionally not used, as it would bias mu
        downward whenever requests queue.
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

            # Service rate must be estimated from PURE service (generation)
            # time only. start_time is stamped at dequeue (generation start,
            # see environment request loop), so completion_time - start_time
            # excludes queue waiting. End-to-end latency
            # (processing_latency*, = completion - arrival) is deliberately
            # NOT used here: it includes queue wait and would bias mu
            # downward under load.
            def _valid_time(x):
                if x is None:
                    return None
                try:
                    v = float(x)
                except (TypeError, ValueError):
                    return None
                return v if (np.isfinite(v) and v > 0) else None

            # Prefer the precomputed decode_time; if it is missing or invalid
            # in this snapshot, recompute the same pure-service quantity from
            # the raw timing attributes (reliably present on completions).
            resp = req.get("response", {}) or {}
            service_time = _valid_time(resp.get("decode_time") if isinstance(resp, dict) else None)
            if service_time is None:
                st = _valid_time(req.get("start_time"))
                ct = _valid_time(req.get("completion_time"))
                if st is not None and ct is not None and ct > st:
                    service_time = ct - st

            if service_time is None:
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

            wandb.define_metric("episode")
            wandb.define_metric("*", step_metric="episode")   # 所有指标都按 episode 索引
            
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
            'dollar_costs': [],
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
                if req.get('dollar_cost') is not None:
                    episode_info['dollar_costs'].append(req.get('dollar_cost'))
                
            if req['status'] == 'completed' and req['episode'] == self.current_episode:
                episode_info['valid_actions'] += 1
            else:
                episode_info['invalid_actions'] += 1

            # Break the non-completed cases out by cause. They mean different
            # things and must not be read as one number: 'failed' is a routing
            # consequence (dispatched to a server that could not accept it, i.e.
            # the capacity penalty fired), whereas 'api_transient_failed' is
            # endpoint noise the router is not responsible for. A non-trivial
            # failed rate means the reward is dominated by capacity penalties
            # and every other metric this episode is confounded.
            _st = req.get('status')
            if _st == 'failed':
                episode_info['n_failed'] = episode_info.get('n_failed', 0) + 1
            elif _st == 'api_transient_failed':
                episode_info['n_api_failed'] = episode_info.get('n_api_failed', 0) + 1
            elif _st != 'completed':
                episode_info['n_incomplete'] = episode_info.get('n_incomplete', 0) + 1

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

        if bool(getattr(Config, "ROUTE_DIST_LOG", True)):
            _dists, _js = self.compute_route_distribution(record)
            episode_info['route_dists'] = {t: d.tolist() for t, d in _dists.items()}
            episode_info['specialization_js'] = _js

        return episode_info

    def run_episode(self) -> dict:
        """Run a single episode using Poisson prompt generator"""
        
        M = len(Config.SERVER_CAPACITIES)
        include_quality_state = bool(getattr(Config, 'INCLUDE_QUALITY_IN_STATE', True)) and not bool(getattr(Config, 'USE_EM_EXACT_MATCH', False))

        _slot_count_norm = max(
            float(Config.POISSON_ARRIVAL_RATE) * float(Config.INTERVAL_LENGTH), 1.0
        )

        def build_state(loads, residuals, alloc_ratio=None):
            M = len(Config.SERVER_CAPACITIES)
            PRICE_SCALE = 1e6
            price_in  = [float(a[0]) * PRICE_SCALE for a in Config.PRICE]
            price_out = [float(a[1]) * PRICE_SCALE for a in Config.PRICE]
            max_lat = max(float(getattr(Config, "MAX_LAT", 30.0)), 1e-6)

            use_alloc = bool(getattr(Config, "QUEUE_USE_ALLOC", False))
            F = 5 + (1 if use_alloc else 0)
            feats_flat = []
            for i in range(M):
                cap  = float(Config.SERVER_CAPACITIES[i])
                load = float(loads[i])
                util = load / max(cap, 1.0)
                resid = float(residuals[i]) / max_lat   # in-flight elapsed, normalized
                # Frozen Config.SERVICE_RATE, not the online EMA, and for the
                # same reason QUOTA_USE_FROZEN_MU already gives the quota: the
                # measured req/s is ENDOGENOUS. mu_hat = completions / service
                # time, and service time depends on how long the answers are,
                # which depends on which task types got routed here -- so the
                # policy changing its routing moves its own observation. That
                # closed loop sat inside the "STATIC capability channel" the
                # quality tower reads, making the static channel not static and
                # muddying dual/quality_spread with drift that has nothing to do
                # with capability. It also broke cross-run reproducibility,
                # since re-measuring SERVICE_RATE shifts the input distribution
                # the tower learned against.
                # The online EMA is still tracked and logged (service_rate/*);
                # it is simply no longer an input to the policy.
                mu = (
                    float(Config.SERVICE_RATE[i])
                    if bool(getattr(Config, "STATE_USE_FROZEN_MU", True))
                    else float(self.last_service_rate[i])
                )
                row = [
                    util, resid,                     # dyn (2)
                    mu, price_in[i], price_out[i],   # stat (3)
                ]
                if use_alloc:
                    # frozen fleet-normalized alloc count, appended AFTER stat
                    ar = 1.0 if alloc_ratio is None else float(alloc_ratio[i])
                    row.append(ar)
                feats_flat.extend(row)

            if len(feats_flat) != M * F:
                raise ValueError(f"build_state length mismatch: got {len(feats_flat)}, expected {M*F}")
            return np.array(feats_flat, dtype=np.float32)

        M = len(Config.SERVER_CAPACITIES)
        F = 5
        slot_counts = np.zeros(M, dtype=np.float32)
        state = build_state(self.env.reset(), self.env.get_residuals())
        # Pin this episode's arrival trace. Which prompts arrive and when are a
        # function of (DATASET_SEED, episode) alone, so every algorithm replays
        # an identical workload and per-episode differences against a baseline
        # are genuinely paired. Returns the trace origin, used below as the
        # episode clock so interval boundaries and arrivals share one t0.
        _trace_t0, _n_arrivals = self.env.begin_episode_arrivals(self.current_episode)
        print(f"[arrivals] episode {self.current_episode}: {_n_arrivals} requests pinned")

        # state = [self.env.reset()[index]/c for index,c in enumerate(Config.SERVER_CAPACITIES)] + [1] * len(Config.SERVER_CAPACITIES) + price

        # state = [self.env.reset()[index]/c for index,c in enumerate(Config.SERVER_CAPACITIES)] + self.last_service_rate + price


        start = _trace_t0
        robin_counter = 0  # Initialize round-robin counter
        
        current_time_slot = 0
        time_slot_buffer = []
        current = -1

        # Cumulative (per-EPISODE) allocation counter for QUEUE_USE_ALLOC.
        # alloc_counts accumulates every routing decision across the whole
        # episode (reset here at episode start, NOT per interval); at each
        # interval boundary its fleet-normalized value (counts/mean, =1 fair)
        # is frozen into alloc_ratio_frozen as a per-server state feature.
        alloc_counts = np.zeros(M, dtype=np.float32)
        alloc_ratio_frozen = np.ones(M, dtype=np.float32)
        
        while True:
            routing_start = time.time()
            current_time_slot = (routing_start - start) // Config.INTERVAL_LENGTH
            if current_time_slot >= Config.EPISODE_TIME_INTERVAL :
                print(f"Episode {self.current_episode} timed out after {Config.EPISODE_TIME_INTERVAL} seconds")
                break
            # # NEW: 当前 interval 内已经过去多久 / 还剩多久（归一化到 [0, 1]）
            # elapsed_in_episode = routing_start - start
            # elapsed_in_interval = elapsed_in_episode - current_time_slot * Config.INTERVAL_LENGTH
            # interval_remaining_norm = max(
            #     0.0,
            #     1.0 - float(elapsed_in_interval) / max(float(Config.INTERVAL_LENGTH), 1e-6)
            # )

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
                task_type = str(prompt_entry.get('task_type', '') or '')
            else:
                prompt = str(prompt_entry)
                ground_truth = None
                task_type = ''


            if Config.NAIVE_PPO:
                current_loads = self.env.get_state()
                state = build_state(current_loads, self.env.get_residuals(), alloc_ratio_frozen)
            else:
                if current != current_time_slot:
                    # freeze the CUMULATIVE (episode-so-far) fleet-normalized
                    # allocation counts. NOT reset per interval: alloc_counts
                    # accumulates over the whole episode (long-horizon burden),
                    # so the ratio is smoother than a single interval's ~1-2
                    # counts. =1 fair, >1 this server got more than its share.
                    tot = float(alloc_counts.sum())
                    if tot > 0.0:
                        alloc_ratio_frozen = alloc_counts / max(tot / M, 1e-6)
                    else:
                        alloc_ratio_frozen = np.ones(M, dtype=np.float32)
                    current_loads = self.env.get_state()
                    # fresh telemetry snapshot at the interval boundary
                    state = build_state(current_loads, self.env.get_residuals(), alloc_ratio_frozen)
                    current = current_time_slot
                else:
                    # within the interval: reuse the frozen snapshot as-is
                    # (util / residual / mu / prices are all frozen)
                    pass

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
            # slot_counts[action] += 1.0

            next_state, done = self.env.step(action, prompt, ground_truth, task_type=task_type)
            alloc_counts[action] += 1.0   # count this routing for the interval
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
        # Bounded drain: never let one stuck request hang the whole run.
        _drain_timeout = float(getattr(Config, "EPISODE_COMPLETION_TIMEOUT", 180))
        _drain_start = time.time()
        while self.env.check_get_episode_completed() == False:
            if time.time() - _drain_start > _drain_timeout:
                print(
                    f"[WARN] episode {self.current_episode}: drain timed out after "
                    f"{_drain_timeout:.0f}s; proceeding with completed requests only "
                    f"(shortfall shows up as outcome/incomplete_rate)."
                )
                break
            time.sleep(1)
        
        episode_record = self.env.get_episode_data()
        self.env.pause_all_servers()
        self.env.clean_all_queues()
        
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

                # True dollars for the rolling-window scheme; fall back to
                # the env's normalized price only if dollar_cost is missing.
                pr = req.get("dollar_cost")
                if pr is None and use_price_raw:
                    pr = req.get("price_raw")
                if pr is None:
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

            # Price: percentile normalization in sqrt space (Router-R1's
            # scheme). Two window modes:
            #   frozen  -- bounds computed ONCE from a uniform-routing
            #              calibration file and shared by every arm/episode:
            #              price_norm is the same function of dollars
            #              everywhere, so rewards stay commensurable across
            #              methods (a per-run rolling window drifts with the
            #              run's own policy).
            #   rolling -- Router-R1's original adaptive buffer (kept as the
            #              fallback / ablation path).
            if norm_price:
                import math as _math
                _q_lo, _q_hi = getattr(Config, "ROUND_MINMAX_PERCENTILES", (5, 95))
                _sq = [_math.sqrt(max(float(p), 0.0)) for p in prices]
                _mode = str(getattr(Config, "ROUND_MINMAX_WINDOW_MODE", "rolling")).lower()
                if _mode == "frozen":
                    if not hasattr(self, "_price_frozen_bounds"):
                        import json as _json
                        _fp = str(getattr(Config, "ROUND_MINMAX_WINDOW_FILE",
                                          "price_window_v2p.json"))
                        _costs = _json.load(open(_fp))["costs"]
                        _sqc = np.sqrt(np.maximum(
                            np.asarray(_costs, dtype=np.float64), 0.0))
                        self._price_frozen_bounds = (
                            float(np.percentile(_sqc, _q_lo)),
                            float(np.percentile(_sqc, _q_hi)),
                        )
                        print(f"[price-norm] frozen window {_fp}: n={len(_costs)}, "
                              f"sqrt-bounds={self._price_frozen_bounds}")
                    _lo, _hi = self._price_frozen_bounds
                else:
                    if not hasattr(self, "_price_window"):
                        from collections import deque
                        _W = int(getattr(Config, "ROUND_MINMAX_WINDOW", 1000))
                        self._price_window = deque(maxlen=_W)
                        for _pin, _pout in Config.PRICE:
                            for _tin, _tout in ((350, 60), (350, 300)):
                                self._price_window.append(
                                    _math.sqrt(max(_pin * _tin + _pout * _tout, 0.0)))
                    self._price_window.extend(_sq)
                    _arr = np.asarray(self._price_window, dtype=np.float64)
                    _lo = float(np.percentile(_arr, _q_lo))
                    _hi = float(np.percentile(_arr, _q_hi))
                if (_hi - _lo) < eps:
                    price_norms = [0.5] * len(_sq)
                else:
                    price_norms = [
                        float(np.clip((s - _lo) / (_hi - _lo), 0.0, 1.0))
                        for s in _sq
                    ]
            else:
                price_norms = [0.0] * len(prices)

            for k, i in enumerate(idxs):
                req = episode_record[i]
                q = float(req.get("quality_score", 0.0) or 0.0)
                reward = float(getattr(Config, "ALPHA", 1.0)) * q

                if norm_lat:
                    req["processing_latency_norm"] = float(lat_norms[k])
                    reward -= float(getattr(Config, "BETA", 0.0)) * float(lat_norms[k])
                else:
                    # env deferred the latency term together with price;
                    # re-apply the absolute min(lat, MAX_LAT)/MAX_LAT here.
                    _ml = max(float(getattr(Config, "MAX_LAT", 30.0)), 1e-6)
                    _ln = min(float(lats[k]), _ml) / _ml
                    req["processing_latency_norm"] = float(_ln)
                    reward -= float(getattr(Config, "BETA", 0.0)) * float(_ln)

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

            # Weighted Jain over z = counts/mu (frozen). 1.0 = capacity-
            # proportional; plain Jain's optimum (equal counts) scores BELOW
            # 1 here whenever mu is heterogeneous. episode = this episode's
            # counts (noisy); cumulative = run-to-date totals (paper metric).
            _mu = np.asarray(Config.SERVICE_RATE, dtype=float)
            _cts = np.asarray(service_rate_info["service_rate_counts"], dtype=float)
            if not hasattr(self, "_wjain_cum_counts"):
                self._wjain_cum_counts = np.zeros_like(_cts)
            self._wjain_cum_counts = self._wjain_cum_counts + _cts

            def _wjain(c):
                z = c / np.maximum(_mu, 1e-9)
                s2 = float(np.square(z).sum())
                return float(z.sum() ** 2 / (len(z) * s2)) if s2 > 0 else 1.0

            log_dict["fairness/wjain_episode"] = _wjain(_cts)
            log_dict["fairness/wjain_cumulative"] = _wjain(self._wjain_cum_counts)

            # Log WITHOUT an explicit step, matching every other wandb.log
            # call in this file. Mixing explicit step=episode here with the
            # step-less (auto-incrementing) calls elsewhere made wandb drop
            # these logs after episode 0 (their episode-numbered step fell
            # behind the auto-incremented internal step). Use the "episode"
            # data key for the x-axis instead, like the main metrics block.
            log_dict["episode"] = self.current_episode
            wandb.log(log_dict)

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
            
            if str(getattr(Config, "FAIRNESS_MODE", "legacy")).lower() != "wf_dual":
                # Legacy quota/softmin modes: FAIR is the live strength knob,
                # optionally warmed up over episodes.
                warmup = int(getattr(Config, "FAIR_WARMUP_EPISODES", 0))
                fair_target = float(getattr(Config, "FAIR_TARGET", getattr(Config, "FAIR", 1.0)))
                if warmup <= 0:
                    Config.FAIR = fair_target
                else:
                    Config.FAIR = min(fair_target, fair_target * episode / max(warmup, 1))
                print(f"[Episode {episode}] FAIR = {Config.FAIR:.3f}")
            elif episode == 0:
                print(
                    f"[wf_dual] fairness strength = FAIR_DELTA "
                    f"({getattr(Config, 'FAIR_DELTA', 0.5)}) + adaptive nu "
                    f"(dual={'on' if getattr(Config, 'FAIR_DUAL_ENABLE', True) else 'off'}, "
                    f"nu_init={getattr(Config, 'FAIR_MU_INIT', 0.0)}); "
                    f"Config.FAIR/FAIR_TARGET are IGNORED in this mode."
                )

            self.current_episode = episode  # Update current episode
            self.env.set_episode(episode)  # Update environment episode tracking
            
            print(f"\nRunning Episode {episode}...")
 
            # Run episode
            episode_info = self.get_episode_data()
            self.episode_rewards.append(episode_info['rewards'])
            self.episode_stats.append(episode_info)

            # ---- 每个 interval 用到的不同 server 数的平均 ----
            interval_servers = {}
            for step in self.buffer.current_episode:
                ts = int(step["time_slot"])
                interval_servers.setdefault(ts, set()).add(int(step["action"]))
            avg_active_servers_per_interval = (
                float(np.mean([len(s) for s in interval_servers.values()]))
                if interval_servers else 0.0
            )
            
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

                    # SLO violation rate = fraction of completed requests with
                    # end-to-end latency above each threshold (tail metric).
                    _slo_lats = episode_info.get('latencies') or []
                    slo_dict = {}
                    for _T in getattr(Config, "SLO_LATENCIES", [10.0]):
                        slo_dict[f"slo/violation_rate_{int(_T)}s"] = (
                            float(np.mean([1.0 if float(l) > float(_T) else 0.0 for l in _slo_lats]))
                            if _slo_lats else None
                        )

                    # Actual dollar cost: total $ spent this episode and $/request.
                    _dollars = episode_info.get('dollar_costs') or []
                    _qs = episode_info.get('quality_scores') or []
                    cost_dict = {
                        "cost/dollar_total_per_episode": float(np.sum(_dollars)) if _dollars else None,
                        "cost/dollar_per_request": float(np.mean(_dollars)) if _dollars else None,
                        # Quality bought per dollar: the single number that says
                        # whether a cheaper policy is actually a better one, or
                        # is just buying less quality.
                        "cost/quality_per_dollar": (
                            float(np.mean(_qs) / np.mean(_dollars))
                            if _qs and _dollars and float(np.mean(_dollars)) > 0 else None
                        ),
                    }


                    # Request outcome breakdown. completion_rate should sit at
                    # ~1.0. A non-trivial failed_rate means dispatches are
                    # landing on full queues, so the capacity penalty dominates
                    # the reward and every other metric this episode is
                    # confounded; api_failed_rate is endpoint noise instead and
                    # is not something the router can act on.
                    _n_done = int(episode_info.get('valid_actions', 0))
                    _n_all = _n_done + int(episode_info.get('invalid_actions', 0))
                    _den = max(_n_all, 1)
                    outcome_dict = {
                        "outcome/completion_rate": _n_done / _den,
                        "outcome/failed_rate": int(episode_info.get('n_failed', 0)) / _den,
                        "outcome/api_failed_rate": int(episode_info.get('n_api_failed', 0)) / _den,
                        "outcome/incomplete_rate": int(episode_info.get('n_incomplete', 0)) / _den,
                    }

                    wandb.log({
                        "episode": episode,
                        **slo_dict,
                        **cost_dict,
                        **outcome_dict,
                        "total_reward": float(np.sum(episode_info['rewards'])) if episode_info.get('rewards') else None,
                        "mean_reward": float(np.mean(episode_info['rewards'])) if episode_info.get('rewards') else None,
                        "std_reward": float(np.std(episode_info['rewards'])) if episode_info.get('rewards') else None,
                        "softmin_reward": self.softmin_value(episode_info['rewards']) if episode_info.get('rewards') else None,
                        "reward/server_avg_reward_tilted_ConfigT": episode_info.get('server_avg_reward_tilted_ConfigT'),
                        "reward/server_avg_reward_active_server_count": len(episode_info.get('active_server_avg_rewards', [])),
                        "route/avg_active_servers_per_interval": avg_active_servers_per_interval,   # ← 新增
                    
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
                        "fairness/quota_F_load": training_metrics.get('quota_f_load') if training_metrics else None,
                        "fairness/quota_jain_norm_load": training_metrics.get('quota_jain_norm_load') if training_metrics else None,

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
                        # Same estimator, honest name: the logged quantity is
                        # the per-request mean KL; the joint interval KL is
                        # approximately the SUM of per-request KLs.
                        "train/mean_request_kl": training_metrics['approx_kl'] if training_metrics else None,
                        # ---- WF-dual fairness (constrained, RCPO) ----
                        "fair/v_observed": training_metrics.get('fair_v_obs') if training_metrics else None,
                        "fair/v_expected_policy": training_metrics.get('fair_v_hat') if training_metrics else None,
                        "fair/v_mismatch": training_metrics.get('fair_v_mismatch') if training_metrics else None,
                        "fair/v_sampling": training_metrics.get('fair_v_sampling') if training_metrics else None,
                        "fair/g": training_metrics.get('fair_g') if training_metrics else None,
                        "dual/h": training_metrics.get('fair_h') if training_metrics else None,
                        "dual/nu": training_metrics.get('fair_nu') if training_metrics else None,
                        "dual/saturated": training_metrics.get('fair_nu_saturated') if training_metrics else None,
                        "train/mean_interval_sum_kl": training_metrics.get('mean_interval_sum_kl') if training_metrics else None,
                        "fair/excluded_intervals": training_metrics.get('fair_excluded_intervals') if training_metrics else None,
                        "route/specialization_js": episode_info.get('specialization_js'),
                        # PPO signal diagnostics: is the policy stuck because the
                        # advantage signal collapsed? explained_variance says why.
                        "raw_adv_std": training_metrics.get('raw_adv_std') if training_metrics else None,
                        "raw_adv_absmean": training_metrics.get('raw_adv_absmean') if training_metrics else None,
                        "explained_variance": training_metrics.get('explained_variance') if training_metrics else None,
                        "policy_entropy": training_metrics.get('policy_entropy') if training_metrics else None,
                        # Dual-tower actor: per-server spread of each score.
                        # quality_spread>0 => quality tower differentiates servers;
                        # queue_spread>0 => queue tower fires on load.
                        "dual/quality_spread": training_metrics.get('dual_quality_spread') if training_metrics else None,
                        "dual/queue_spread": training_metrics.get('dual_queue_spread') if training_metrics else None,
                        # Learnable per-tower balance scales (s_q*quality + s_k*queue).
                        "dual/scale_q": training_metrics.get('dual_scale_q') if training_metrics else None,
                        "dual/scale_k": training_metrics.get('dual_scale_k') if training_metrics else None,
                        # Running-norm divisors (ACTOR_DUAL_RUNNING_NORM). rms_q
                        # climbing while rms_k stays flat is the magnitude drift
                        # that used to shrink queue influence; after the norm the
                        # effective weights are s_q and s_k alone.
                        "dual/rms_q": training_metrics.get('dual_rms_q') if training_metrics else None,
                        "dual/rms_k": training_metrics.get('dual_rms_k') if training_metrics else None,
                        # Per-head gradient norm (pre-clip). grad_queue ~ 0 with a
                        # healthy grad_quality => queue tower has no learning
                        # signal (flat spread is correct, not starved).
                        "dual/grad_quality": training_metrics.get('grad_quality') if training_metrics else None,
                        "dual/grad_queue": training_metrics.get('grad_queue') if training_metrics else None,
                        "dual/grad_fuse": training_metrics.get('grad_fuse') if training_metrics else None,
                        # Lagrangian fairness: mu rises while Jain sits below the
                        # floor and decays back toward 0 once it is satisfied.
                        "lagrangian/mu": training_metrics.get('lagrangian_mu') if training_metrics else None,
                        "lagrangian/jain_ema": training_metrics.get('lagrangian_jain_ema') if training_metrics else None,
                        # Episode-cumulative concentration. effective_endpoints
                        # = M * Jain = 1/HHI reads as "how many of the M
                        # endpoints are in effective use" (8.0 = all, 2.0 =
                        # collapsed onto two).
                        "fairness/jain_cumulative": training_metrics.get('jain_cumulative') if training_metrics else None,
                        "fairness/effective_endpoints": training_metrics.get('effective_endpoints') if training_metrics else None,
                        "fairness/max_endpoint_share": training_metrics.get('max_endpoint_share') if training_metrics else None,
                        # Load-balancing performance. makespan = max_m z_m (>1
                        # means the worst server cannot clear within one
                        # interval); overload_frac = share of servers with z>1.
                        # These must not regress while fairness improves.
                        "load/makespan": training_metrics.get('lb_makespan') if training_metrics else None,
                        "load/overload_frac": training_metrics.get('lb_overload_frac') if training_metrics else None,
                        # Scale-free companions to makespan: both divide by the
                        # mean, so they separate "how uneven" from "how loaded"
                        # and stay comparable across arrival rates.
                        "load/cov": training_metrics.get('lb_cov') if training_metrics else None,
                        # Fraction of per-request reward variance the V^task
                        # head explains. The interval baseline removes that
                        # share of the gradient noise, so this is the number
                        # that says whether the baseline is doing anything.
                        "vtask/explained_var": training_metrics.get('vtask_explained_var') if training_metrics else None,
                        # Variance b_t actually removes from the interval
                        # advantage -- the number that says whether the
                        # baseline earns its place. explained_var above scores
                        # per-request prediction and understates it, since b_t
                        # only has to predict the interval mean.
                        "vtask/explained_var_interval": training_metrics.get('vtask_explained_var_interval') if training_metrics else None,
                        # Spread of the head's predictions. ~0 means it has
                        # collapsed onto the unconditional mean, which is the
                        # failure mode explained_var=0 cannot distinguish from
                        # varied-but-uncorrelated predictions.
                        "vtask/pred_std": training_metrics.get('vtask_pred_std') if training_metrics else None,
                        "vtask/replay_size": training_metrics.get('vtask_replay_size') if training_metrics else None,
                        "vtask/baseline_mean": training_metrics.get('vtask_baseline_mean') if training_metrics else None,
                        "load/imbalance": training_metrics.get('lb_imbalance') if training_metrics else None,
                    })


                # Add server usage percentage if available
                if self.wandb_available and training_metrics and 'server_usage_percentage' in training_metrics:
                    for server_id, usage in training_metrics['server_usage_percentage'].items():
                        wandb.log({f"server_{server_id}_usage": usage})

                # Per-task conditional routing distribution P(endpoint | task):
                # the mechanism evidence the marginal panels cannot show.
                if self.wandb_available and episode_info.get('route_dists'):
                    _rd_log = {}
                    for _task, _dist in episode_info['route_dists'].items():
                        for _m, _p in enumerate(_dist):
                            _rd_log[f"route/dist_{_task}_{_m}"] = float(_p)
                    wandb.log(_rd_log)

                # Add avg reward per server
                if self.wandb_available and episode_info and 'mean_reward_per_server' in episode_info:
                    for server_id, mean_reward in episode_info['mean_reward_per_server'].items():
                        if mean_reward is not None:
                            wandb.log({f"reward/server_{server_id}_avg_reward": float(mean_reward)})
                        wandb.log({f"reward/server_{server_id}_count": int(episode_info['reward_count_per_server'][server_id])})
                        wandb.log({f"reward/server_{server_id}_sum": float(episode_info['reward_sum_per_server'][server_id])})
                    agg_server_avg_reward = episode_info.get('server_avg_reward_tilted_ConfigT')
                    if agg_server_avg_reward is not None:
                        wandb.log({"reward/server_avg_reward_softmin_ConfigT": float(agg_server_avg_reward)})

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
                            wandb.log({f"queue_length/server_{server_id}_queue_length_ratio": float(ratio_vec[server_id])})
                            wandb.log({f"queue_length/server_{server_id}_queue_length": float(q_vec[server_id])})
                
                        # ---- Others remain (summary) ----
                        wandb.log({"queue_length/server_queue_length_ratio_p90": self.safe_percentile(ratio_list, 90)})
                        wandb.log({"queue_length/server_queue_length_ratio_p50": self.safe_percentile(ratio_list, 50)})
                        wandb.log({"queue_length/server_queue_length_ratio_p95": self.safe_percentile(ratio_list, 95)})
                        wandb.log({"queue_length/server_queue_length_ratio_p99": self.safe_percentile(ratio_list, 99)})
                
                        max_ratio = float(np.max(ratio_vec))
                        min_ratio = float(np.min(ratio_vec))
                
                        wandb.log({"queue_length/server_queue_length_ratio_mean": float(np.mean(ratio_vec))})
                        wandb.log({"queue_length/server_queue_length_ratio_min": min_ratio})  # FIXED
                        wandb.log({"queue_length/server_queue_length_ratio_max": max_ratio})  # FIXED
                        wandb.log({"queue_length/server_queue_length_ratio_gap": max_ratio - min_ratio})
                        wandb.log({"queue_length/server_queue_length_ratio_softmax": self.softmax_value(ratio_list)})

                # Add server scores if available
                # if training_metrics and 'each_server_score' in training_metrics:
                #     for server_id, score in training_metrics['each_server_score'].items():
                #         wandb.log({f"server_{server_id}_score": score})
                        
                # # Add min score server 
                # if training_metrics and 'min_score_server' in training_metrics:
                #     wandb.log({"min_score_server": training_metrics['min_score_server']})
                    
                # # Add mean reward per server if available
                # if training_metrics and 'mean_reward_per_server' in training_metrics:
                #     for server_id, mean_reward in training_metrics['mean_reward_per_server'].items():
                #         wandb.log({f"server_{server_id}_reward": mean_reward})
                        
                # # Add min mean reward server
                # if training_metrics and 'min_mean_reward_server' in training_metrics:
                #     wandb.log({"min_mean_reward_server": training_metrics['min_mean_reward_server']})
                    
                # # Add return of each server
                # if training_metrics and 'each_server_returns' in training_metrics:
                #     for server_id, ret in training_metrics['each_server_returns'].items():
                #         wandb.log({f"server_{server_id}_trajectory_return": ret})
                
                # # Add min return server
                # if training_metrics and 'min_return_server' in training_metrics:
                #     wandb.log({"min_return_server": training_metrics['min_return_server']})

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
                        })
                
                # server usage percentage
                server_usage = {i: 0 for i in range(len(Config.SERVER_CAPACITIES))}
                for step in self.buffer.current_episode:
                    server_usage[step['action']] += 1
                total_actions = sum(server_usage.values())
                if total_actions > 0:
                    server_usage = {k: v / total_actions for k, v in server_usage.items()}
                if self.wandb_available:
                    for server_id, usage in server_usage.items():
                        wandb.log({f"server_{server_id}_usage": usage})
                    if episode_info and 'mean_reward_per_server' in episode_info:
                        for server_id, mean_reward in episode_info['mean_reward_per_server'].items():
                            if mean_reward is not None:
                                wandb.log({f"reward/server_{server_id}_avg_reward": float(mean_reward)})
                            wandb.log({f"reward/server_{server_id}_count": int(episode_info['reward_count_per_server'][server_id])})
                            wandb.log({f"reward/server_{server_id}_sum": float(episode_info['reward_sum_per_server'][server_id])})
                        agg_server_avg_reward = episode_info.get('server_avg_reward_tilted_ConfigT')
                        if agg_server_avg_reward is not None:
                            wandb.log({"reward/server_avg_reward_softmin_ConfigT": float(agg_server_avg_reward)})
                    
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
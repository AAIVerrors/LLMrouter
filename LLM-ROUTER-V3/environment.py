from vllm import LLM, SamplingParams
import torch
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from config import Config
from queueMonitor import QueueUpdateMonitor
import torch.multiprocessing as mp
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from queue import Empty

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

@dataclass
class Request:
    """Represents a single request in the system"""
    id: str
    prompt: str
    arrival_time: float
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    processing_latency: Optional[float] = None
    quality_score: Optional[float] = None
    server_id: Optional[int] = None
    response: Optional[Dict] = None
    status: str = "pending"

def server_worker_process(model_name: str,
                         capacity: int,
                         server_id: int,
                         request_queue,
                         response_queue,
                         stats_dict,
                         running,
                         gpu_id: Optional[int] = None):
    """Worker function for server process"""
    try:
        # Set GPU
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)
        
        # Initialize vLLM inside the process
        # llm = LLM(
        #     model=model_name,
        #     tensor_parallel_size=1,
        #     gpu_memory_utilization=0.4,
        #     max_num_batched_tokens=256,  
        #     max_model_len=128,  
        #     trust_remote_code=True,
        #     enforce_eager=True,
        #     disable_custom_all_reduce=True,
        # )
        
        # sampling_params = SamplingParams(
        #     temperature=0.7,
        #     max_tokens=128,
        #     top_p=0.95,
        # )
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'  # Important for batch generation
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with FlashAttention 2
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        
        model.eval()
        print(f"Server {server_id} ({model_name}) initialized on GPU {gpu_id}")
        
        # Maintain local processing queue
        processing_queue = []
        
        # Process requests
        while running.value:
            try:
                request = request_queue.get(timeout=0.1)
                if request is None:
                    continue

                # Get current state with proper queue length
                current_queue_length = stats_dict['queue_length']

                # Get queue state before processing
                queue_state_before = {
                    'current_load': current_queue_length,
                    'utilization': current_queue_length / capacity,
                    'pending_completions': 0,
                    'avg_processing_time': stats_dict.get('avg_processing_time', 0.0),
                    'queue_requests': list(stats_dict.get('processing_requests', []))
                }

                # Start processing
                request.status = 'processing'
                request.start_time = time.time()
                
                # Update local queue and shared stats
                processing_queue.append(request)
                new_load = len(processing_queue)
                
                # Update shared stats atomically
                stats_dict.update({
                    'processing_requests': [r.id for r in processing_queue],
                    'current_load': new_load
                })

                # Calculate queue state after adding
                queue_state_after = {
                    'current_load': new_load,
                    'utilization': new_load / capacity,
                    'pending_completions': 0,
                    'avg_processing_time': stats_dict.get('avg_processing_time', 0.0),
                    'queue_requests': [r.id for r in processing_queue]
                }

                try:
                    # Generate response
                    inputs = tokenizer(
                        request.prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128
                    ).to(model.device)
                    
                    # Process request
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=200,
                                temperature=0.7,
                                top_p=0.95,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                min_length=10,
                                use_cache=True
                            )
                            
                    # Decode response
                    input_length = inputs['input_ids'].size(1)
                    if outputs.size(1) <= input_length:
                        raise ValueError("No new tokens generated")

                    response_text = tokenizer.decode(
                        outputs[0][input_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )

                    # Update request with completion info
                    request.completion_time = time.time()
                    request.processing_latency = request.completion_time - request.start_time
                    request.status = 'completed'
                    request.response = {
                        "response_text": response_text,
                        "decode_time": request.processing_latency,
                        "tokens_generated": len(outputs[0]) - inputs['input_ids'].size(1)
                    }

                    # Update queue state atomically after completion
                    processing_requests = list(stats_dict.get('processing_requests', []))
                    if request.id in processing_requests:
                        processing_requests.remove(request.id)
                        
                    stats_dict.update({
                        'processing_requests': processing_requests,
                        'queue_length': max(0, len(processing_requests)),
                        'current_load': max(0, len(processing_requests)),
                        'completed_count': stats_dict.get('completed_count', 0) + 1,
                        'total_processing_time': stats_dict.get('total_processing_time', 0.0) + request.processing_latency
                    })

                    # Get final queue state
                    queue_state_final = {
                        'current_load': len(processing_requests),
                        'utilization': len(processing_requests) / capacity,
                        'pending_completions': 0,
                        'avg_processing_time': stats_dict.get('avg_processing_time', 0.0),
                        'queue_requests': processing_requests.copy()
                    }

                    response_queue.put((request, queue_state_before, queue_state_final))
                    print(f"Server {server_id}: Request {request.id} completed. "
                          f"Queue: {queue_state_before['current_load']}->{queue_state_final['current_load']}")

                except Exception as e:
                    print(f"Error processing request {request.id}: {e}")
                    # Update queue state on failure
                    processing_requests = list(stats_dict.get('processing_requests', []))
                    if request.id in processing_requests:
                        processing_requests.remove(request.id)
                    
                    stats_dict.update({
                        'processing_requests': processing_requests,
                        'queue_length': max(0, len(processing_requests)),
                        'current_load': max(0, len(processing_requests))
                    })
                    
                    request.status = 'failed'
                    request.completion_time = time.time()
                    response_queue.put((request, queue_state_before, None))

            except Empty:
                continue

    except Exception as e:
        print(f"Server {server_id} error: {e}")
        import traceback
        traceback.print_exc()

    print(f"Server {server_id} shutting down")

class LLMServerWrapper:
    """Wrapper for LLM server with process management"""
    def __init__(self, 
                 model_name: str,
                 capacity: int,
                 server_id: int,
                 manager: mp.Manager,
                 response_queue,
                 gpu_id: Optional[int] = None):
        self.model_name = model_name
        self.capacity = capacity
        self.server_id = server_id
        self.gpu_id = gpu_id
        self.running = mp.Value('b', True)
        
        # Add lock for queue operations
        self.queue_lock = manager.Lock()
        
        # Create queues and shared data
        self.request_queue = manager.Queue()
        self.response_queue = response_queue
        self.stats = manager.dict({
            'current_load': 0,
            'completed_count': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'processing_requests': [],
            'queue_length': 0  # Add explicit queue length tracking
        })
        
        # Start process
        self.process = mp.Process(
            target=server_worker_process,
            args=(model_name, capacity, server_id, 
                  self.request_queue, self.response_queue,
                  self.stats, self.running, gpu_id)
        )
        self.process.start()
    
    def put_request(self, request: Request):
        """Add request to this server's queue"""
        with self.queue_lock:
            # Update queue length before adding request
            current_length = self.stats['queue_length']
            self.stats['queue_length'] = current_length + 1
            self.stats['current_load'] = current_length + 1
            self.request_queue.put(request)
    
    def stop(self):
        """Stop the server process"""
        self.running.value = False
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
    
    def can_accept_request(self) -> bool:
        """Check if server can accept new requests"""
        return self.stats.get('current_load', 0) < self.capacity
    
    def get_current_load(self) -> int:
        """Get current number of requests being processed"""
        return self.stats.get('current_load', 0)
    
    def get_stats(self) -> Dict:
        """Get server statistics"""
        return dict(self.stats)

class QualityScorer:
    """Quality scoring for prompt-model pairs"""
    def __init__(self):
        self.model_elo_scores = {
            0: 1200,  # GPT-2
            1: 1100,  # Qwen
        }
    
    def compute_quality_score(self, prompt: str, server_id: int) -> float:
        """Compute quality score for prompt-server pair"""
        prompt_length = len(prompt.split())
        prompt_complexity = min(prompt_length / 50.0, 2.0)
        
        base_elo = self.model_elo_scores.get(server_id, 1150)
        base_score = base_elo / 1000.0
        
        if server_id == 0:
            complexity_adjustment = max(0.8, 1.2 - prompt_complexity * 0.2)
        else:
            complexity_adjustment = min(1.2, 0.9 + prompt_complexity * 0.15)
        
        noise = np.random.normal(0, 0.05)
        final_score = base_score * complexity_adjustment + noise
        return max(0.1, final_score)

def response_collector_worker(response_queue,
                            collected_rewards,
                            total_completed,
                            running,
                            config_alpha: float,
                            config_beta: float,
                            config_lambda: float,
                            queue_monitor=None):
    """Worker function for response collection"""
    print("Response collector started")
    
    while running.value:
        try:
            data = response_queue.get(timeout=0.1)
            if data is None:
                continue
            
            request, queue_state_before, queue_state_after = data
            current_time = time.time()
            
            if request.status == 'completed':
                # Calculate reward
                if (request.completion_time is not None and 
                    request.arrival_time is not None and 
                    request.quality_score is not None):
                    
                    latency = min(
                        request.completion_time - request.arrival_time,
                        10.0  
                    )
                    
                    # Scale latency penalty
                    normalized_latency = latency / 10.0
                    quality_reward = config_alpha * request.quality_score
                    latency_penalty = config_beta * normalized_latency
                    reward = quality_reward - latency_penalty
                    
                    print(f"Response completed - ID: {request.id}, "
                          f"Quality: {request.quality_score:.3f}, "
                          f"Latency: {latency:.3f}s, "
                          f"Reward: {reward:.3f}")
                    
                    # Log to queue monitor
                    if queue_monitor:
                        queue_monitor.log_request_completed(
                            server_id=request.server_id,
                            request_id=request.id,
                            current_time=current_time,
                            reward=reward,
                            queue_state_before=queue_state_before,
                            queue_state_after=queue_state_after
                        )
                    
                    collected_rewards.append(reward)
                    with total_completed.get_lock():
                        total_completed.value += 1
                    
                else:
                    print(f"Warning: Incomplete response data for request {request.id}")
                
            elif request.status == 'failed':
                collected_rewards.append(-config_lambda)
                print(f"Response failed - ID: {request.id}, Penalty: {-config_lambda}")
                
                if queue_monitor:
                    queue_monitor.log_request_failed(
                        server_id=request.server_id,
                        request_id=request.id,
                        prompt=request.prompt,
                        current_time=current_time,
                        reason="Processing failed"
                    )
                
        except Empty:
            continue
        except EOFError:
            if not running.value:
                break
            continue
        except Exception as e:
            print(f"Error in response collector: {e}")
            continue

class EnhancedRouterEnvironment:
    """Environment for LLM request routing with parallel processing"""
    def __init__(self, enable_monitoring=True):
        # Set monitoring flag first
        self.enable_monitoring = enable_monitoring
        
        # Initialize queue monitor before other components
        if self.enable_monitoring:
            try:
                self.queue_monitor = QueueUpdateMonitor(wandb_available=False)
            except ImportError:
                print("Queue monitor not available")
                self.enable_monitoring = False
        
        # Initialize manager
        self.manager = mp.Manager()
        
        # Initialize components
        self.quality_scorer = QualityScorer()
        
        # Create shared response queue
        self.response_queue = self.manager.Queue()
        
        # Response collection
        self.response_collector_running = mp.Value('b', True)
        self.collected_rewards = self.manager.list()
        self.total_completed = mp.Value('i', 0)
        
        # Start response collector
        self.response_collector = mp.Process(
            target=response_collector_worker,
            args=(self.response_queue, 
                  self.collected_rewards, 
                  self.total_completed,
                  self.response_collector_running, 
                  Config.ALPHA, 
                  Config.BETA, 
                  Config.LAMBDA,
                  self.queue_monitor if self.enable_monitoring else None)
        )
        self.response_collector.daemon = True
        self.response_collector.start()
        
        # Create server processes
        print(f"Initializing {len(Config.MODEL_NAMES)} servers...")
        self.servers = []
        self.gpu_ids = list(range(torch.cuda.device_count()))
        
        for i in range(len(Config.MODEL_NAMES)):
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)] if self.gpu_ids else None
            print(f"Creating server {i} with model {Config.MODEL_NAMES[i]} on GPU {gpu_id}")
            server = LLMServerWrapper(
                model_name=Config.MODEL_NAMES[i],
                capacity=Config.SERVER_CAPACITIES[i],
                server_id=i,
                manager=self.manager,
                response_queue=self.response_queue,
                gpu_id=gpu_id
            )
            self.servers.append(server)
        
        # Environment state
        self.current_time = 0.0
        self.time_step = 0.1
        self.request_counter = 0
        self.current_episode = 0
        
        # Wait for initialization
        print("Waiting for servers to initialize...")
        time.sleep(5)
        
        self.reset()
        print("Environment initialization complete!")
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_time = 0.0
        self.request_counter = 0
        self.collected_rewards[:] = []
        with self.total_completed.get_lock():
            self.total_completed.value = 0
        
        if self.enable_monitoring and hasattr(self, 'queue_monitor'):
            self.queue_monitor.reset()
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        state = []
        for server in self.servers:
            load = server.get_current_load()
            utilization = load / server.capacity if server.capacity > 0 else 0
            state.extend([load, utilization])
        return np.array(state, dtype=np.float32)
    
    def get_action_mask(self) -> np.ndarray:
        """Get valid actions mask"""
        return np.array([server.can_accept_request() for server in self.servers], dtype=np.float32)
    
    def step(self, action: int, prompt: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute routing action"""
        # Update current time
        self.current_time += self.time_step
        
        # Collect accumulated rewards
        step_rewards = sum(self.collected_rewards)
        self.collected_rewards[:] = []
        completed_count = self.total_completed.value
        with self.total_completed.get_lock():
            self.total_completed.value = 0
        
        # Create request
        request_id = f"req_{self.request_counter}"
        self.request_counter += 1
        
        request = Request(
            id=request_id,
            prompt=prompt,
            arrival_time=self.current_time,
            server_id=action,
        )
        
        # Check server availability
        server = self.servers[action]
        if not server.can_accept_request():
            immediate_reward = -Config.LAMBDA * 2.0
            
            # Log failed request
            if self.enable_monitoring and hasattr(self, 'queue_monitor'):
                self.queue_monitor.log_request_failed(
                    server_id=action,
                    request_id=request_id,
                    prompt=prompt,
                    current_time=self.current_time,
                    reason="Server at capacity",
                    episode=self.current_episode
                )
            
            info = self._create_info(
                request, immediate_reward, step_rewards, completed_count, False
            )
            return self.get_state(), immediate_reward + step_rewards, False, info
        
        # Valid action - compute quality and log request
        request.quality_score = self.quality_scorer.compute_quality_score(prompt, action)
        
        # Get queue state before adding request
        queue_state_before = {
            'current_load': server.get_current_load(),
            'utilization': server.get_current_load() / server.capacity,
            'pending_completions': 0,
            'avg_processing_time': server.get_stats().get('avg_processing_time', 0.0),
            'queue_requests': server.get_stats().get('processing_requests', [])
        }
        
        # Send request to server
        server.put_request(request)
        
        # Get queue state after adding request
        queue_state_after = {
            'current_load': server.get_current_load(),
            'utilization': server.get_current_load() / server.capacity,
            'pending_completions': 0,
            'avg_processing_time': server.get_stats().get('avg_processing_time', 0.0),
            'queue_requests': server.get_stats().get('processing_requests', [])
        }
        
        # Log request added event
        if self.enable_monitoring and hasattr(self, 'queue_monitor'):
            self.queue_monitor.log_request_added(
                server_id=action,
                request_id=request_id,
                prompt=prompt,
                current_time=self.current_time,
                processing_latency=None,  # Will be set when completed
                quality_score=request.quality_score,
                queue_state_before=queue_state_before,
                queue_state_after=queue_state_after,
                episode=self.current_episode
            )
        
        immediate_reward = Config.ALPHA * request.quality_score * 0.5
        info = self._create_info(
            request, immediate_reward, step_rewards, completed_count, True
        )
        
        # Add small delay to prevent too fast execution
        time.sleep(1)  

        return self.get_state(), immediate_reward + step_rewards, False, info

    def _create_info(self, request: Request, immediate_reward: float, 
                     step_rewards: float, completed_count: int, 
                     valid_action: bool) -> Dict:
        """Create info dictionary with detailed queue states"""
        server_stats = []
        for s in self.servers:
            stats = s.get_stats()
            server_stats.append({
                'current_load': stats.get('current_load', 0),
                'capacity': s.capacity,
                'utilization': stats.get('current_load', 0) / s.capacity,
                'completed_count': stats.get('completed_count', 0),
                'avg_processing_time': stats.get('avg_processing_time', 0.0),
                'processing_requests': stats.get('processing_requests', [])
            })
        
        return {
            'quality_score': request.quality_score if request.quality_score else 0.0,
            'capacity_penalty': 0.0 if valid_action else Config.LAMBDA * 2.0,
            'valid_action': valid_action,
            'completed_requests': completed_count,
            'immediate_reward': immediate_reward,
            'step_rewards': step_rewards,
            'request_id': request.id,
            'server_loads': [s.get_current_load() for s in self.servers],
            'server_utilizations': [s.get_current_load()/s.capacity for s in self.servers],
            'queue_details': server_stats,
            'current_time': self.current_time,
            'episode': self.current_episode
        }
    
    def get_environment_stats(self) -> Dict:
        """Get comprehensive environment statistics"""
        return {
            'current_time': self.current_time,
            'total_requests': self.request_counter,
            'servers': {f'server_{i}': s.get_stats() for i, s in enumerate(self.servers)}
        }
    
    def set_episode(self, episode: int):
        """Set current episode"""
        self.current_episode = episode
    
    def __del__(self):
        """Cleanup processes"""
        if hasattr(self, 'response_collector_running'):
            self.response_collector_running.value = False
        
        if hasattr(self, 'response_collector') and self.response_collector.is_alive():
            self.response_collector.join(timeout=2)
            if self.response_collector.is_alive():
                self.response_collector.terminate()
        
        if hasattr(self, 'servers'):
            for server in self.servers:
                server.stop()
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
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,
            max_num_batched_tokens=256,  
            max_model_len=128,  
            trust_remote_code=True,
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=200,
            top_p=0.95,
        )
        
        print(f"Server {server_id} ({model_name}) initialized on GPU {gpu_id}")
        
        # Process requests
        while running.value:
            try:
                request = request_queue.get(timeout=0.5)
            except:
                continue
                
            if request is None:
                continue
            
            # Update load
            processing_list = list(stats_dict.get('processing_requests', []))
            processing_list.append(request.id)
            stats_dict['processing_requests'] = processing_list
            stats_dict['current_load'] = len(processing_list)
            
            try:
                # Process request
                request.status = 'processing'
                request.start_time = time.time()
                
                # Generate response
                outputs = llm.generate(
                    prompts=[request.prompt],
                    sampling_params=sampling_params
                )
                response_text = outputs[0].outputs[0].text
                
                # Update request
                request.completion_time = time.time()
                request.processing_latency = request.completion_time - request.start_time
                request.response = {
                    "response_text": response_text,
                    "decode_time": request.processing_latency,
                    "tokens_generated": len(response_text.split())
                }
                request.status = 'completed'
                
                # Update stats
                stats_dict['completed_count'] = stats_dict.get('completed_count', 0) + 1
                stats_dict['total_processing_time'] = stats_dict.get('total_processing_time', 0.0) + request.processing_latency
                if stats_dict['completed_count'] > 0:
                    stats_dict['avg_processing_time'] = stats_dict['total_processing_time'] / stats_dict['completed_count']
                
            except Exception as e:
                print(f"Error processing request {request.id}: {e}")
                request.status = 'failed'
                request.completion_time = time.time()
            
            finally:
                # Remove from processing list
                processing_list = list(stats_dict.get('processing_requests', []))
                if request.id in processing_list:
                    processing_list.remove(request.id)
                stats_dict['processing_requests'] = processing_list
                stats_dict['current_load'] = len(processing_list)
                
                # Send response
                response_queue.put(request)
                
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
        
        # Create queues and shared data
        self.request_queue = manager.Queue()
        self.response_queue = response_queue
        self.stats = manager.dict({
            'current_load': 0,
            'completed_count': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'processing_requests': []
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
                            config_lambda: float):
    """Worker function for response collection"""
    # Import Empty exception from queue module
    from queue import Empty
    
    while running.value:
        try:
            response = response_queue.get(timeout=0.1)
        except Empty:
            # This is normal when queue is empty, just continue
            continue
        except EOFError:
            # This can happen when queue is closed
            if not running.value:
                break
            continue
        except Exception as e:
            # Only print truly unexpected errors
            if running.value and "Empty" not in str(type(e).__name__):
                print(f"Unexpected response collector error: {type(e).__name__}: {e}")
            continue
            
        if response is None:
            continue
        
        try:
            if response.status == 'completed':
                # Calculate reward
                if response.completion_time and response.arrival_time and response.quality_score:
                    latency = response.completion_time - response.arrival_time
                    quality_reward = config_alpha * response.quality_score
                    latency_penalty = config_beta * latency
                    reward = quality_reward - latency_penalty
                else:
                    reward = 0.0
                    
                collected_rewards.append(reward)
                with total_completed.get_lock():
                    total_completed.value += 1
                    
            elif response.status == 'failed':
                collected_rewards.append(-config_lambda)
        except Exception as e:
            print(f"Error processing response: {e}")

class EnhancedRouterEnvironment:
    """Environment for LLM request routing with parallel processing"""
    def __init__(self, enable_monitoring=True):
        # Initialize manager
        self.manager = mp.Manager()
        
        # Initialize components
        self.quality_scorer = QualityScorer()
        
        # Create shared response queue
        self.response_queue = self.manager.Queue()
        
        # Create server processes
        self.servers = []
        self.gpu_ids = list(range(torch.cuda.device_count()))
        
        print(f"Initializing {len(Config.MODEL_NAMES)} servers...")
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
        
        # Response collection
        self.response_collector_running = mp.Value('b', True)
        self.collected_rewards = self.manager.list()
        self.total_completed = mp.Value('i', 0)
        
        # Start response collector
        self.response_collector = mp.Process(
            target=response_collector_worker,
            args=(self.response_queue, self.collected_rewards, self.total_completed,
                  self.response_collector_running, Config.ALPHA, Config.BETA, Config.LAMBDA)
        )
        self.response_collector.daemon = True
        self.response_collector.start()
        
        # Monitoring
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            try:
                self.queue_monitor = QueueUpdateMonitor(wandb_available=False)
            except ImportError:
                print("Queue monitor not available")
                self.enable_monitoring = False
        
        # Wait a bit for servers to initialize
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
            info = self._create_info(
                request, immediate_reward, step_rewards, completed_count, False
            )
            return self.get_state(), immediate_reward + step_rewards, False, info
        
        # Valid action - compute quality and send request
        request.quality_score = self.quality_scorer.compute_quality_score(prompt, action)
        server.put_request(request)
        
        immediate_reward = Config.ALPHA * request.quality_score * 0.5
        info = self._create_info(
            request, immediate_reward, step_rewards, completed_count, True
        )
        
        return self.get_state(), immediate_reward + step_rewards, False, info
    
    def _create_info(self, request: Request, immediate_reward: float, 
                     step_rewards: float, completed_count: int, 
                     valid_action: bool) -> Dict:
        """Create info dictionary"""
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
            'server_stats': [s.get_stats() for s in self.servers]
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
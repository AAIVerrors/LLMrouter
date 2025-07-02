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
from PoissonPromptGenerator import PoissonPromptGenerator
from data_loader import AlpacaDataLoader

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
    reward: Optional[float] = 0  # Reward for this request
    episode: Optional[int] = None  # Track episode for monitoring
    

def server_worker_process(model_name: str,
                         capacity: int,
                         server_id: int,
                         request_queue,
                         response_queue,
                         running,
                         gpu_id: Optional[int] = None,
                         queue_monitor: Optional[QueueUpdateMonitor] = None):
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
        avg_processing_time = 0.0
        
        # Process requests
        while running.value:
            try:
                # Get current state with proper queue length
                current_queue_length = request_queue.qsize()    
                
                # Get queue state before processing
                queue_state_before = {
                    'current_load': current_queue_length,
                    'utilization': current_queue_length / capacity,
                    'pending_completions': 0,
                    'avg_processing_time': avg_processing_time,
                }

                # Get queue state after adding
                queue_state_added = {
                    'current_load': current_queue_length + 1,
                    'utilization': (current_queue_length + 1) / capacity,
                    'pending_completions': 0,
                    'avg_processing_time': avg_processing_time,
                }
                
                #  Get request from queue
                request = request_queue.get(timeout=0.1)
                if request is None:
                    continue

                # Start processing
                request.status = 'processing'
                request.start_time = time.time()
                
                #  Update monitoring queue, added
                queue_monitor.log_request_added(
                    server_id=request.server_id,    
                    request_id=request.id,
                    prompt=request.prompt,
                    current_time=request.start_time,
                    processing_latency=None,
                    quality_score=request.quality_score,
                    queue_state_before= queue_state_before,
                    queue_state_after= queue_state_added,
                    episode=request.episode 
                )

                try:
                    # Generate response
                    inputs = tokenizer(
                        request.prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128
                    ).to(model.device)
                    
                    # Validate input tensors
                    if inputs['input_ids'].size(0) == 0 or inputs['input_ids'].size(1) == 0:
                        raise ValueError("Empty input tensor")
                    
                    # Process request with proper validation
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            outputs = model.generate(
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                max_new_tokens=200,
                                temperature=0.7,
                                top_p=0.95,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                min_length=10,
                                use_cache=True,
                                num_return_sequences=1
                            )
                    
                    # Validate output and extract new tokens
                    if outputs is None or outputs.size(0) == 0:
                        raise ValueError("Model returned empty output")
                        
                    input_length = inputs['input_ids'].size(1)
                    if outputs.size(1) <= input_length:
                        raise ValueError("No new tokens generated")
                        
                    # Get only the new tokens
                    new_tokens = outputs[0, input_length:]
                    if len(new_tokens) == 0:
                        raise ValueError("No tokens generated after input")

                    # Decode response
                    response_text = tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    if not response_text.strip():
                        raise ValueError("Empty response after decoding")

                    # Update request with completion info
                    request.completion_time = time.time()
                    request.processing_latency = request.completion_time - request.start_time
                    request.status = 'completed'
                    request.response = {
                        "response_text": response_text,
                        "decode_time": request.processing_latency,
                        "tokens_generated": len(new_tokens)
                    }
                        
                    # Calculate average processing time
                    if request.processing_latency:
                        avg_processing_time = (
                            (avg_processing_time * (current_queue_length + 1)) + 
                            request.processing_latency
                        ) / (current_queue_length + 1)
        
                    # Get final queue state
                    queue_state_final = {
                        'current_load': current_queue_length,
                        'utilization': (current_queue_length) / capacity,
                        'pending_completions': 0,
                        'avg_processing_time': avg_processing_time,
                    }
                    
                    response_queue.put([request, queue_state_added, queue_state_final])
                    print(f"Server {server_id}: Request {request.id} completed. "
                        f"Queue: {queue_state_before['current_load']}->{queue_state_final['current_load']}")

                except Exception as e:
                    print(f"Error processing request {request.id}: {str(e)}")
                    request.status = 'failed'
                    request.completion_time = time.time()
                    request.processing_latency = request.completion_time - request.start_time
                    response_queue.put([request, queue_state_before, None])
                    continue

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
                 gpu_id: Optional[int] = None,
                 queue_monitor: Optional[QueueUpdateMonitor] = None):
        self.model_name = model_name
        self.capacity = capacity
        self.server_id = server_id
        self.gpu_id = gpu_id
        self.running = mp.Value('b', True)
        
        # Create queues and shared data
        self.request_queue = manager.Queue()
        self.response_queue = response_queue
        
        # Create shared stats with explicit locks
        self.current_queue_length = manager.Value('i', 0)
        
        # Start process
        self.process = mp.Process(
            target=server_worker_process,
            args=(model_name, capacity, server_id, 
                  self.request_queue, self.response_queue,
                  self.running, gpu_id, queue_monitor)
        )
        self.process.start()
        
    
    def put_request(self, request: Request):
        """Add request to this server's queue"""
        # Atomic update of queue length
        current_length = self.current_queue_length.value
        self.current_queue_length.value = current_length + 1
        
        self.request_queue.put(request)
    
    def stop(self):
        """Stop the server process"""
        self.running.value = False
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
    
    def can_accept_request(self) -> bool:
        """Check if server can accept new requests"""
        return self.request_queue.qsize() < self.capacity
    
    def get_current_load(self) -> int:
        """Get current number of requests being processed"""
        return self.request_queue.qsize()
    
    def get_server_id(self) -> int:
        """Get server ID"""
        return self.server_id
    

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
                              episode_completed,
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
            data = response_queue.get(timeout=0.05)
            if data is None:
                continue
            
            request = data[0]
            queue_state_before = data[1]
            queue_state_after = data[2]

            if request.status == 'completed':
                # Calculate reward
                if (request.processing_latency is not None):
                    
                    latency = request.processing_latency
                    
                    # Scale latency penalty
                    normalized_latency = latency / 10.0
                    quality_reward = config_alpha * request.quality_score
                    latency_penalty = config_beta * normalized_latency
                    reward = quality_reward - latency_penalty
                    
                    print(f"Response completed - ID: {request.id}, "
                          f"Quality: {request.quality_score:.3f}, "
                          f"Latency: {latency:.3f}s, "
                          f"Reward: {reward:.3f}")
                    
                    request.reward = reward 
                    
                    #  Update monitoring queue, completed
                    queue_monitor.log_request_completed(
                        current_time=request.completion_time,
                        server_id=request.server_id,
                        request_id=request.id,
                        queue_state_before=queue_state_before,
                        queue_state_after=queue_state_after,
                        reward=reward,
                        episode=request.episode,
                    )
                    
                    with total_completed.get_lock():
                        total_completed.value += 1
                    
                    episode_completed.put(request)
                else:
                    print(f"Warning: Incomplete response data for request {request.id}")
                
            elif request.status == 'failed':
                request.reward = -config_lambda
                print(f"Response failed - ID: {request.id}, Penalty: {-config_lambda}")
                
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
        self.response_queue = mp.Queue()
        
        # Episode completion tracking
        self.episode_completed = mp.Queue()
        
        # Create prompt queue
        self.prompt_queue = self.manager.Queue()
        self.dataloader = AlpacaDataLoader()
        PoissonPromptGenerator(
            arrival_rate=Config.POISSON_ARRIVAL_RATE,
            data_loader=self.dataloader,
            prompt_queue=self.prompt_queue,
            max_queue_size=Config.MAX_PROMPT_QUEUE_SIZE
        ).start()
        
        # Response collection
        self.response_collector_running = mp.Value('b', True)
        self.total_completed = mp.Value('i', 0)
        
        # Start response collector
        self.response_collector = mp.Process(
            target=response_collector_worker,
            args=(self.response_queue,
                  self.episode_completed,
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
                gpu_id=gpu_id,
                queue_monitor=self.queue_monitor
            )
            self.servers.append(server)
        
        # Environment state
        self.request_counter = 0
        self.current_episode = 0
        
        # Wait for initialization
        print("Waiting for servers to initialize...")
        time.sleep(5)
        
        self.reset()
        print("Environment initialization complete!")
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.request_counter = 0
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
            # utilization = load / server.capacity if server.capacity > 0 else 0
            # capacity = server.capacity
            state.extend([load])
        return np.array(state, dtype=np.float32)
    
    def get_action_mask(self) -> np.ndarray:
        """Get valid actions mask"""
        return np.array([server.can_accept_request() for server in self.servers], dtype=np.float32)


    def get_episode_data(self) -> List[Dict[str, Any]]:
        episode = []
        while True:
            try:
                data = self.episode_completed.get_nowait()
                episode.append(data.__dict__.copy())  # Use __dict__ to get a dict of fields
            except Empty:
                break  # No more completed requests in the queue
        print(episode[:5]) 
        return episode

    
    def step(self, action: int, prompt: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute routing action"""
        
        # Collect accumulated rewards
        # completed_count = self.total_completed.value
        with self.total_completed.get_lock():
            self.total_completed.value = 0
        
        # Create request
        request_id = f"req_{self.request_counter}"
        self.request_counter += 1
        
        while True:
            try:
                prompt = self.prompt_queue.get_nowait()['prompt']
                break  # Exit loop if prompt is successfully retrieved
            except Empty:
                print("Warning: Prompt queue is empty")
        
        request = Request(
            id=request_id,
            prompt=prompt,
            arrival_time=time.time(),
            server_id=action,
            episode=self.current_episode
        )
        
        # Check server availability
        server = self.servers[action]
        if not server.can_accept_request():

            # Log failed request
            if self.enable_monitoring and hasattr(self, 'queue_monitor'):
                self.queue_monitor.log_request_failed(
                    server_id=action,
                    request_id=request_id,
                    prompt=prompt,
                    current_time=time.time(),
                    reason="Server at capacity",
                    episode=self.current_episode
                )
            
            
            return self.get_state(), False
        
        # Valid action - compute quality and log request
        request.quality_score = self.quality_scorer.compute_quality_score(prompt, action)

        # Send request to server
        server.put_request(request)
            
        return self.get_state(), False

    def _create_info(self, request: Request, immediate_reward: float, 
                     step_rewards: float, completed_count: int, 
                     valid_action: bool) -> Dict:
        """Create info dictionary with detailed queue states"""
        server_stats = []
        for s in self.servers:
            server_stats.append({
                'current_load': s.get_current_load(),
                'capacity': s.capacity,
                'utilization': s.get_current_load() / s.capacity,
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
            'current_time': time.time(),
            'episode': self.current_episode
        }
    
    def get_environment_stats(self) -> Dict:
        """Get comprehensive environment statistics"""
        return {
            'current_time': time.time(),
            'total_requests': self.request_counter,
            'servers': {f'server_{i}': s.get_server_id() for i, s in enumerate(self.servers)}
        }
    
    def set_episode(self, episode: int):
        """Set current episode"""
        self.current_episode = episode
        
    def clean_response_queue(self):
        """Clean up response queue"""
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except Empty:
                break
    
    def clean_prompt_queue(self):
        """Clean up prompt queue"""
        while not self.prompt_queue.empty():
            try:
                self.prompt_queue.get_nowait()
            except Empty:
                break
    
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
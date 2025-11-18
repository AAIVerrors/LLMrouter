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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from queue import Empty
from PoissonPromptGenerator import PoissonPromptGenerator
from quality_model import P2LPredictor
import anthropic
from google import genai
from google.genai import types
from openai import OpenAI
from mistralai import Mistral

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

@dataclass
class Request:
    """Represents a single request in the system"""
    id: int
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
    price: Optional[float] = None  # Price for this request
    

def server_worker_process(model_name: str,
                         capacity: int,
                         server_id: int,
                         request_queue,
                         response_queue,
                         running,
                         gpu_id: Optional[int] = None,
                         queue_monitor: Optional[QueueUpdateMonitor] = None,
                         pause_event=None):
    """Worker function for server process"""
    try:
        # Set GPU
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)

        
        if "t5" in model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16, 
                device_map="auto",
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",
            )
        elif "gpt" in model_name or "o1" in model_name or "o3" in model_name:
            model = OpenAI()
        elif "gemini" in model_name:
            model = genai.Client(api_key=os.environ["GEMINI_API"])
        elif "claude" in model_name:
            model = anthropic.Anthropic()
        elif "mistral" in model_name or "mixtral" in model_name or "ministral" in model_name:
            api_key = os.environ["MISTRAL_API_KEY"]
            client = Mistral(api_key=api_key)
        else:
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side='left'  # Important for batch generation
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16, 
                device_map="auto",
                trust_remote_code=True,
                # attn_implementation="flash_attention_2",
            )
        
            model.eval()
            
        print(f"Server {server_id} ({model_name}) initialized on GPU {gpu_id}")
        
        # Maintain local processing queue
        avg_processing_time = 0.0
        
        # Process requests
        while running.value:
            while pause_event is not None and pause_event.is_set():
                time.sleep(0.1)
                continue
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
                request = request_queue.get_nowait()
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
                    if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
                        # if model_name == "o1-mini":
                        #     model_name = "o1-mini-2024-09-12"
                        response = model.responses.create(
                            model=model_name,
                            input= request.prompt,
                            max_output_tokens=200
                            )
                        response_text = response.output_text
                    elif "gemini" in model_name:
                        if model_name == "gemini-1.5-flash-001":
                            model_name = "gemini-1.5-flash"
                        elif model_name == "gemini-2.0-flash-exp":
                            model_name = "gemini-2.0-flash-lite"
                        elif model_name == "gemini-1.5-flash-8b-001":
                            model_name = "gemini-1.5-flash-8b"
                        message = model.models.generate_content(
                            model=model_name, contents=request.prompt,
                            config=types.GenerateContentConfig(
                                max_output_tokens=200
                            )
                        )
                        response_text = message.text
                    elif "mistral" in model_name or "mixtral" in model_name or "ministral" in model_name:
                        
                        model = model_name
                        if model_name == "mistral-7b-instruct-v0.2":
                            model = "open-mistral-7b"
                        elif model_name == "mistral-medium":
                            model = "mistral-medium-2508"
                        elif model_name == "mistral-small-24b-instruct-2501":
                            model = "mistral-small-2501"
                        elif model_name == "mixtral-8x7b-instruct-v0.1":
                            model = "open-mixtral-8x7b"
                        
                        
                        chat_response = client.chat.complete(
                                model= model,
                                max_tokens=200,
                                temperature=0.7,
                                top_p=0.95,
                                messages = [
                                    {
                                        "role": "user",
                                        "content": request.prompt,
                                    },
                                ]
                            )
                        response_text = chat_response.choices[0].message.content
                       
                    elif "claude" in model_name:
                        message = model.messages.create(
                            model=model_name,
                            max_tokens=200,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": request.prompt
                                        }
                                    ]
                                }
                            ]
                        )
                        response_text = message.content[0].text
                    else:
                    
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
                            with torch.amp.autocast('cuda', dtype=torch.float16):
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
                                    num_return_sequences=1,
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
                    request.processing_latency = max(min(request.completion_time - request.arrival_time, 60.0), 0)
                    request.status = 'completed'
                    request.response = {
                        "response_text": response_text,
                        "decode_time": request.completion_time - request.start_time,
                        "tokens_generated": len(response_text)
                    }
                    print(f"response_text: {response_text}")

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
                    request.processing_latency = max(min(request.completion_time - request.arrival_time, 60.0), 0)
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
        
        self.pause_event = mp.Event()
        self.pause_event.clear()
        
        # Create queues and shared data
        self.request_queue = mp.Queue()
        self.response_queue = response_queue
        
        # Create shared stats with explicit locks
        self.current_queue_length = manager.Value('i', 0)
        
        # Start process
        self.process = mp.Process(
            target=server_worker_process,
            args=(model_name, capacity, server_id, 
                  self.request_queue, self.response_queue,
                  self.running, gpu_id, queue_monitor, self.pause_event)
        )
        self.process.start()
        
    def clean_queue(self):
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except Empty:
                break
    
    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()   
    
    def put_request(self, request: Request):
        """Add request to this server's queue"""
        # Atomic update of queue length
        current_length = self.current_queue_length.value
        self.current_queue_length.value = current_length + 1
        
        self.request_queue.put(request)
    
    def stop(self):
        """Stop the server process"""
        self.running.value = False
        self.process.join(timeout=2)
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
    
    def get_model_name(self) -> str:
        """Get model name"""
        return self.model_name
    

class QualityScorer:
    """Quality scoring for prompt-model pairs"""
    def __init__(self):
        self.model_elo_scores = Config.MODEL_ELO_SCORES
        self.quality_model = P2LPredictor()
        
    def compute_quality_score_real(self, prompt: str, model_name: str) -> float:
        coefs = self.quality_model.get_coefficients(prompt)
        if "/" in model_name:
            model_name = model_name.split("/")[1].lower()
        # print("coefs: " + str(coefs))
        all_coefs = {m.split("/")[1].lower() if "/" in m else m: coefs.get(m.split("/")[1].lower() if "/" in m else m) for m in Config.MODEL_NAMES}
        # print("all_coefs: " + str(all_coefs))
        score = coefs.get(model_name)
        normalized_score = (score - min(all_coefs.values())) / (max(all_coefs.values()) - min(all_coefs.values()))  
        print("normalized_score" + str(normalized_score))
        return normalized_score
    
    def compute_quality_score_all(self, prompt: str) -> float:
        coefs = self.quality_model.get_coefficients(prompt)
        # print("coefs: " + str(coefs))
        all_coefs = {m: coefs.get(m.split("/")[1].lower() if "/" in m else m) for m in Config.MODEL_NAMES}
        # Normalize scores to [0, 1]
        # print("all_coefs: " + str(all_coefs))
        min_score = min(all_coefs.values())
        max_score = max(all_coefs.values())
        normalized_scores = {
            model: (score - min_score) / (max_score - min_score) if max_score > min_score else 0.0
            for model, score in all_coefs.items()
        }
        print("normalized_scores: " + str(normalized_scores))
        print("prompt: " + prompt)
        return normalized_scores
        

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
                            #   episode_completed,
                            total_completed,
                            running,
                            config_alpha: float,
                            config_beta: float,
                            config_lambda: float,
                            routed_prompts,
                            completed_prompts=None,
                            queue_monitor=None):
    """Worker function for response collection"""
    print("Response collector started")
    
    while running.value:
        try:
            data = response_queue.get_nowait()
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
                    normalized_latency = latency / 60
                    quality_reward = config_alpha * request.quality_score
                    latency_penalty = config_beta * normalized_latency
                    price = Config.REWARD_GAMMA*(Config.PRICE[request.server_id][0]*len(request.prompt)*10000 
                                                 + 10000*Config.PRICE[request.server_id][1]*len(request.response['response_text']))
                    price = max(min(price, 1.0), 0)
                    price = price 
                    reward = quality_reward - latency_penalty - price

                    print(f"Response completed - ID: {request.id}, "
                          f"Quality: {request.quality_score:.3f}, "
                          f"Latency: {latency:.3f}s, "
                          f"Reward: {reward:.3f}",
                          f"Price: {price:.3f}")
                    
                    request.reward = reward 
                    request.price = price
                    
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

                    routed_prompts[request.id] = request
                    completed_prompts.value += 1
                    # episode_completed.put(request)
                else:
                    print(f"Warning: Incomplete response data for request {request.id}")
                
            elif request.status == 'failed':
                
                request.reward = - config_beta - Config.REWARD_GAMMA
                request.price = 0.0
                routed_prompts[request.id] = request
                completed_prompts.value += 1
                # episode_completed.put(request)
                print(f"Response failed - ID: {request.id}, Penalty: {- config_beta - Config.REWARD_GAMMA}")
                
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
        
        self.routed_prompts = self.manager.dict()
        self.completed_prompts = self.manager.Value('i', 0)
        
        # Initialize components
        self.quality_scorer = QualityScorer()
        
        # Create shared response queue
        self.response_queue = mp.Queue()
        
        # Episode completion tracking
        # self.episode_completed = self.manager.Queue()
        
        # Create prompt queue and generator
        self.prompt_queue = self.manager.Queue()
        self.prompt_generator = PoissonPromptGenerator(
            arrival_rate=Config.POISSON_ARRIVAL_RATE,
            prompt_queue=self.prompt_queue,
            max_queue_size=Config.MAX_PROMPT_QUEUE_SIZE,
            dataset_name=Config.DATASET_NAME
        )
        self.prompt_generator.start()
        
        
        
        # Response collection
        self.response_collector_running = mp.Value('b', True)
        self.total_completed = mp.Value('i', 0)
        
        # Start response collector
        self.response_collector = mp.Process(
            target=response_collector_worker,
            args=(self.response_queue,
                #   self.episode_completed,
                  self.total_completed,
                  self.response_collector_running, 
                  Config.ALPHA, 
                  Config.BETA, 
                  Config.LAMBDA,
                  self.routed_prompts,
                  self.completed_prompts,
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
        time.sleep(10)
        
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
    
    def get_prompt_generator(self):
        return self.prompt_generator
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        state = []
        for server in self.servers:
            load = server.get_current_load()
            state.extend([load])
        return np.array(state, dtype=np.float32)
    
    def get_action_mask(self) -> np.ndarray:
        """Get valid actions mask"""
        return torch.tensor([server.can_accept_request() for server in self.servers], dtype=torch.float32)

    def get_episode_data(self) -> List[Dict[str, Any]]:
        episode = [self.routed_prompts[key].__dict__ for key in sorted(self.routed_prompts.keys()) if self.routed_prompts[key].episode == self.current_episode]
        print(episode[:5]) 
        return episode
    
    def clean_episode_completed(self):
        self.routed_prompts.clear()
        
    def check_get_episode_completed(self):
        print(f"Completed prompts: {self.completed_prompts.value}, Total routed: {len(self.routed_prompts)}")
        return self.completed_prompts.value == len(self.routed_prompts)
    
    def get_next_prompt(self) -> str:
        prompt = None
        try:
            prompt = self.prompt_queue.get_nowait()
        except Empty:
            # print("Warning: Prompt queue is empty")
            pass
        
        return prompt
         
    
    def step(self, action: int, prompt: str) -> Tuple[np.ndarray, bool]:
        """Execute routing action"""
        
        # Collect accumulated rewards
        with self.total_completed.get_lock():
            self.total_completed.value = 0
        
        # Create request
        request_id = self.request_counter
        self.request_counter += 1
        
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
                
            request.status = 'failed'
            request.reward = -2
            request.completion_time = time.time()
            request.processing_latency = 60
            request.quality_score = 0

            self.response_queue.put([request, None, None])
            print(f"Server {action} at capacity. Request {request_id} failed.")
            return self.get_state(), False
        
        # Valid action - compute quality and log request
        request.quality_score = self.quality_scorer.compute_quality_score_real(prompt, server.get_model_name())

        self.routed_prompts[request.id] = request

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
        
    def pause_prompt_generator(self):
        """Pause prompt generator"""
        self.prompt_generator.stop()
        
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
            
    def pause_all_servers(self):
        print("Pausing all servers...")
        for server in self.servers:
            server.pause()
        self.prompt_generator.stop()
        print("All servers paused.")

    def resume_all_servers(self):
        print("Resuming all servers...")
        for server in self.servers:
            server.resume()
        self.prompt_generator.start()
        print("All servers resumed.")

    def clean_all_queues(self):
        print("Cleaning all server queues...")
        for server in self.servers:
            server.clean_queue()
        self.clean_prompt_queue()
        self.clean_response_queue()
        self.clean_episode_completed()
        self.completed_prompts.value = 0
        print("All queues cleaned.")
    
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
        
        if hasattr(self, 'prompt_generator'):
            self.prompt_generator.stop()
import torch
import numpy as np
import time
import heapq
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from config import Config
from queueMonitor import QueueUpdateMonitor

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

class LLMServer:
    def __init__(self, model_name: str, capacity: int, server_id: int):
        self.model_name = model_name
        self.capacity = capacity
        self.server_id = server_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        self.model.to(Config.DEVICE)
        
        # Queue management
        self.processing_queue = []  # Currently processing requests
        self.completion_heap = []   # Min-heap for request completion times
        
        # Monitoring and statistics
        self.recent_request_times = []  # Track recent request arrivals
        self.completed_requests_count = 0
        self.total_processing_time = 0.0
        
        # Server-specific latency parameters
        self.base_latency_range = self._get_latency_range(model_name)
        
    def _get_latency_range(self, model_name: str) -> Tuple[float, float]:
        """Get latency range based on model type"""
        if "gpt2" in model_name.lower():
            return (0.5, 1.0)  # GPT-2 latency range
        elif "qwen" in model_name.lower():
            return (0.7, 1.2)  # Qwen latency range
        else:
            return (0.6, 1.1)  # Default range
    
    def get_current_load(self) -> int:
        """Get current number of requests being processed"""
        return len(self.processing_queue)
    
    def is_available(self) -> bool:
        """Check if server can accept new requests"""
        return self.get_current_load() < self.capacity
    
    def can_accept_request(self) -> bool:
        """Check if server is not at capacity"""
        return self.get_current_load() < self.capacity
    
    def add_request(self, request: Request, current_time: float) -> bool:
        """Add request to server if capacity allows"""
        if not self.can_accept_request():
            return False
        
        # Set start time and server assignment
        request.start_time = current_time
        request.server_id = self.server_id
        
        # Track request arrival time for monitoring
        self.recent_request_times.append(current_time)
        self.recent_request_times = [t for t in self.recent_request_times if current_time - t <= 60.0]
        
        # Calculate base latency from model range
        base_min, base_max = self.base_latency_range
        load_factor = 1.0 + (self.get_current_load() / self.capacity) * 0.5
        base_latency = np.random.uniform(base_min, base_max)
        
        # Perform real decoding and measure time
        start_decode_time = time.time()
        with torch.no_grad():
            
            inputs = self.tokenizer(
                request.prompt,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            ).to(Config.DEVICE)
            
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=200,  # Instead of max_length
                do_sample=True,      # Enable sampling since you're using temperature
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=inputs["attention_mask"]  # Add attention mask
            )
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        actual_decode_time = time.time() - start_decode_time
        
        # Store response in request object
        request.response = {
            "response_text": response_text,
            "decode_time": actual_decode_time,
            "tokens_generated": len(outputs[0])
        }
        
        # Total processing latency combines base latency, load factor and actual decode time
        processing_latency = (base_latency * load_factor) + actual_decode_time
        
        request.processing_latency = processing_latency
        request.completion_time = current_time + processing_latency
        
        # Add to processing queue and completion heap
        self.processing_queue.append(request)
        heapq.heappush(self.completion_heap, (request.completion_time, request.id))
        
        return True
    
    def update_completions(self, current_time: float) -> List[Request]:
        """Update and return completed requests"""
        completed_requests = []
        
        # Check for completed requests
        while (self.completion_heap and 
               self.completion_heap[0][0] <= current_time):
            
            completion_time, request_id = heapq.heappop(self.completion_heap)
            
            # Find and remove request from processing queue
            for i, request in enumerate(self.processing_queue):
                if request.id == request_id:
                    completed_request = self.processing_queue.pop(i)
                    completed_request.completion_time = completion_time
                    completed_requests.append(completed_request)
                    
                    # Update statistics
                    self.completed_requests_count += 1
                    if completed_request.processing_latency:
                        self.total_processing_time += completed_request.processing_latency
                    break
        
        return completed_requests
    
    def get_queue_info(self) -> Dict:
        """Get information about current queue state"""
        avg_processing_time = 0.0
        if self.completed_requests_count > 0:
            avg_processing_time = self.total_processing_time / self.completed_requests_count
        
        # Count recent requests (last 60 seconds)
        current_time = time.time()  # This should be passed in for consistency
        recent_requests = len([t for t in self.recent_request_times if current_time - t <= 60.0])
        
        return {
            'current_load': self.get_current_load(),
            'capacity': self.capacity,
            'utilization': self.get_current_load() / self.capacity,
            'queue_requests': [req.id for req in self.processing_queue],
            'pending_completions': len(self.completion_heap),
            'avg_processing_time': avg_processing_time,
            'completed_requests_total': self.completed_requests_count,
            'recent_requests_per_minute': recent_requests,
            'processing_request_details': [
                {
                    'id': req.id,
                    'start_time': req.start_time,
                    'estimated_completion': req.completion_time,
                    'processing_latency': req.processing_latency
                } for req in self.processing_queue
            ]
        }

class QualityScorer:
    def __init__(self):
        # Use sentence transformer for embedding-based quality scoring
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.model_embeddings = {}
        
        # ELO-like scores for different models (can be learned/updated)
        self.model_elo_scores = {
            0: 1200,  # GPT-2
            1: 1100,  # Qwen
        }
    
    def compute_quality_score(self, prompt: str, server_id: int) -> float:
        """Compute quality score for prompt-server pair"""
        # Get prompt embedding and characteristics
        prompt_embedding = self.encoder.encode([prompt])
        prompt_length = len(prompt.split())
        prompt_complexity = min(prompt_length / 50.0, 2.0)  # Normalized complexity
        
        # Base ELO score for the server/model
        base_elo = self.model_elo_scores.get(server_id, 1150)
        base_score = base_elo / 1000.0  # Normalize to ~1.0-1.3 range
        
        # Adjust based on prompt complexity
        # Some models might be better at complex tasks
        if server_id == 0:  # GPT-2 might be better for simple tasks
            complexity_adjustment = max(0.8, 1.2 - prompt_complexity * 0.2)
        else:  # Qwen might be better for complex tasks
            complexity_adjustment = min(1.2, 0.9 + prompt_complexity * 0.15)
        
        # Add some realistic noise
        noise = np.random.normal(0, 0.05)
        
        final_score = base_score * complexity_adjustment + noise
        return max(0.1, final_score)  # Ensure positive score

class EnhancedRouterEnvironment:
    def __init__(self, enable_monitoring=True):
        self.servers = [
            LLMServer(Config.MODEL_NAMES[i], Config.SERVER_CAPACITIES[i], i)
            for i in range(len(Config.MODEL_NAMES))
        ]
        self.quality_scorer = QualityScorer()
        
        # Environment state
        self.current_time = 0.0
        self.time_step = 0.1  # Time increment per step
        self.request_counter = 0
        self.completed_requests = []
        self.pending_rewards = []
        self.current_episode = 0  # Initialize episode tracking
        
        # Queue monitoring
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            try:
                self.queue_monitor = QueueUpdateMonitor(wandb_available=False)  # Will be set by trainer
            except ImportError:
                print("Queue update monitor not available - continuing without monitoring")
                self.enable_monitoring = False
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        self.current_time = 0.0
        self.request_counter = 0
        self.completed_requests = []
        self.pending_rewards = []
        
        # Clear all server queues
        for server in self.servers:
            server.processing_queue = []
            server.completion_heap = []
            server.recent_request_times = []
            server.completed_requests_count = 0
            server.total_processing_time = 0.0
        
        # Reset queue monitoring
        if self.enable_monitoring and hasattr(self, 'queue_monitor'):
            self.queue_monitor.reset()
        
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state (server loads and utilization)"""
        state = []
        for server in self.servers:
            # Current load (number of requests)
            current_load = server.get_current_load()
            # Utilization ratio
            utilization = current_load / server.capacity if server.capacity > 0 else 0
            state.extend([current_load, utilization])
        
        return np.array(state, dtype=np.float32)
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask for valid actions (servers not at capacity)"""
        return np.array([server.can_accept_request() for server in self.servers], 
                       dtype=np.float32)
    
    def step(self, action: int, prompt: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)"""
        # Update current time
        self.current_time += self.time_step
        
        # Update server completions and monitor them
        step_rewards = 0.0
        total_completed = 0
        
        for server in self.servers:
            # Get queue state before completion check
            queue_state_before = server.get_queue_info()
            
            completed_requests = server.update_completions(self.current_time)
            total_completed += len(completed_requests)
            
            # Monitor each completion
            if self.enable_monitoring and hasattr(self, 'queue_monitor') and completed_requests:
                queue_state_after = server.get_queue_info()
                
                for request in completed_requests:
                    if request.quality_score is not None:
                        reward = self._calculate_reward(request)
                        step_rewards += reward
                        self.completed_requests.append(request)
                        
                        # Monitor request completion
                        self.queue_monitor.log_request_completed(
                            server_id=server.server_id,
                            request_id=request.id,
                            current_time=self.current_time,
                            reward=reward,
                            queue_state_before=queue_state_before,
                            queue_state_after=queue_state_after,
                            episode=self.current_episode
                        )
            else:
                # No monitoring - just calculate rewards
                for request in completed_requests:
                    if request.quality_score is not None:
                        reward = self._calculate_reward(request)
                        step_rewards += reward
                        self.completed_requests.append(request)
        
        # Process current action (route new request)
        server = self.servers[action]
        request_id = f"req_{self.request_counter}"
        self.request_counter += 1
        
        request = Request(
            id=request_id,
            prompt=prompt,
            arrival_time=self.current_time
        )
        
        # Get queue state before routing attempt
        queue_state_before = server.get_queue_info()
        
        # Check if action is valid (server not overloaded)
        if not server.can_accept_request():
            # Invalid action - monitor the failure
            if self.enable_monitoring and hasattr(self, 'queue_monitor'):
                self.queue_monitor.log_request_failed(
                    server_id=action,
                    request_id=request_id,
                    prompt=prompt,
                    current_time=self.current_time,
                    reason="Server at capacity",
                    episode=self.current_episode
                )
            
            immediate_reward = -Config.LAMBDA * 2.0  # Heavy penalty for invalid action
            info = {
                'quality_score': 0.0,
                'latency': 0.0,
                'capacity_penalty': Config.LAMBDA * 2.0,
                'valid_action': False,
                'completed_requests': total_completed,
                'immediate_reward': immediate_reward,
                'step_rewards': step_rewards,
                'server_loads': [s.get_current_load() for s in self.servers],
                'server_utilizations': [s.get_current_load()/s.capacity for s in self.servers],
                'queue_details': [s.get_queue_info() for s in self.servers]
            }
            
            return self.get_state(), immediate_reward + step_rewards, False, info
        
        # Valid action - add request to server
        request.quality_score = self.quality_scorer.compute_quality_score(prompt, action)
        success = server.add_request(request, self.current_time)
        
        if not success:
            # This shouldn't happen if our capacity check worked
            if self.enable_monitoring and hasattr(self, 'queue_monitor'):
                self.queue_monitor.log_request_failed(
                    server_id=action,
                    request_id=request_id,
                    prompt=prompt,
                    current_time=self.current_time,
                    reason="Add request failed",
                    episode=self.current_episode
                )
            
            immediate_reward = -Config.LAMBDA
            info = {
                'quality_score': 0.0,
                'latency': 0.0,
                'capacity_penalty': Config.LAMBDA,
                'valid_action': False,
                'completed_requests': total_completed,
                'immediate_reward': immediate_reward,
                'step_rewards': step_rewards,
                'server_loads': [s.get_current_load() for s in self.servers],
                'server_utilizations': [s.get_current_load()/s.capacity for s in self.servers],
                'queue_details': [s.get_queue_info() for s in self.servers]
            }
            return self.get_state(), immediate_reward + step_rewards, False, info
        
        # Request successfully added - monitor it
        if self.enable_monitoring and hasattr(self, 'queue_monitor'):
            queue_state_after = server.get_queue_info()
            
            # Monitor request addition
            self.queue_monitor.log_request_added(
                server_id=action,
                request_id=request_id,
                prompt=prompt,
                current_time=self.current_time,
                processing_latency=request.processing_latency,
                quality_score=request.quality_score,
                queue_state_before=queue_state_before,
                queue_state_after=queue_state_after,
                episode=self.current_episode
            )
        
        # Calculate immediate reward for accepting the request (partial)
        immediate_reward = Config.ALPHA * request.quality_score * 0.5  # Partial quality reward
        
        info = {
            'quality_score': request.quality_score,
            'estimated_latency': request.processing_latency,
            'capacity_penalty': 0.0,
            'valid_action': True,
            'completed_requests': total_completed,
            'immediate_reward': immediate_reward,
            'step_rewards': step_rewards,
            'request_id': request_id,
            'server_loads': [s.get_current_load() for s in self.servers],
            'server_utilizations': [s.get_current_load()/s.capacity for s in self.servers],
            'queue_details': [s.get_queue_info() for s in self.servers]
        }
        
        # Log periodic summary
        if self.enable_monitoring and hasattr(self, 'queue_monitor'):
            self.queue_monitor.log_periodic_summary(self.current_time, self.current_episode)
        
        # Total reward is immediate reward plus rewards from completed requests
        total_reward = immediate_reward + step_rewards
        
        return self.get_state(), total_reward, False, info
    
    def _calculate_reward(self, request: Request) -> float:
        """Calculate reward for a completed request"""
        if request.completion_time is None or request.start_time is None:
            return 0.0
        
        # Calculate actual end-to-end latency
        end_to_end_latency = request.completion_time - request.arrival_time
        
        # Quality component
        quality_reward = Config.ALPHA * request.quality_score
        
        # Latency penalty
        latency_penalty = Config.BETA * end_to_end_latency
        
        # No capacity penalty for completed requests (they were valid when accepted)
        
        total_reward = quality_reward - latency_penalty
        return total_reward
    
    def get_environment_stats(self) -> Dict:
        """Get comprehensive environment statistics"""
        stats = {
            'current_time': self.current_time,
            'total_requests_created': self.request_counter,
            'total_requests_completed': len(self.completed_requests),
            'servers': {}
        }
        
        for i, server in enumerate(self.servers):
            server_info = server.get_queue_info()
            stats['servers'][f'server_{i}'] = server_info
        
        if self.completed_requests:
            latencies = [req.completion_time - req.arrival_time 
                        for req in self.completed_requests 
                        if req.completion_time and req.arrival_time]
            
            if latencies:
                stats['avg_latency'] = np.mean(latencies)
                stats['std_latency'] = np.std(latencies)
                stats['min_latency'] = np.min(latencies)
                stats['max_latency'] = np.max(latencies)
        
        return stats
    
    def set_episode(self, episode: int):
        """Set current episode number for monitoring"""
        self.current_episode = episode
        if self.enable_monitoring and hasattr(self, 'queue_monitor'):
            self.queue_monitor.current_episode = episode
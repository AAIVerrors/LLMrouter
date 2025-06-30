import numpy as np
import time
import threading
from queue import Queue, Empty as QueueEmpty
from typing import Optional, Tuple, Dict
import torch.multiprocessing as mp

class PoissonPromptGenerator:
    """Generates prompts according to a Poisson arrival process"""
    
    def __init__(self, 
                 arrival_rate: float,
                 data_loader,
                 prompt_queue,
                 max_queue_size: int = 1000):
        """
        Initialize Poisson prompt generator
        
        Args:
            arrival_rate: Average number of arrivals per second (lambda parameter)
            data_loader: Data loader instance (e.g., AlpacaDataLoader)
            prompt_queue: Shared queue to push prompts
            max_queue_size: Maximum queue size to prevent memory issues
        """
        self.arrival_rate = arrival_rate
        self.data_loader = data_loader
        self.prompt_queue = prompt_queue
        self.max_queue_size = max_queue_size
        self.running = False
        self.thread = None
        self.total_generated = 0
        self.start_time = None
        
    def _generate_prompts(self):
        """Background thread that generates prompts according to Poisson process"""
        self.start_time = time.time()
        
        while self.running:
            # Generate inter-arrival time from exponential distribution
            inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
            
            # Sleep for the inter-arrival time
            time.sleep(inter_arrival_time)
            
            if not self.running:
                break
                
            # Check queue size to prevent overflow
            if self.prompt_queue.qsize() < self.max_queue_size:
                try:
                    # Get next prompt from data loader
                    prompt = self.data_loader.get_next_prompt()
                    
                    # Create prompt entry with metadata
                    prompt_entry = {
                        'prompt': prompt,
                        'arrival_time': time.time(),
                        'id': f"prompt_{self.total_generated}"
                    }
                    
                    # Put prompt in queue
                    self.prompt_queue.put(prompt_entry)
                    self.total_generated += 1
                    
                except Exception as e:
                    print(f"Error generating prompt: {e}")
            else:
                # Queue is full, skip this arrival
                print(f"Warning: Prompt queue full ({self.prompt_queue.qsize()} items)")
    
    def start(self):
        """Start the Poisson prompt generation process"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._generate_prompts, daemon=True)
            self.thread.start()
            print(f"Started Poisson prompt generator with arrival rate {self.arrival_rate} prompts/second")
    
    def stop(self):
        """Stop the prompt generation process"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"Stopped Poisson prompt generator. Total prompts generated: {self.total_generated}")
    
    def get_stats(self) -> Dict:
        """Get generator statistics"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            actual_rate = self.total_generated / elapsed_time if elapsed_time > 0 else 0
        else:
            actual_rate = 0
            
        return {
            'total_generated': self.total_generated,
            'queue_size': self.prompt_queue.qsize(),
            'configured_rate': self.arrival_rate,
            'actual_rate': actual_rate,
            'is_running': self.running
        }
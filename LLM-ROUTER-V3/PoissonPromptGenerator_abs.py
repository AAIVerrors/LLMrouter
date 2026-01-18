import numpy as np
import time
import threading
from queue import Queue, Empty as QueueEmpty
from typing import Optional, Tuple, Dict
import torch.multiprocessing as mp
from datasets import load_dataset
import random


class PoissonPromptGenerator:
    """Generates prompts according to a Poisson arrival process"""
    
    def __init__(self, 
                 arrival_rate: float,
                 prompt_queue,
                 max_queue_size: int = 1000,
                 dataset_name: str = "tatsu-lab/alpaca"):
        """
        Initialize Poisson prompt generator
        
        Args:
            arrival_rate: Average number of arrivals per second (lambda parameter)
            prompt_queue: Shared queue to push prompts
            max_queue_size: Maximum queue size to prevent memory issues
            dataset_name: Name of the dataset to load prompts from
        """
        self.arrival_rate = arrival_rate
        self.prompt_queue = prompt_queue
        self.max_queue_size = max_queue_size
        self.running = False
        self.thread = None
        self.total_generated = 0
        self.start_time = None
        
        np.random.seed(42)
        random.seed(42)
        
        # Load dataset internally
        self.dataset = None
        self.dataset_index = 0
        self.load_dataset(dataset_name)
        
    def load_dataset(self, dataset_name: str):
        """Load and prepare the dataset"""
        try:
            # Load the Alpaca dataset
            dataset = load_dataset(dataset_name, split='train')
            self.dataset = list(dataset)
            print(f"Loaded {len(self.dataset)} samples from {dataset_name}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback to dummy data
            self._create_dummy_dataset()
    
    def _create_dummy_dataset(self):
        """Create dummy dataset as fallback"""
        templates = [
            "Explain the concept of {topic}",
            "Write a short story about {topic}",
            "What are the benefits of {topic}?",
            "How does {topic} work?",
            "Compare and contrast {topic1} and {topic2}",
            "Describe the history of {topic}",
            "What are the challenges in {topic}?",
            "How can {topic} be improved?",
            "What is the future of {topic}?",
            "Analyze the impact of {topic}",
        ]
        
        topics = [
            "machine learning", "artificial intelligence", "quantum computing",
            "renewable energy", "space exploration", "biotechnology",
            "climate change", "cryptocurrency", "virtual reality",
            "robotics", "cybersecurity", "nanotechnology", "blockchain",
            "neural networks", "data science", "cloud computing"
        ]
        
        dummy_data = []
        for i in range(1000):  # Generate 1000 dummy samples
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            topic2 = topics[(i + 1) % len(topics)]
            
            instruction = template.format(topic=topic, topic1=topic, topic2=topic2)
            
            dummy_data.append({
                'instruction': instruction,
                'input': '',
                'output': f'Sample response for: {instruction}'
            })
        
        self.dataset = dummy_data
        print(f"Created {len(self.dataset)} dummy samples")
    
    def get_next_prompt(self) -> Tuple[str, str]:
        """Get next prompt from dataset"""
        if not self.dataset:
            return "Default prompt: Explain artificial intelligence."
        
        # Get current sample and advance index
        sample = self.dataset[self.dataset_index]
        self.dataset_index = (self.dataset_index + 1) % len(self.dataset)
        
        # Convert to prompt format
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"
        ground_truth = sample.get('output', '')

        return prompt, ground_truth
    def _generate_prompts(self):
        """Background thread that generates prompts according to Poisson process"""
        self.start_time = time.time()
        
        while self.running:
            # Generate inter-arrival time from exponential distribution
            inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate)
            # inter_arrival_time = 0.01
            
            # Sleep for the inter-arrival time
            time.sleep(inter_arrival_time)
            
            if not self.running:
                break
                
            # Check queue size to prevent overflow
            if self.prompt_queue.qsize() < self.max_queue_size:
                try:
                    # Get next prompt from dataset
                    prompt, ground_truth = self.get_next_prompt()
                    
                    # Create prompt entry with metadata
                    prompt_entry = {
                        'prompt': prompt,
                        'ground_truth': ground_truth,
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
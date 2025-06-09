import random
from datasets import load_dataset
from typing import List, Dict, Iterator
from config import Config

class AlpacaDataLoader:
    def __init__(self):
        self.dataset = None
        self.current_index = 0
        self.load_dataset()
    
    def load_dataset(self):
        """Load the Alpaca dataset"""
        try:
            # Load the Alpaca dataset
            dataset = load_dataset(Config.DATASET_NAME, split='train')
            
            # Convert to list and shuffle
            self.dataset = list(dataset)
            random.shuffle(self.dataset)
            
            # Limit dataset size for testing
            if len(self.dataset) > Config.MAX_SAMPLES:
                self.dataset = self.dataset[:Config.MAX_SAMPLES]
            
            print(f"Loaded {len(self.dataset)} samples from {Config.DATASET_NAME}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback to dummy data
            self.dataset = self._create_dummy_dataset()
    
    def _create_dummy_dataset(self) -> List[Dict]:
        """Create dummy dataset for testing"""
        dummy_data = []
        templates = [
            "Explain the concept of {topic}",
            "Write a short story about {topic}",
            "What are the benefits of {topic}?",
            "How does {topic} work?",
            "Compare and contrast {topic1} and {topic2}",
            "Provide step-by-step instructions for {topic}",
            "What are the challenges related to {topic}?",
            "Describe the history of {topic}",
        ]
        
        topics = [
            "machine learning", "artificial intelligence", "quantum computing",
            "renewable energy", "space exploration", "biotechnology",
            "climate change", "cryptocurrency", "virtual reality",
            "robotics", "cybersecurity", "nanotechnology"
        ]
        
        for i in range(Config.MAX_SAMPLES):
            template = random.choice(templates)
            if "{topic1}" in template and "{topic2}" in template:
                topic1, topic2 = random.sample(topics, 2)
                instruction = template.format(topic1=topic1, topic2=topic2)
            else:
                topic = random.choice(topics)
                instruction = template.format(topic=topic)
            
            dummy_data.append({
                'instruction': instruction,
                'input': '',
                'output': f'This is a sample response for: {instruction}'
            })
        
        return dummy_data
    
    def get_prompt(self, sample: Dict) -> str:
        """Convert dataset sample to prompt string"""
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"
        
        return prompt
    
    def get_next_batch(self, batch_size: int) -> List[str]:
        """Get next batch of prompts"""
        batch = []
        for _ in range(batch_size):
            if self.current_index >= len(self.dataset):
                # Reset and shuffle when we reach the end
                self.current_index = 0
                random.shuffle(self.dataset)
            
            sample = self.dataset[self.current_index]
            prompt = self.get_prompt(sample)
            batch.append(prompt)
            self.current_index += 1
        
        return batch
    
    def get_random_prompt(self) -> str:
        """Get a single random prompt"""
        sample = random.choice(self.dataset)
        return self.get_prompt(sample)
    
    def reset(self):
        """Reset the data loader"""
        self.current_index = 0
        random.shuffle(self.dataset)
    
    def __len__(self):
        return len(self.dataset) if self.dataset else 0
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        if not self.dataset:
            return {}
        
        total_samples = len(self.dataset)
        avg_instruction_length = sum(len(sample['instruction'].split()) 
                                   for sample in self.dataset) / total_samples
        
        return {
            'total_samples': total_samples,
            'avg_instruction_length': avg_instruction_length,
            'current_index': self.current_index
        }

class EpisodeBuffer:
    """Buffer to store episode trajectories"""
    def __init__(self):
        self.trajectories = []
        self.current_episode = []
    
    def add_step(self, state, prompt, action, log_prob, value, reward, action_mask=None):
        """Add a step to the current episode"""
        step_data = {
            'state': state,
            'prompt': prompt,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'action_mask': action_mask
        }
        self.current_episode.append(step_data)
    
    def finish_episode(self):
        """Finish current episode and add to trajectories"""
        if self.current_episode:
            self.trajectories.extend(self.current_episode)
            self.current_episode = []
    
    def get_trajectories(self):
        """Get all trajectories and clear buffer"""
        trajectories = self.trajectories.copy()
        self.clear()
        return trajectories
    
    def clear(self):
        """Clear all trajectories"""
        self.trajectories = []
        self.current_episode = []
    
    def size(self):
        """Get total number of steps stored"""
        return len(self.trajectories) + len(self.current_episode)
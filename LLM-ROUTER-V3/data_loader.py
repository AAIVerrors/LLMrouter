import random
from datasets import load_dataset
from typing import List, Dict, Iterator
from config import Config

class AlpacaDataLoader:
    def __init__(self):
        self.dataset = None
        self.current_episode = 0
        self.current_step = 0
        self.partitioned_data = []
        self.load_dataset()
    
    def load_dataset(self):
        """Load and partition the Alpaca dataset"""
        try:
            # Load the Alpaca dataset
            dataset = load_dataset(Config.DATASET_NAME, split='train')
            
            # Convert to list
            self.dataset = list(dataset)
            
            # Calculate total samples needed
            total_samples_needed = Config.EPISODE_LENGTH * Config.MAX_EPISODES
            
            # If dataset is smaller than needed, repeat it
            if len(self.dataset) < total_samples_needed:
                repetitions = (total_samples_needed + len(self.dataset) - 1) // len(self.dataset)
                self.dataset = self.dataset * repetitions
            
            # Trim to exact size needed
            self.dataset = self.dataset[:total_samples_needed]
            
            # Partition the dataset
            self.partitioned_data = []
            for i in range(Config.MAX_EPISODES):
                start_idx = i * Config.EPISODE_LENGTH
                end_idx = start_idx + Config.EPISODE_LENGTH
                episode_data = self.dataset[start_idx:end_idx]
                self.partitioned_data.append(episode_data)
            
            print(f"Loaded and partitioned {len(self.dataset)} samples into "
                  f"{len(self.partitioned_data)} episodes of {Config.EPISODE_LENGTH} steps each")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            # Fallback to dummy data
            self._create_dummy_dataset()
    
    def _create_dummy_dataset(self):
        """Create partitioned dummy dataset"""
        total_samples = Config.EPISODE_LENGTH * Config.MAX_EPISODES
        dummy_data = []
        templates = [
            "Explain the concept of {topic}",
            "Write a short story about {topic}",
            "What are the benefits of {topic}?",
            "How does {topic} work?",
            "Compare and contrast {topic1} and {topic2}",
        ]
        
        topics = [
            "machine learning", "artificial intelligence", "quantum computing",
            "renewable energy", "space exploration", "biotechnology",
            "climate change", "cryptocurrency", "virtual reality",
            "robotics", "cybersecurity", "nanotechnology"
        ]
        
        # Generate deterministic dummy data
        for i in range(total_samples):
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            instruction = template.format(topic=topic, topic1=topics[i % len(topics)], 
                                       topic2=topics[(i + 1) % len(topics)])
            
            dummy_data.append({
                'instruction': instruction,
                'input': '',
                'output': f'Sample response {i}: {instruction}'
            })
        
        # Partition dummy data
        self.dataset = dummy_data
        self.partitioned_data = []
        for i in range(Config.MAX_EPISODES):
            start_idx = i * Config.EPISODE_LENGTH
            end_idx = start_idx + Config.EPISODE_LENGTH
            self.partitioned_data.append(dummy_data[start_idx:end_idx])
    
    def get_next_prompt(self) -> str:
        """Get next prompt in sequence"""
        if not self.partitioned_data:
            raise RuntimeError("Dataset not properly initialized")
        
        # Get current episode's data
        episode_data = self.partitioned_data[self.current_episode]
        sample = episode_data[self.current_step]
        prompt = self.get_prompt(sample)
        
        # Update step counter
        self.current_step += 1
        if self.current_step >= Config.EPISODE_LENGTH:
            self.current_step = 0
            self.current_episode = (self.current_episode + 1) % Config.MAX_EPISODES
        
        return prompt
    
    def reset_episode(self, episode_num: int):
        """Reset to start of specified episode"""
        if episode_num < 0 or episode_num >= Config.MAX_EPISODES:
            raise ValueError(f"Invalid episode number: {episode_num}")
        
        self.current_episode = episode_num
        self.current_step = 0
    
    def get_prompt(self, sample: Dict) -> str:
        """Convert dataset sample to prompt string"""
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"
        
        return prompt
    
    def get_episode_data(self, episode_num: int) -> List[Dict]:
        """Get all samples for a specific episode"""
        if episode_num < 0 or episode_num >= Config.MAX_EPISODES:
            raise ValueError(f"Invalid episode number: {episode_num}")
        return self.partitioned_data[episode_num]

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
            self.trajectories.append(self.current_episode)
            self.current_episode = []
            
    def get_current_episode(self) -> List[Dict]:
        """Get current episode data without clearing it"""
        return self.current_episode.copy()
    
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
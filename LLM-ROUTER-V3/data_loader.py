import random
from typing import List, Dict, Iterator
from config import Config

class EpisodeBuffer:
    """Buffer to store episode trajectories"""
    def __init__(self):
        self.trajectories = []
        self.current_episode = []
    
    def add_step(self, time_slot, route_time, state, prompt, action, log_prob, value, reward, action_mask=None, service_rate=None, queue_length=None):
        """Add a step to the current episode"""
        step_data = {
            'time_slot': time_slot,
            'route_time': route_time,
            'state': state,
            'prompt': prompt,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'action_mask': action_mask,
            'service_rate': service_rate,
            'queue_length': queue_length
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
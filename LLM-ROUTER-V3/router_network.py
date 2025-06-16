import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config
import torch.multiprocessing as mp

class RouterNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(RouterNetwork, self).__init__()
        
        # Prompt encoder
        self.prompt_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.prompt_encoder.eval()
        
        # Freeze prompt encoder parameters
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        
        # Input dimensions
        prompt_dim = 384 
        total_input_dim = prompt_dim + state_dim
    
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(total_input_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM // 2, 1)
        )
        
        self.action_dim = action_dim
        
    def encode_prompt(self, prompts):
        """Encode prompts using sentence transformer"""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        with torch.no_grad():
            embeddings = self.prompt_encoder.encode(prompts, convert_to_tensor=True)
        
        return embeddings.to(Config.DEVICE)
    
    def forward(self, state, prompt, action_mask=None):
        """
        Forward pass
        Args:
            state: Server loads [batch_size, state_dim]
            prompt: Text prompts (list of strings or single string)
            action_mask: Valid action mask [batch_size, action_dim]
        """
        # Encode prompt
        prompt_embedding = self.encode_prompt(prompt)
        
        # Ensure proper dimensions
        if len(prompt_embedding.shape) == 1:
            prompt_embedding = prompt_embedding.unsqueeze(0)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Concatenate state and prompt embedding
        combined_input = torch.cat([state, prompt_embedding], dim=-1)
        
        # Shared forward pass
        shared_output = self.shared_layers(combined_input)
        
        # Actor output (logits)
        logits = self.actor(shared_output)
        
        # Apply action mask if provided
        if action_mask is not None:
            logits = logits + (action_mask - 1) * 1e8  # Mask invalid actions
        
        # Value output
        value = self.critic(shared_output)
        
        return logits, value
    
    def get_action_and_value(self, state, prompt, action_mask=None, action=None):
        """Get action and value for given state and prompt"""
        logits, value = self.forward(state, prompt, action_mask)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.network = RouterNetwork(state_dim, action_dim).to(Config.DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=Config.LEARNING_RATE)
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        
    def get_action(self, state, prompt, action_mask=None):
        """Get action for given state and prompt"""
        state_tensor = torch.FloatTensor(state).to(Config.DEVICE)
        
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).to(Config.DEVICE)
        else:
            action_mask_tensor = None
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action_and_value(
                state_tensor, prompt, action_mask_tensor
            )
        
        return action.cpu().item(), log_prob.cpu().item(), value.cpu().item()
    
    def update(self, trajectories):
        """Update network using PPO algorithm"""
        # Prepare batch data (convert to numpy first to avoid warning)
        states_np = np.array([t['state'] for t in trajectories])
        actions_np = np.array([t['action'] for t in trajectories])
        old_log_probs_np = np.array([t['log_prob'] for t in trajectories])
        rewards_np = np.array([t['reward'] for t in trajectories])
        values_np = np.array([t['value'] for t in trajectories])
        
        states = torch.FloatTensor(states_np).to(Config.DEVICE)
        actions = torch.LongTensor(actions_np).to(Config.DEVICE)
        old_log_probs = torch.FloatTensor(old_log_probs_np).to(Config.DEVICE)
        rewards = torch.FloatTensor(rewards_np).to(Config.DEVICE)
        values = torch.FloatTensor(values_np).to(Config.DEVICE)
        prompts = [t['prompt'] for t in trajectories]
        action_masks = None
        if 'action_mask' in trajectories[0] and trajectories[0]['action_mask'] is not None:
            action_masks_np = np.array([t['action_mask'] for t in trajectories])
            action_masks = torch.FloatTensor(action_masks_np).to(Config.DEVICE)
        
        # Compute advantages and returns
        advantages = self.compute_gae(rewards, values)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(Config.PPO_EPOCHS):
            # Get current policy outputs
            _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                states, prompts, action_masks, actions
            )
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - Config.CLIP_EPSILON, 1 + Config.CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + Config.VALUE_COEF * value_loss + Config.ENTROPY_COEF * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), Config.MAX_GRAD_NORM)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        return {
            'policy_loss': total_policy_loss / Config.PPO_EPOCHS,
            'value_loss': total_value_loss / Config.PPO_EPOCHS,
            'entropy_loss': total_entropy_loss / Config.PPO_EPOCHS
        }
    
    def compute_gae(self, rewards, values, dones=None):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # If no dones provided, assume all episodes continue except the last
        if dones is None:
            dones = torch.zeros_like(rewards)
            dones[-1] = 1
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + Config.GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + Config.GAMMA * Config.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
    
        # Shared layers with improved initialization
        self.shared_layers = nn.Sequential(
            nn.Linear(total_input_dim, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.LayerNorm(Config.HIDDEN_DIM),  # Add layer normalization
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head (policy) with better initialization
        self.actor = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(Config.HIDDEN_DIM // 2, action_dim)
        )
        
        # Critic head (value function) with better initialization
        self.critic = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(Config.HIDDEN_DIM // 2, 1)
        )
        
        self.action_dim = action_dim
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights to prevent gradient explosions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
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
        logits =  self.actor(shared_output)
        
        print(state)
        print(action_mask)
        
        # Apply action mask if provided
      
        # Apply action mask BEFORE softmax if provided
        if action_mask is not None:
            # FIXED: Use action_mask directly (not 1 - action_mask)
            # Where mask=0 (invalid), add large negative value
            # Where mask=1 (valid), add 0 (no change)
            mask_value = -1e9
            logits = logits + (action_mask == 0).float() * mask_value
            
        logits = F.softmax(logits, dim=-1)
        
        # Value output
        value = self.critic(shared_output)
        # print(action_mask)
        # print(logits)
        
        return logits, value
    
    def get_action_and_value(self, state, prompt, action_mask=None, action=None):
        """Get action and value for given state and prompt"""
        logits, value = self.forward(state, prompt, action_mask)
        probs = logits.clone()
        dist = Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value, logits

    def get_queue_scores(self, state, service_rate, factor=Config.QUEUE_SCORE_FACTOR, epslon=Config.QUEUE_EPSLONG):
        """
        Get queue scores for each action based on current state.
        This is used to compute the log probabilities of actions.
        """
        scores = []
        
        for i,element in enumerate(state):
            load = element  
            capacity = Config.SERVER_CAPACITIES[i]
            utilization = load / capacity
            
            # Compute score based on load and service rate
            score = (1-factor)*(1-utilization)*(service_rate[i]/(load+epslon))\
                    + (1-factor)*utilization*(1-(load/capacity)) \
                    + factor*(service_rate[i]/max(service_rate)) 
            scores.append(score)

        return torch.FloatTensor(scores).to(Config.DEVICE)

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.network = RouterNetwork(state_dim, action_dim).to(Config.DEVICE)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=Config.LEARNING_RATE)
        
        self.state_dim = state_dim
        self.action_dim = action_dim


    def get_action(self, state, prompt, action_mask=None, alpha=Config.MERGE_ALPHA, service_rate=[1]*len(Config.SERVER_CAPACITIES)):
        """Get action for given state and prompt"""
        alpha=Config.MERGE_ALPHA
        state_tensor = torch.FloatTensor(state).to(Config.DEVICE)
        
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).to(Config.DEVICE)
        else:
            action_mask_tensor = None
        
        with torch.no_grad():
            action, log_prob, entropy, value, dist_policy = self.network.get_action_and_value(
                state_tensor, prompt, action_mask_tensor
            )
        print(dist_policy)
        print(action)
            
        # Get queue scores and convert to log-probabilities
        queue_scores = self.network.get_queue_scores(state, service_rate=service_rate)
        queue_probs = torch.softmax(queue_scores, dim=0)
        queue_log_probs = torch.log(queue_probs + 1e-8)

        # Merge log-probs for all actions
        merged_log_probs =  (queue_probs * alpha) +  (dist_policy * (1 - alpha))
        # print(f"queue_log_probs: {queue_log_probs}, log_prob: {log_prob}")
        
        merged_probs = torch.softmax(merged_log_probs, dim=-1)
        
        # Sample action from merged distribution
        dist = Categorical(merged_log_probs)
        action = dist.sample()
        merged_log_prob = dist.log_prob(action)
        # print(f"Action: {action}, Merged Log Prob: {merged_log_prob}, dist: {dist.__dict__}")
        print(merged_log_probs)
        print(action)
        return action.cpu().item(), merged_log_prob.cpu().item(), value.cpu().item()

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
            _, new_log_probs, entropy, new_values, dist = self.network.get_action_and_value(
                states, prompts, action_masks, actions
            )
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - Config.CLIP_EPSILON, 1 + Config.CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).sum()
            
            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns, reduction='sum')
            
            # Entropy loss
            entropy_loss = -entropy.sum()
            
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
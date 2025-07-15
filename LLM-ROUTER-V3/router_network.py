import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config
import torch.multiprocessing as mp
from environment import QualityScorer

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
        self.state_dim = state_dim  # 3 * num_servers
        num_servers = state_dim // 3
        self.num_servers = num_servers
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, state_dim, Config.HIDDEN_DIM // 4) / 10
        )
        
        # Initial projection to match hidden dimension
        self.input_projection = nn.Linear(1, Config.HIDDEN_DIM // 4)
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=Config.HIDDEN_DIM // 4,
            num_heads=Config.ATTENTION_HEADS,
            dropout=0.1
        )
        
        # Shared layers with residual connections
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear((Config.HIDDEN_DIM // 4) + prompt_dim, Config.HIDDEN_DIM),
                nn.LayerNorm(Config.HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
                nn.LayerNorm(Config.HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        ])
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(Config.HIDDEN_DIM // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM // 2),
            nn.LayerNorm(Config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(Config.HIDDEN_DIM // 2, 1)
        )
        
        self.action_dim = action_dim
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights to prevent gradient explosions"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    # def encode_prompt(self, prompts):
    #     """Encode prompts using sentence transformer"""
    #     if isinstance(prompts, str):
    #         prompts = [prompts]
        
    #     with torch.no_grad():
    #         embeddings = self.prompt_encoder.encode(prompts, convert_to_tensor=True)
        
    #     return embeddings.to(Config.DEVICE)
    
    def encode_prompt(self, prompts):
        """Encode prompts using sentence transformer"""
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, (float, int)):
            prompts = [str(prompts)]
        elif isinstance(prompts, torch.Tensor):
            if prompts.dim() == 0:  # scalar tensor
                prompts = [str(prompts.item())]
            else:
                prompts = [str(p.item()) if p.dim() == 0 else str(p) for p in prompts]
        elif isinstance(prompts, list):
            # Convert all elements to string, handling various types
            converted_prompts = []
            for p in prompts:
                if isinstance(p, str):
                    converted_prompts.append(p)
                elif isinstance(p, torch.Tensor) and p.dim() == 0:
                    converted_prompts.append(str(p.item()))
                else:
                    converted_prompts.append(str(p))
            prompts = converted_prompts
        else:
            # Fallback for any other type
            prompts = [str(prompts)]
        
        with torch.no_grad():
            embeddings = self.prompt_encoder.encode(prompts, convert_to_tensor=True)
        
        return embeddings.to(Config.DEVICE)
    
    def forward(self, state, prompt, action_mask=None):
        """
        Forward pass
        Args:
            state: Server loads and rates [batch_size, 3 * num_servers]
            prompt: Text prompts (list of strings or single string)
            action_mask: Valid action mask [batch_size, action_dim]
        Returns:
            logits: Policy probabilities [batch_size, action_dim]
            value: Value estimate [batch_size, 1]
        """
        print('state:', state)
        batch_size = state.shape[0] if len(state.shape) > 1 else 1
        state = state.view(batch_size, self.state_dim, 1)  # [batch_size, state_dim, 1]
        
        # Encode prompt
        prompt_embedding = self.encode_prompt(prompt)
        if len(prompt_embedding.shape) == 1:
            prompt_embedding = prompt_embedding.unsqueeze(0)
        prompt_embedding = prompt_embedding.expand(batch_size, -1)
        
        # Project state and add positional encoding
        state_features = self.input_projection(state)  # [batch_size, state_dim, hidden_dim/4]
        state_features = state_features + self.positional_encoding
        
        # Apply attention
        state_features = state_features.permute(1, 0, 2)  # [state_dim, batch_size, hidden_dim/4]
        attn_output, _ = self.attention(state_features, state_features, state_features)
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, state_dim, hidden_dim/4]
        attn_output = attn_output.mean(dim=1)  # [batch_size, hidden_dim/4]
        
        # Combine with prompt embedding
        combined = torch.cat([attn_output, prompt_embedding], dim=-1)
        
        # Shared layers with residual connections
        x = self.shared_layers[0](combined)
        residual = x
        x = self.shared_layers[1](x)
        x = x + residual  # Residual connection
        
        # Actor output (logits)
        logits = self.actor(x)
        
        # Apply action mask BEFORE softmax
        if action_mask is not None:
            mask_value = -1e9
            logits = logits + (action_mask == 0).float() * mask_value
            
        logits = F.softmax(logits, dim=-1)
        
        # Value output
        value = self.critic(x)
        
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

    def get_action_and_value_queue(self, state, state_np, prompt, action_mask=None, action=None, service_rate=None):
        """
        Get action and value for given state and prompt, with queue scores.
        This is used to compute the log probabilities of actions.
        """
        logits, value = self.forward(state, prompt, action_mask)
        probs = logits.clone()
        dist = Categorical(probs)
        
        queue_scores = self.get_queue_scores_batch(state_np, service_rate=service_rate)
        queue_probs = torch.softmax(queue_scores, dim=0)
        queue_log_probs = torch.log(queue_probs + 1e-8)
        print("queue probs, logits")
        print(queue_probs)
        print(logits)

        # Merge log-probs for all actions
        merged_logits =  (queue_probs * Config.ALPHA) +  (logits * (1 - Config.ALPHA))

        merged_dist = Categorical(merged_logits)
        if action is None:
            action = merged_dist.sample()

        log_prob = merged_dist.log_prob(action)
        entropy = merged_dist.entropy()

        return action, log_prob, entropy, value, merged_logits
    
    def get_queue_scores_batch(self, state, service_rate, factor=Config.QUEUE_SCORE_FACTOR, epslon=Config.QUEUE_EPSLONG):
        """
        Get queue scores for each action based on current state.
        Handles both single and batch mode.
        """
        # If state is a batch (2D array or list of lists), process each sample
        if isinstance(state, np.ndarray) and state.ndim == 2:
            batch_scores = []
            for s, sr in zip(state, service_rate):
                batch_scores.append(self.get_queue_scores(s, sr, factor, epslon))
            return torch.stack(batch_scores)
        if isinstance(state, torch.Tensor) and state.dim() == 2:
            batch_scores = []
            for s, sr in zip(state, service_rate):
                batch_scores.append(self.get_queue_scores(s.cpu().numpy(), sr, factor, epslon))
            return torch.stack(batch_scores)
        # If state is a numpy array, convert to list
        if isinstance(state, np.ndarray):
            state = state.tolist()
        scores = []
        for i, capacity in enumerate(Config.SERVER_CAPACITIES):
            utilization = float(state[i])
            load = utilization * capacity
            # Defensive: avoid division by zero
            if i <= 0:
                service_rate[i] = 0.0001
            score = (1-factor)*(1-utilization)*(service_rate[i]/(load+epslon))\
                    + (1-factor)*utilization*(1-(load/capacity)) \
                    + factor*(service_rate[i]/max(service_rate))
            scores.append(score)
        return torch.FloatTensor(scores).to(Config.DEVICE)

    def get_queue_scores(self, state, service_rate, factor=Config.QUEUE_SCORE_FACTOR, epslon=Config.QUEUE_EPSLONG):
        """
        Get queue scores for each action based on current state.
        This is used to compute the log probabilities of actions.
        """
        scores = []
        
        for i,element in enumerate(Config.SERVER_CAPACITIES):
            utilization = state[i] 
            capacity = Config.SERVER_CAPACITIES[i]
            load = utilization * capacity
            
            for index, ele in enumerate(service_rate):
                if index <= 0:
                    service_rate[index] = 0.0001
            
            # Compute score based on load and service rate
            score = (1-factor)*(1-utilization)*(service_rate[i]/(load+epslon))\
                    + (1-factor)*utilization*(1-(load/capacity)) \
                    + factor*(service_rate[i]/max(service_rate)) 
            # score = service_rate[i]/(load+epslon)
            scores.append(score)

        return torch.FloatTensor(scores).to(Config.DEVICE)

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int):
        self.network = RouterNetwork(state_dim, action_dim).to(Config.DEVICE)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=Config.LEARNING_RATE)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.quality_scorer = QualityScorer()


    def get_action(self, state, prompt, action_mask=None, alpha=Config.MERGE_ALPHA, service_rate=[1]*len(Config.SERVER_CAPACITIES), round_robin_counter=0):
        """Get action for given state and prompt"""
        alpha=Config.MERGE_ALPHA
        
        state_tensor = torch.FloatTensor(state).to(Config.DEVICE)
        
        if action_mask is not None:
            action_mask_tensor = torch.FloatTensor(action_mask).to(Config.DEVICE)
        else:
            action_mask_tensor = None
        
        # if Config.ROUND_ROBIN:
        #     action = round_robin_counter % self.action_dim
        #     count = 0
        #     while action_mask_tensor is not None and action_mask_tensor[action] == 0:
        #         round_robin_counter += 1
        #         action = round_robin_counter % self.action_dim
        #         count += 1
        #         if count > self.action_dim:
        #             print("All actions are masked, returning random action")
        #             action = torch.randint(0, self.action_dim, (1,)).item()
        #             break
        #     log_prob = torch.tensor(0.0).to(Config.DEVICE)
        #     entropy = torch.tensor(0.0).to(Config.DEVICE)
        #     value = torch.tensor(0.0).to(Config.DEVICE)
        #     dist_policy = torch.zeros(self.action_dim).to(Config.DEVICE)
        #     dist_policy[action] = 1.0  # Set probability for the chosen action
        #     print(f"Round Robin Action: {action}")
        #     round_robin_counter += 1
        #     return action, log_prob.cpu().item(), value.cpu().item(), round_robin_counter
        
        pre_len = 2*len(Config.MODEL_NAMES)
        coefs = self.quality_scorer.compute_quality_score_all(prompt)
        for index,server in enumerate(Config.MODEL_NAMES):
            state_tensor[pre_len + index] = float(coefs[server])
            print(f"Quality score for {server}: {coefs[server]}")
        print(state_tensor)

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
        
        if Config.ROUND_ROBIN:
            action = round_robin_counter % self.action_dim
            count = 0
            while action_mask_tensor is not None and action_mask_tensor[action] == 0:
                round_robin_counter += 1
                action = round_robin_counter % self.action_dim
                count += 1
                if count > self.action_dim:
                    print("All actions are masked, returning random action")
                    action = torch.randint(0, self.action_dim, (1,)).item()
                    break
            log_prob = torch.tensor(0.0).to(Config.DEVICE)
            entropy = torch.tensor(0.0).to(Config.DEVICE)
            value = torch.tensor(0.0).to(Config.DEVICE)
            dist_policy = torch.zeros(self.action_dim).to(Config.DEVICE)
            dist_policy[action] = 1.0  # Set probability for the chosen action
            print(f"Round Robin Action: {action}")
            round_robin_counter += 1
            return action, log_prob.cpu().item(), value.cpu().item(), round_robin_counter
        return action.cpu().item(), merged_log_prob.cpu().item(), value.cpu().item(), round_robin_counter

        
    def update(self, trajectories):
        """Update network using PPO algorithm"""
        # Prepare batch data (convert to numpy first to avoid warning)
        states_np = np.array([t['state'] for t in trajectories])
        actions_np = np.array([t['action'] for t in trajectories])
        old_log_probs_np = np.array([t['log_prob'] for t in trajectories])
        rewards_np = np.array([t['reward'] for t in trajectories])
        values_np = np.array([t['value'] for t in trajectories])
        service_rate = [t['service_rate'] for t in trajectories]
        
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
            if Config.USE_MERGE_TO_TRAIN:

                _, new_log_probs, entropy, new_values, dist = self.network.get_action_and_value_queue(
                    states,          # state (tensor)
                    states_np,       # state_np (numpy array)
                    prompts,         # prompt (list of strings)
                    action_masks,    # action_mask (tensor or None)
                    actions,         # action (tensor)
                    service_rate=service_rate  # service_rate (list)
                )

            else:
                _, new_log_probs, entropy, new_values, dist = self.network.get_action_and_value(
                    states, prompts, action_masks, actions
                )
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - Config.CLIP_EPSILON, 1 + Config.CLIP_EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns, reduction='mean')
            
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
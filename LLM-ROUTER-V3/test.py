#!/usr/bin/env python3
"""
Test script for the enhanced LLM router system.
This script demonstrates the key features and validates the implementation.
"""

import numpy as np
import torch
from environment import RouterEnvironment, Request
from router_network import PPOAgent
from data_loader import AlpacaDataLoader
from config import Config

def test_environment_basic():
    """Test basic environment functionality"""
    print("Testing Enhanced Router Environment...")
    print("=" * 50)
    
    env = RouterEnvironment()
    data_loader = AlpacaDataLoader()
    
    # Test initial state
    initial_state = env.reset()
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Initial state: {initial_state}")
    print(f"State contains: load and utilization for {len(Config.SERVER_CAPACITIES)} servers")
    
    # Test action mask
    action_mask = env.get_action_mask()
    print(f"Initial action mask: {action_mask}")
    print("All servers should be available initially")
    
    # Test a few steps
    print("\nTesting environment steps...")
    for step in range(5):
        prompt = data_loader.get_random_prompt()
        print(f"\nStep {step + 1}:")
        print(f"  Prompt: {prompt[:100]}...")
        
        # Choose a valid action
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            print(f"  Chosen action (server): {action}")
            
            state, reward, done, info = env.step(action, prompt)
            print(f"  Reward: {reward:.3f}")
            print(f"  Valid action: {info['valid_action']}")
            print(f"  Quality score: {info.get('quality_score', 0):.3f}")
            print(f"  Estimated latency: {info.get('estimated_latency', 0):.3f}")
            print(f"  Completed requests: {info.get('completed_requests', 0)}")
            print(f"  New state: {state}")
            
            action_mask = env.get_action_mask()
            print(f"  Updated action mask: {action_mask}")
        else:
            print("  No valid actions available!")
            break
    
    # Test invalid action
    print(f"\nTesting invalid action...")
    # Try to overload a server
    overload_action = 0
    for _ in range(Config.SERVER_CAPACITIES[0] + 2):
        prompt = "Test overload prompt"
        state, reward, done, info = env.step(overload_action, prompt)
        if not info['valid_action']:
            print(f"Invalid action detected! Reward: {reward:.3f}")
            break
    
    # Get environment stats
    print(f"\nEnvironment Statistics:")
    stats = env.get_environment_stats()
    for key, value in stats.items():
        if key != 'servers':
            print(f"  {key}: {value}")
    
    print(f"\nServer Details:")
    for i, server_stats in stats['servers'].items():
        print(f"  {i}: {server_stats}")

def test_agent_integration():
    """Test PPO agent integration with enhanced environment"""
    print("\n\nTesting PPO Agent Integration...")
    print("=" * 50)
    
    env = EnhancedRouterEnvironment()
    data_loader = AlpacaDataLoader()
    
    # Initialize agent
    state_dim = len(Config.SERVER_CAPACITIES) * 2  # load + utilization per server
    action_dim = len(Config.SERVER_CAPACITIES)
    agent = PPOAgent(state_dim, action_dim)
    
    print(f"Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    # Test agent action selection
    state = env.reset()
    prompt = data_loader.get_random_prompt()
    action_mask = env.get_action_mask()
    
    print(f"\nTesting agent action selection:")
    print(f"State: {state}")
    print(f"Action mask: {action_mask}")
    
    action, log_prob, value = agent.get_action(state, prompt, action_mask)
    print(f"Agent selected action: {action}")
    print(f"Log probability: {log_prob:.3f}")
    print(f"Value estimate: {value:.3f}")
    
    # Test multiple steps with agent
    print(f"\nRunning 10 steps with agent...")
    total_reward = 0
    trajectories = []
    
    for step in range(10):
        prompt = data_loader.get_random_prompt()
        action_mask = env.get_action_mask()
        
        action, log_prob, value = agent.get_action(state, prompt, action_mask)
        next_state, reward, done, info = env.step(action, prompt)
        
        # Store trajectory
        trajectory = {
            'state': state.copy(),
            'prompt': prompt,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'action_mask': action_mask.copy()
        }
        trajectories.append(trajectory)
        
        total_reward += reward
        state = next_state
        
        print(f"  Step {step + 1}: Action={action}, Reward={reward:.3f}, Valid={info['valid_action']}")
    
    print(f"Total reward over 10 steps: {total_reward:.3f}")
    
    # Test agent update
    print(f"\nTesting agent update with collected trajectories...")
    if trajectories:
        training_metrics = agent.update(trajectories)
        print(f"Training metrics: {training_metrics}")

def test_server_queuing():
    """Test realistic server queuing and completion"""
    print("\n\nTesting Server Queuing System...")
    print("=" * 50)
    
    env = EnhancedRouterEnvironment()
    data_loader = AlpacaDataLoader()
    
    # Fill up server 0 to capacity
    server_0_capacity = Config.SERVER_CAPACITIES[0]
    print(f"Filling server 0 to capacity ({server_0_capacity} requests)...")
    
    state = env.reset()
    requests_added = 0
    
    for i in range(server_0_capacity + 2):  # Try to add more than capacity
        prompt = f"Test request {i + 1}"
        action = 0  # Always try server 0
        
        state, reward, done, info = env.step(action, prompt)
        
        if info['valid_action']:
            requests_added += 1
            print(f"  Added request {i + 1}, server load now: {info['server_loads'][0]}")
        else:
            print(f"  Request {i + 1} REJECTED - server at capacity")
            print(f"  Penalty reward: {reward:.3f}")
            break
    
    print(f"Successfully added {requests_added} requests to server 0")
    
    # Now advance time and see completions
    print(f"\nAdvancing time to see request completions...")
    for step in range(50):  # Advance time
        # Use server 1 for new requests while server 0 processes
        prompt = f"Time step {step} request"
        action = 1  # Use server 1
        
        state, reward, done, info = env.step(action, prompt)
        
        if info['completed_requests'] > 0:
            print(f"  Step {step}: {info['completed_requests']} requests completed!")
            print(f"    Server loads: {info['server_loads']}")
            print(f"    Step reward from completions: {info.get('step_rewards', 0):.3f}")
        
        # Check if server 0 has capacity again
        action_mask = env.get_action_mask()
        if action_mask[0] > 0:
            print(f"  Step {step}: Server 0 has capacity again!")
            break
    
    # Final environment stats
    final_stats = env.get_environment_stats()
    print(f"\nFinal Environment Stats:")
    print(f"  Total requests created: {final_stats['total_requests_created']}")
    print(f"  Total requests completed: {final_stats['total_requests_completed']}")
    if 'avg_latency' in final_stats:
        print(f"  Average end-to-end latency: {final_stats['avg_latency']:.3f}s")

def test_reward_calculation():
    """Test reward calculation components"""
    print("\n\nTesting Reward Calculation...")
    print("=" * 50)
    
    env = EnhancedRouterEnvironment()
    data_loader = AlpacaDataLoader()
    
    print(f"Reward function: R = {Config.ALPHA}*Q - {Config.BETA}*L - {Config.LAMBDA}*max(0, D-C)")
    print(f"Where Q=quality, L=latency, D=load, C=capacity")
    
    state = env.reset()
    
    # Test different prompts and servers
    test_prompts = [
        "Simple task: What is 2+2?",
        "Complex analysis: Explain the implications of quantum computing on modern cryptography",
        "Medium task: Write a short story about a robot"
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest {i + 1}: {prompt}")
        
        for server_id in range(len(Config.SERVER_CAPACITIES)):
            action_mask = env.get_action_mask()
            
            if action_mask[server_id] > 0:  # If server is available
                state, reward, done, info = env.step(server_id, prompt)
                
                print(f"  Server {server_id}:")
                print(f"    Quality score: {info.get('quality_score', 0):.3f}")
                print(f"    Estimated latency: {info.get('estimated_latency', 0):.3f}s")
                print(f"    Capacity penalty: {info.get('capacity_penalty', 0):.3f}")
                print(f"    Immediate reward: {info.get('immediate_reward', 0):.3f}")
                print(f"    Total step reward: {reward:.3f}")
            else:
                print(f"  Server {server_id}: Not available (at capacity)")

def main():
    """Run all tests"""
    print("Enhanced LLM Router System Test Suite")
    print("====================================")
    
    # Set random seed for reproducible tests
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        test_environment_basic()
        test_agent_integration()
        test_server_queuing()
        test_reward_calculation()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("The enhanced system is ready for training.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
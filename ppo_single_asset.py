"""
PPO implementation for single-asset trading with reference policy regularization.

Requirements:
pip install torch gym pandas numpy tqdm matplotlib

Usage:
python ppo_single_asset.py --total-steps 2_000_000 --beta 0.05
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import argparse
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt

class BollingerPolicy:
    """Reference policy using Bollinger Bands strategy"""
    def action_prob(self, state: np.ndarray) -> np.ndarray:
        # Extract price series (assume first feature is price)
        prices = state[:, 0]
        
        # Calculate Bollinger Bands (20-period)
        rolling_mean = pd.Series(prices).rolling(window=20).mean().iloc[-1]
        rolling_std = pd.Series(prices).rolling(window=20).std().iloc[-1]
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        current_price = prices[-1]
        
        # Set action probabilities based on price position
        if current_price > upper_band:
            return np.array([0.7, 0.2, 0.1])  # [short, flat, long]
        elif current_price < lower_band:
            return np.array([0.1, 0.2, 0.7])  # [short, flat, long]
        else:
            return np.array([0.2, 0.6, 0.2])  # [short, flat, long]

class ActorCritic(nn.Module):
    def __init__(self, input_dim: Tuple[int, int], n_actions: int):
        super().__init__()
        
        # Shared features
        self.features = nn.Sequential(
            nn.Conv1d(input_dim[1], 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (input_dim[0] - 4), 128),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        x = x.permute(0, 2, 1)  # [batch, features, timesteps]
        features = self.features(x)
        
        # Actor: output action distribution
        action_probs = torch.softmax(self.actor(features), dim=-1)
        dist = Categorical(action_probs)
        
        # Critic: output state value
        value = self.critic(features)
        
        return dist, value

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def clear(self):
        self.__init__()
        
    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [True])
        
        # Compute GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1]
            next_done = dones[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            
        returns = advantages + values[:-1]
        return torch.FloatTensor(returns), torch.FloatTensor(advantages)

def train_ppo(
    env,
    actor_critic: ActorCritic,
    ref_policy,
    total_steps: int,
    beta: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[float]:
    # Initialize
    actor_critic.to(device)
    actor_optimizer = optim.Adam(actor_critic.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(actor_critic.parameters(), lr=1e-3)
    buffer = RolloutBuffer()
    
    # Training loop variables
    step_count = 0
    episode_rewards = []
    best_mean_reward = float("-inf")
    no_improvement_count = 0
    
    # Progress bar setup
    pbar = tqdm(total=total_steps, desc='Training')
    running_reward = 0
    
    # Buy and hold strategy returns
    buy_hold_returns = []
    
    while step_count < total_steps:
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Track buy & hold returns for this episode
        buy_hold_reward = 0
        prices = []
        
        while not done:
            # Store price for buy & hold calculation
            prices.append(state[:, 0][-1])  # Assuming first feature is price
            
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                dist, value = actor_critic(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action.item())
            
            # Store transition
            buffer.states.append(state)
            buffer.actions.append(action.item())
            buffer.rewards.append(reward)
            buffer.values.append(value.item())
            buffer.log_probs.append(log_prob.item())
            buffer.dones.append(done)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Update progress bar
            pbar.update(1)
            running_reward = 0.95 * running_reward + 0.05 * reward
            pbar.set_postfix({
                'reward': f'{running_reward:.2f}',
                'episode': len(episode_rewards)
            })
            
            # Update if episode ends or buffer is full
            if done or len(buffer.states) >= 2048:
                # Get final value estimate
                with torch.no_grad():
                    _, last_value = actor_critic(
                        torch.FloatTensor(state).unsqueeze(0).to(device)
                    )
                
                # Compute returns and advantages
                returns, advantages = buffer.compute_returns_and_advantages(
                    last_value.item(), gamma=0.99, gae_lambda=0.95
                )
                
                # Convert buffer data to tensors
                states = torch.FloatTensor(np.array(buffer.states)).to(device)
                actions = torch.LongTensor(buffer.actions).to(device)
                old_log_probs = torch.FloatTensor(buffer.log_probs).to(device)
                
                # PPO update (multiple epochs)
                for _ in range(10):
                    # Generate mini-batches
                    indices = np.random.permutation(len(buffer.states))
                    for start_idx in range(0, len(buffer.states), 64):
                        batch_indices = indices[start_idx:start_idx + 64]
                        
                        # Get batch data
                        batch_states = states[batch_indices]
                        batch_actions = actions[batch_indices]
                        batch_advantages = advantages[batch_indices]
                        batch_returns = returns[batch_indices]
                        batch_old_log_probs = old_log_probs[batch_indices]
                        
                        # Forward pass
                        dist, value = actor_critic(batch_states)
                        new_log_probs = dist.log_prob(batch_actions)
                        
                        # Calculate ratios and surrogate losses
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * batch_advantages
                        
                        # KL regularization with reference policy
                        ref_probs = torch.FloatTensor(
                            np.array([ref_policy.action_prob(s) for s in batch_states.cpu().numpy()])
                        ).to(device)
                        kl_div = (ref_probs * torch.log(ref_probs / torch.softmax(dist.logits, dim=-1))).sum(-1)
                        
                        # Calculate losses
                        actor_loss = -torch.min(surr1, surr2).mean() + beta * kl_div.mean()
                        critic_loss = nn.MSELoss()(value.squeeze(), batch_returns)
                        
                        # Update actor and critic
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        actor_loss.backward()
                        critic_loss.backward()
                        actor_optimizer.step()
                        critic_optimizer.step()
                
                buffer.clear()
            
            # Calculate buy & hold return for this episode
            if len(prices) > 1:
                buy_hold_return = np.log(prices[-1] / prices[0])  # Log return
                buy_hold_returns.append(buy_hold_return)
        
        episode_rewards.append(episode_reward)
        
        # Display episode summary
        if len(episode_rewards) % 10 == 0:
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            mean_buy_hold = np.mean(buy_hold_returns[-100:]) if len(buy_hold_returns) >= 100 else np.mean(buy_hold_returns)
            print(f"\nEpisode {len(episode_rewards)}")
            print(f"Mean PPO reward (last 100): {mean_reward:.3f}")
            print(f"Mean Buy&Hold return (last 100): {mean_buy_hold:.3f}")
            print(f"PPO vs B&H ratio: {(mean_reward/mean_buy_hold):.2f}x")
        
        # Early stopping check
        if len(episode_rewards) >= 100:
            mean_reward = np.mean(episode_rewards[-100:])
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(actor_critic.state_dict(), "best_model.pt")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= 20:
                print(f"Early stopping at step {step_count}")
                break
    
    pbar.close()
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label='PPO')
    plt.plot(buy_hold_returns, label='Buy & Hold')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Performance: PPO vs Buy & Hold')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.close()
    
    return episode_rewards, buy_hold_returns

def evaluate(env, actor_critic: ActorCritic, n_episodes: int = 10) -> float:
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            with torch.no_grad():
                dist, _ = actor_critic(torch.FloatTensor(state).unsqueeze(0))
                action = dist.sample()
            state, reward, done, _ = env.step(action.item())
            episode_reward += reward
            
        rewards.append(episode_reward)
    
    return np.mean(rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--beta", type=float, default=0.05)
    args = parser.parse_args()
    
    # Initialize environment and models
    env = gym.make("Stock_gym-v0")  # Assume this exists
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    actor_critic = ActorCritic(state_shape, n_actions)
    ref_policy = BollingerPolicy()
    
    # Train with updated return values
    rewards, buy_hold_returns = train_ppo(env, actor_critic, ref_policy, args.total_steps, args.beta)
    
    # Final evaluation
    mean_reward = evaluate(env, actor_critic)
    mean_buy_hold = np.mean(buy_hold_returns[-100:])
    print(f"\nFinal Results:")
    print(f"PPO mean reward: {mean_reward:.3f}")
    print(f"Buy&Hold mean return: {mean_buy_hold:.3f}")
    print(f"PPO vs B&H ratio: {(mean_reward/mean_buy_hold):.2f}x") 
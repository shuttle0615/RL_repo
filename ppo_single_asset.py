"""
PPO implementation for single-asset trading with reference policy regularization.

Requirements:
pip install torch gymnasium pandas numpy tqdm matplotlib ta-lib

Usage:
python ppo_single_asset.py --total-steps 2_000_000 --beta 0.05
"""
import math
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
from src.environment import BitcoinTradingEnv

class ActorCritic(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], n_actions: int):
        super().__init__()
        
        # CNN for processing market data
        self.cnn = nn.Sequential(
            nn.Conv1d(input_shape[1], 32, kernel_size=7),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten()
        )
        
        # Position embedding
        self.position_embedding = nn.Embedding(3, 32)  # 3 positions to 32-dim
        
        # Combine features
        self.fc = nn.Sequential(
            nn.Linear(128 + 32, 128),  # CNN output + position embedding
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
    
    def forward(self, market_data: torch.Tensor, position: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        # Process market data through CNN
        x = market_data.permute(0, 2, 1)  # [batch, features, timesteps]
        cnn_features = self.cnn(x)
        
        # Process position
        pos_features = self.position_embedding(position)
        
        # Combine features
        combined = torch.cat([cnn_features, pos_features], dim=1)
        features = self.fc(combined)
        
        # Actor: output action distribution
        action_probs = torch.softmax(self.actor(features), dim=-1)
        dist = Categorical(action_probs)
        
        # Critic: output state value
        value = self.critic(features)
        
        return dist, value

class BollingerPolicy:
    """Reference policy using Bollinger Bands strategy"""
    def action_prob(self, state: dict) -> np.ndarray:
        market_data = state["market_data"]
        current_price = market_data[-1, 3]  # Last close price
        
        # Calculate Bollinger Bands on the window
        prices = market_data[:, 3]  # All close prices
        rolling_mean = np.mean(prices[-20:])
        rolling_std = np.std(prices[-20:])
        
        upper_band = rolling_mean + 2 * rolling_std
        lower_band = rolling_mean - 2 * rolling_std
        
        # Set action probabilities based on price position
        if current_price > upper_band:
            return np.array([0.7, 0.2, 0.1])  # [short, flat, long]
        elif current_price < lower_band:
            return np.array([0.1, 0.2, 0.7])  # [short, flat, long]
        else:
            return np.array([0.2, 0.6, 0.2])  # [short, flat, long]

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
    env: BitcoinTradingEnv,
    actor_critic: ActorCritic,
    ref_policy: BollingerPolicy,
    total_steps: int,
    beta: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[float]:
    # Initialize with single optimizer
    actor_critic.to(device)
    optimizer = optim.Adam(actor_critic.parameters(), lr=1e-4)
    
    # Loss coefficients
    value_coef = 0.5       # c1 in PPO paper
    entropy_coef = 0.01    # c2 in PPO paper
    imitation_coef = beta  # Bollinger imitation weight
    
    # Training loop variables
    step_count = 0
    episode_rewards = []
    best_mean_reward = float("-inf")
    no_improvement_count = 0
    
    # Progress bar setup
    pbar = tqdm(total=total_steps, desc='Training')
    
    buffer = RolloutBuffer()
    
    while step_count < total_steps:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Convert state dict to tensors
            market_data = torch.FloatTensor(state["market_data"]).unsqueeze(0).to(device)
            position = torch.LongTensor([state["position"]]).to(device)
            
            # Get action from policy
            with torch.no_grad():
                dist, value = actor_critic(market_data, position)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action.item())
            
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
            pbar.set_postfix({
                'reward': f'{np.exp(episode_reward):.5f}',
                'episode': len(episode_rewards)
            })
            
            # Update if episode ends or buffer is full
            if done or len(buffer.states) >= 2048:
                # Get final value estimate
                with torch.no_grad():
                    _, last_value = actor_critic(
                        torch.FloatTensor(state["market_data"]).unsqueeze(0).to(device),
                        torch.LongTensor([state["position"]]).to(device)
                    )
                
                # Compute returns and advantages
                returns, advantages = buffer.compute_returns_and_advantages(
                    last_value.item(), gamma=0.99, gae_lambda=0.95
                )
                
                # normalize the advantage
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Convert buffer data to tensors
                market_data = torch.FloatTensor(
                    np.array([s["market_data"] for s in buffer.states])
                ).to(device)
                positions = torch.LongTensor(
                    [s["position"] for s in buffer.states]
                ).to(device)
                actions = torch.LongTensor(buffer.actions).to(device)
                old_log_probs = torch.FloatTensor(buffer.log_probs).to(device)
                
                # PPO update (multiple epochs)
                for _ in range(10):
                    # Generate mini-batches
                    indices = np.random.permutation(len(buffer.states))
                    for start_idx in range(0, len(buffer.states), 64):
                        batch_indices = indices[start_idx:start_idx + 64]
                        
                        # Get batch data
                        batch_market_data = market_data[batch_indices]
                        batch_positions = positions[batch_indices]
                        batch_actions = actions[batch_indices]
                        batch_advantages = advantages[batch_indices]
                        batch_returns = returns[batch_indices]
                        batch_old_log_probs = old_log_probs[batch_indices]
                        
                        # Forward pass
                        dist, value = actor_critic(batch_market_data, batch_positions)
                        new_log_probs = dist.log_prob(batch_actions)
                        
                        # Entropy bonus
                        entropy = dist.entropy().mean()
                        
                        # PPO surrogate losses
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value function loss
                        value_loss = nn.functional.mse_loss(value.squeeze(), batch_returns)
                        
                        # KL regularization with reference policy
                        batch_states = [buffer.states[i] for i in batch_indices]
                        ref_probs = torch.FloatTensor(
                            np.array([ref_policy.action_prob(s) for s in batch_states])
                        ).to(device)
                        kl_div = (ref_probs * torch.log(ref_probs / torch.softmax(dist.logits, dim=-1))).sum(-1)
                        imitation_loss = kl_div.mean()
                        
                        # Compute total loss
                        total_loss = (
                            policy_loss
                            + value_coef * value_loss
                            - entropy_coef * entropy  # Note minus sign: maximize entropy
                            + imitation_coef * imitation_loss
                        )
                        
                        # Single backward pass and optimization step
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=1)
                        optimizer.step()
                        
                        # Optional: log different loss components
                        if step_count % 1000 == 0:
                            print(f"\nLoss components:")
                            print(f"Policy loss: {policy_loss.item():.6f}")
                            print(f"Value loss: {value_loss.item():.6f}")
                            print(f"Entropy: {entropy.item():.6f}")
                            print(f"Imitation loss: {imitation_loss.item():.6f}")
                            print(f"Total loss: {total_loss.item():.6f}")
                            print('value_loss', value_loss.item(),
                                    'adv mean', batch_advantages.mean().item(),
                                    'adv sig',  batch_advantages.std().item())
                
                buffer.clear()
        
        episode_rewards.append(episode_reward)
        
        # Display episode summary
        if len(episode_rewards) % 10 == 0:
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"\nEpisode {len(episode_rewards)}")
            print(f"Mean reward (last 100): {np.exp(mean_reward):.3f}")
        
        # Early stopping check
        if len(episode_rewards) >= 100:
            mean_reward = np.mean(episode_rewards[-100:])
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(actor_critic.state_dict(), "best_model.pt")
                print(f"Best mean reward: {best_mean_reward:.3f} - model saved")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= 20000:
                print(f"Early stopping at step {step_count}")
                break
    
    pbar.close()
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Training Performance')
    plt.grid(True)
    plt.savefig('training_curve.png')
    plt.close()
    
    return episode_rewards

def evaluate(env: BitcoinTradingEnv, actor_critic: ActorCritic, n_episodes: int = 100, mode: str = 'test') -> float:
    device = next(actor_critic.parameters()).device
    rewards = []
    
    for _ in tqdm(range(n_episodes)):
        state, _ = env.reset(mode=mode)
        episode_reward = 0
        done = False
        
        while not done:
            market_data = torch.FloatTensor(state["market_data"]).unsqueeze(0).to(device)
            position = torch.LongTensor([state["position"]]).to(device)
            
            with torch.no_grad():
                dist, _ = actor_critic(market_data, position)
                action = dist.sample()
            
            state, reward, done, _ = env.step(action.item())
            episode_reward += reward
            
        rewards.append(episode_reward)
    
    return np.mean(rewards), rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--beta", type=float, default=0.00)
    parser.add_argument("--data-path", type=str, default="data/BTCUSDT_1h_full_history.csv")
    args = parser.parse_args()
    
    # Define date ranges for train/val/test splits
    date_ranges = {
        'train': ('2019-09-01', '2022-12-31'),
        'val': ('2023-01-01', '2023-06-30'),
        'test': ('2023-07-01', '2023-12-31')
    }
    
    # Initialize environment
    env = BitcoinTradingEnv(
        csv_path=args.data_path,
        window_size=168,  # 7 days * 24 hours
        episode_length=168,
        date_ranges_by_mode=date_ranges
    )
    
    # Initialize models
    input_shape = env.observation_space["market_data"].shape
    n_actions = env.action_space.n
    
    actor_critic = ActorCritic(input_shape, n_actions)
    ref_policy = BollingerPolicy()
    
    # Train
    #rewards = train_ppo(env, actor_critic, ref_policy, args.total_steps, args.beta)
    
    # Final evaluation
    actor_critic.load_state_dict(torch.load("best_model.pt"))
    test_reward, test_rewards = evaluate(env, actor_critic, mode='test')
    print(f"\nTest set evaluation mean reward: {np.exp(test_reward):.3f}") 
    print(f"Test set rewards: {[np.exp(r) for r in test_rewards]}")
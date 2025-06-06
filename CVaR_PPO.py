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

torch.set_num_threads(4)        
torch.set_num_interop_threads(2) 

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

class PositionalEncoding(nn.Module):
    """Classic sine–cos positional encodings, fixed (no grads)."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        return x + self.pe[: x.size(1)]

class ActorCritic_transformer(nn.Module):
    def __init__(self, input_shape: Tuple[int, int], n_actions: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        """
        input_shape: (T, F)  where T = window length (time), F = features.
        n_actions  : 3 (short / flat / long) by default.
        """
        super().__init__()
        T, F = input_shape                 # time steps, feature dims

        # 1) Linear projection of raw features → d_model
        self.feat_proj = nn.Linear(F, d_model)

        # 2) Positional encoding + tiny Transformer encoder
        self.pos_enc = PositionalEncoding(d_model, max_len=T)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=n_layers)

        # 3) Attention-pool (learned) over the time dimension
        self.attn_pool = nn.Linear(d_model, 1)

        # 4) Position embedding (same as before)
        self.position_embedding = nn.Embedding(3, 32)        # 3 positions

        # 5) Separate torsos for policy / value (upgrade #5)
        torso_dim = d_model + 32
        self.policy_torso = nn.Sequential(
            nn.LayerNorm(torso_dim),
            nn.Linear(torso_dim, 128), nn.ReLU())
        self.value_torso = nn.Sequential(
            nn.LayerNorm(torso_dim),
            nn.Linear(torso_dim, 128), nn.ReLU())

        # 6) Heads
        self.actor_head  = nn.Linear(128, n_actions)
        self.critic_head = nn.Linear(128, 1)

    # ─────────────────────────────────────────────────────────────

    def _transformer_features(self, market: torch.Tensor) -> torch.Tensor:
        """
        market: (B, T, F)  →  returns pooled per-batch feature (B, d_model)
        """
        x = self.feat_proj(market)          # (B, T, D)
        x = self.pos_enc(x)
        x = self.transformer(x)             # (B, T, D)

        # Learned attention pooling
        w = self.attn_pool(x).softmax(dim=1)   # (B, T, 1)
        pooled = (x * w).sum(dim=1)           # (B, D)
        return pooled

    def forward(self,
                market_data: torch.Tensor,    # (B, T, F)
                position: torch.Tensor        # (B,) long int
                ) -> Tuple[Categorical, torch.Tensor]:

        batch_feat = self._transformer_features(market_data)
        pos_feat   = self.position_embedding(position)
        combined   = torch.cat([batch_feat, pos_feat], dim=-1)

        # Split torsos so critic can't overwrite policy features
        pol_latent = self.policy_torso(combined)
        val_latent = self.value_torso(combined)

        logits = self.actor_head(pol_latent)
        dist   = Categorical(logits=logits)
        value  = self.critic_head(val_latent)

        return dist, value

# ---------- 1. Dilated Temporal Convolution Block ------------------------ #
class TCNBlock(nn.Module):
    """1-D causal convolution with dilation + residual skip."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation        # *causal* left padding
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=kernel, dilation=dilation,
            padding=pad, bias=False
        )
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Conv1d(in_ch, out_ch, 1, bias=False)

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x):
        y = self.conv(x)
        y = y[..., :-self.conv.padding[0]]          # strip extra timesteps
        y = self.act(self.bn(y))

        res = x if self.downsample is None else self.downsample(x)
        return self.act(y + res)                    # residual sum


class DilatedTCN(nn.Module):
    """3-layer stack with dilations 1,2,4; kernel_size = 3 by default."""
    def __init__(self, in_ch: int, hidden: int, kernel: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_ch
        for d in (1, 2, 4):
            layers.append(TCNBlock(ch, hidden, kernel, dilation=d))
            ch = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):               # x: (B, F, T)
        return self.net(x)              # -> (B, hidden, T)
# ------------------------------------------------------------------------- #


# ---------- 2. Self-Attention over the time axis ------------------------- #
class TimeAttention(nn.Module):
    """Multi-head self-attention across the *temporal* dimension."""
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=True
        )

    def forward(self, h):               # h: (B, T, H)
        # Allow the model to re-weigh timesteps before pooling
        out, _ = self.attn(h, h, h)
        return out
# ------------------------------------------------------------------------- #


# ---------- 3. Actor-Critic --------------------------------------------- #
class ActorCritic_tcn(nn.Module):
    """
    PPO-style actor-critic for *single-asset* trading.
    Input  : (B, T, F)  – window of market features
    position: (B,)      – discrete 0=flat,1=long,2=short
    Output :   dist over n_actions  &  value estimate
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],  # (T, F)
        n_actions: int,
        hidden: int = 64,
        pos_dim: int = 32,
        attn_heads: int = 4,
    ):
        super().__init__()
        _, feat = input_shape

        # (1) Temporal encoder → (B, H, T)
        self.tcn = DilatedTCN(in_ch=feat, hidden=hidden)

        # (2) Time-attention → (B, T, H)
        self.t_attn = TimeAttention(dim=hidden, n_heads=attn_heads)

        # (3) Global representation
        self.gap = nn.AdaptiveAvgPool1d(1)          # (B, H, 1)
        self.ln  = nn.LayerNorm(hidden)

        # Position embedding (long / flat / short)
        self.pos_emb = nn.Embedding(3, pos_dim)

        # Fusion MLP
        self.fuse = nn.Sequential(
            nn.Linear(hidden + pos_dim, 128),
            nn.ReLU(inplace=True)
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, n_actions)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    # --------------------------- forward ---------------------------------- #
    def forward(
        self,
        market_data: torch.Tensor,   # (B, T, F)
        position: torch.Tensor      # (B,)
    ):
        # step-1  Temporal conv
        x = market_data.permute(0, 2, 1)           # → (B, F, T)
        h = self.tcn(x)                            # → (B, H, T)

        # step-2  Attention across time
        h_t = h.permute(0, 2, 1)                   # (B, T, H)
        h_t = self.t_attn(h_t)                     # (B, T, H)

        # step-3  Global average pool + norm
        g = self.gap(h_t.permute(0, 2, 1)).squeeze(-1)   # (B, H)
        g = self.ln(g)

        # step-4  Position embedding & fusion
        p = self.pos_emb(position)                 # (B, pos_dim)
        z = torch.cat([g, p], dim=1)               # (B, H+pos_dim)
        z = self.fuse(z)                           # (B, 128)

        # Actor
        logits = self.actor(z)
        dist   = Categorical(logits=logits)

        # Critic
        value  = self.critic(z)

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

class OracleLookaheadPolicy:
    """
    *Cheating* baseline that sees `horizon` steps into the future and chooses
    the action with the highest profit after transaction fees.

    Parameters
    ----------
    env : BitcoinTradingEnv
        The live environment instance (so the policy can read `df` and
        `current_df_idx`).
    horizon : int, default=1
        How many rows ahead to peek. horizon=1 -> look at the next bar only.
    fee : float, default=0.001
        One-way fee used to decide whether a tiny change is worth trading.
    """

    def __init__(self, env, horizon: int = 1, fee: float = 0.001):
        self.env     = env
        self.horizon = horizon
        self.fee     = fee          # fee as decimal (0.1 % → 0.001)

    # ------------------------------------------------------------------ #
    # interface identical to your BollingerPolicy
    # ------------------------------------------------------------------ #
    def action_prob(self, state: dict) -> np.ndarray:
        """Return a probability vector `[p_short, p_flat, p_long]`."""

        idx_now     = self.env.current_df_idx
        idx_future  = idx_now + self.horizon

        # If we're at the very end of the data, fall back to flat.
        if idx_future > self.env.active_mode_end_idx:
            return np.array([0.2, 0.6, 0.2], dtype=np.float32)

        price_now    = self.env.df.loc[idx_now,  'Close']
        price_future = self.env.df.loc[idx_future, 'Close']

        gross_ret    = (price_future - price_now) / price_now   # simple %
        thr          = self.fee * 2.0                           # fee round-trip

        if   gross_ret >  thr:   # sufficiently large rise
            return np.array([0.05, 0.05, 0.90], dtype=np.float32)   # go long
        elif gross_ret < -thr:   # sufficiently large drop
            return np.array([0.90, 0.05, 0.05], dtype=np.float32)   # go short
        else:                    # price roughly unchanged → stay flat
            return np.array([0.20, 0.60, 0.20], dtype=np.float32)


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.traj_returns = []  # Store episode returns for CVaR
        
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
    optimizer = optim.Adam(actor_critic.parameters(), lr=5e-5)
    
    # CVaR hyperparameters
    alpha = 0.1  # tail fraction
    lr_lambda = 1e-3
    eta = 0.0  # CVaR threshold
    cvar_lambda = 0.0  # Lagrange multiplier
    gamma = 0.99  # discount factor
    
    # Loss coefficients
    value_coef = 0.5
    entropy_coef = 0.005
    imitation_coef = beta
    
    # Training loop variables
    step_count = 0
    episode_rewards = []
    best_mean_reward = float("-inf")
    no_improvement_count = 0
    
    pbar = tqdm(total=total_steps, desc='Training')
    buffer = RolloutBuffer()
    
    while step_count < total_steps:
        state, _ = env.reset()
        episode_reward = 0
        disc_return = 0
        gamma_pow = 1.0
        done = False
        
        while not done:
            market_data = torch.FloatTensor(state["market_data"]).unsqueeze(0).to(device)
            position = torch.LongTensor([state["position"]]).to(device)
            
            with torch.no_grad():
                dist, value = actor_critic(market_data, position)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            next_state, reward, done, info = env.step(action.item())
            
            # Store transition
            buffer.states.append(state)
            buffer.actions.append(action.item())
            buffer.rewards.append(reward)
            buffer.values.append(value.item())
            buffer.log_probs.append(log_prob.item())
            buffer.dones.append(done)
            
            # Track discounted return
            disc_return += gamma_pow * reward
            gamma_pow *= gamma
            
            state = next_state
            episode_reward += reward
            step_count += 1

            pbar.update(1)
            pbar.set_postfix({
                'reward': f'{np.exp(episode_reward):.5f}',
                'episode': len(episode_rewards)
            })
            
            if done or len(buffer.states) >= 2048:
                if done:
                    buffer.traj_returns.append(disc_return)
                
                with torch.no_grad():
                    _, last_value = actor_critic(
                        torch.FloatTensor(state["market_data"]).unsqueeze(0).to(device),
                        torch.LongTensor([state["position"]]).to(device)
                    )
                
                returns, advantages = buffer.compute_returns_and_advantages(
                    last_value.item(), gamma=0.99, gae_lambda=0.95
                )
                
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                returns = returns.to(device)
                advantages = advantages.to(device)

                # Update CVaR parameters
                if len(buffer.traj_returns) > 0:
                    eta = float(np.quantile(buffer.traj_returns, alpha))
                    pos_part = np.maximum(0, eta - np.array(buffer.traj_returns))
                    cvar_lambda += lr_lambda * (np.mean(pos_part)/(1-alpha) + beta - eta)

                # Convert buffer data to tensors
                market_data = torch.FloatTensor(
                    np.array([s["market_data"] for s in buffer.states])
                ).to(device)
                positions = torch.LongTensor(
                    [s["position"] for s in buffer.states]
                ).to(device)
                actions = torch.LongTensor(buffer.actions).to(device)
                old_log_probs = torch.FloatTensor(buffer.log_probs).to(device)
                
                # Create tensor of discounted returns for each transition
                batch_disc_returns = torch.FloatTensor([
                    buffer.traj_returns[i] for i in range(len(buffer.traj_returns))
                    for _ in range(len(buffer.states) // len(buffer.traj_returns))
                ]).to(device)
                
                for _ in range(10):
                    indices = np.random.permutation(len(buffer.states))
                    for start_idx in range(0, len(buffer.states), 128):
                        batch_indices = indices[start_idx:start_idx + 128]
                        
                        batch_market_data = market_data[batch_indices]
                        batch_positions = positions[batch_indices]
                        batch_actions = actions[batch_indices]
                        batch_advantages = advantages[batch_indices]
                        batch_returns = returns[batch_indices]
                        batch_old_log_probs = old_log_probs[batch_indices]
                        batch_disc_rets = batch_disc_returns[batch_indices]
                        
                        dist, value = actor_critic(batch_market_data, batch_positions)
                        new_log_probs = dist.log_prob(batch_actions)
                        entropy = dist.entropy().mean()
                        
                        # CVaR-adjusted advantage
                        adv_cvar = batch_advantages - (cvar_lambda/(1-alpha)) * torch.clamp(
                            -(batch_disc_rets - eta), min=0.0
                        )
                        
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        policy_loss = -torch.min(
                            ratio * adv_cvar,
                            torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_cvar
                        ).mean()
                        
                        value_loss = nn.functional.mse_loss(value.squeeze(), batch_returns)
                        
                        batch_states = [buffer.states[i] for i in batch_indices]
                        ref_probs = torch.FloatTensor(
                            np.array([ref_policy.action_prob(s) for s in batch_states])
                        ).to(device)
                        kl_div = (ref_probs * torch.log(ref_probs / torch.softmax(dist.logits, dim=-1))).sum(-1)
                        imitation_loss = kl_div.mean()
                        
                        total_loss = (
                            policy_loss
                            + value_coef * value_loss
                            - entropy_coef * entropy
                            + imitation_coef * imitation_loss
                        )
                        
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), max_norm=1)
                        optimizer.step()
                        
                # Log CVaR estimate
                if len(buffer.traj_returns) > 0:
                    curr_cvar = eta + np.mean(np.maximum(0, eta - np.array(buffer.traj_returns)))/(1-alpha)
                    pbar.set_postfix({
                        'CVaR': f'{curr_cvar:.3f}',
                        'eta': f'{eta:.3f}',
                        'lambda': f'{cvar_lambda:.3f}'
                    })
                
                buffer.clear()
        
        episode_rewards.append(episode_reward)
        
        if len(episode_rewards) % 10 == 0:
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"\nEpisode {len(episode_rewards)}")
            print(f"Mean reward (last 100): {np.exp(mean_reward):.3f}")
            print(f"Max reward: {np.exp(max(episode_rewards[-100:])):.3f}")
            print(f"Min reward: {np.exp(min(episode_rewards[-100:])):.3f}")
            print("sign", env.flip_sign)
        
        if len(episode_rewards) >= 10:
            mean_reward = np.mean(episode_rewards[-1])

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(actor_critic.state_dict(), f"best_model_{np.exp(best_mean_reward):.3f}.pt")
                print(f"Best mean reward: {best_mean_reward:.3f} - model saved")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= 400:
                print(f"Early stopping at step {step_count}")
                break
    
    pbar.close()
    return episode_rewards

def evaluate(env: BitcoinTradingEnv, actor_critic: ActorCritic, n_episodes: int = 100, mode: str = 'test') -> float:
    device = next(actor_critic.parameters()).device
    rewards = []
    actions = []
    
    for _ in tqdm(range(n_episodes)):
        state, _ = env.reset(mode=mode)
        episode_reward = 0
        done = False
        action_list = []
        reward_list = []
        
        while not done:
            market_data = torch.FloatTensor(state["market_data"]).unsqueeze(0).to(device)
            position = torch.LongTensor([state["position"]]).to(device)
            
            with torch.no_grad():
                dist, _ = actor_critic(market_data, position)
                action = dist.logits.argmax(dim=-1) #dist.sample() #dist.logits.argmax(dim=-1) #dist.sample() #
            
            state, reward, done, _ = env.step(action.item(), mode=mode)
            episode_reward += reward
            action_list.append(action.item())
            reward_list.append(reward)
        rewards.append((episode_reward, reward_list))
        actions.append(action_list)

    return np.mean([r[0] for r in rewards]), rewards, actions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=4_000_000)
    parser.add_argument("--beta", type=float, default=0.000)
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
        episode_length=26000,
        date_ranges_by_mode=date_ranges
    )
    
    # Initialize models
    input_shape = env.observation_space["market_data"].shape
    n_actions = env.action_space.n
    
    actor_critic = ActorCritic_tcn(input_shape, n_actions)
    #ref_policy = BollingerPolicy()
    ref_policy = OracleLookaheadPolicy(env)

    # Train
    rewards = train_ppo(env, actor_critic, ref_policy, args.total_steps, args.beta)
    
    # Final evaluation
    actor_critic.load_state_dict(torch.load("best_model_505.673.pt"))
    # test_reward, test_rewards = evaluate(env, actor_critic, mode='val')
    # print(f"\nTest set evaluation mean reward: {np.exp(test_reward):.3f}") 
    # print(f"Test set rewards: {[np.exp(r) for r in test_rewards]}")

    # evaluate the entire test set 
    # test_data_ranges = {
    #     'train': ('2019-09-01', '2022-12-31'),
    #     'val': ('2023-01-01', '2023-06-30'),
    #     'test': ('2025-01-01', '2025-04-01')
    # }
    # env_test = BitcoinTradingEnv(
    #     csv_path=args.data_path,
    #     window_size=168,  # 7 days * 24 hours
    #     episode_length=2100,
    #     date_ranges_by_mode=test_data_ranges
    # )
    
    test_data_ranges = {
        'train': ('2020-01-01', '2022-12-31'),
        'val': ('2023-01-01', '2023-02-01'),
        'test': ('2023-03-01', '2025-05-01')
    }
    env_test = BitcoinTradingEnv(
        csv_path=args.data_path,
        window_size=168,  # 7 days * 24 hours
        episode_length=18700,
        date_ranges_by_mode=test_data_ranges
    )
    

    test_reward, test_rewards, test_actions = evaluate(env_test, actor_critic, n_episodes=3, mode='test')
    print(f"\nTest set evaluation mean reward: {np.exp(test_reward):.3f}") 
    print(f"Test set rewards: {[np.exp(r[0]) for r in test_rewards]}")
    #print(f"Test set reward changes: {[r[1] for r in test_rewards]}")

    #print(test_actions)

    # action statistics
    action_stats = {}
    for action in test_actions[0]:
        if action not in action_stats:
            action_stats[action] = 0
        action_stats[action] += 1
    print(action_stats)
    
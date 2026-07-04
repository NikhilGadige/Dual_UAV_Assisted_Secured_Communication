"""Generic per-agent DDPG, used for the MADDPG leg of this study.

Architecture (Actor/Critic MLP shapes, OU exploration noise, soft target
updates) is carried over directly from rl/ddpg_train.py — the
already-built single-agent DDPG in this repo — just repackaged behind the
same BaseAgent interface as madrl_updated_exp's MAPPOAgent/MATD3Agent so
one generic trainer loop (common/trainer.py) can drive all three learned
algorithms interchangeably. Each agent keeps its own actor/critic/replay
buffer (independent learners, shared team reward) — the standard
decentralized-actor-decentralized-critic simplification of MADDPG; a
fully centralized critic (all agents' obs+actions) would be the next
step, not implemented here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from madrl_updated_exp.agents.base import BaseAgent


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


class OUNoise:
    def __init__(self, size: int, theta: float = 0.15, dt: float = 1.0):
        self.size, self.theta, self.dt = size, theta, dt
        self.state = np.zeros(size, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(self.size, dtype=np.float32)

    def sample(self, sigma: float) -> np.ndarray:
        dx = self.theta * (-self.state) * self.dt
        dx += sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.state = (self.state + dx).astype(np.float32)
        return self.state.copy()


def soft_update(src: nn.Module, dst: nn.Module, tau: float) -> None:
    for p_src, p_dst in zip(src.parameters(), dst.parameters()):
        p_dst.data.copy_(tau * p_src.data + (1.0 - tau) * p_dst.data)


class DDPGAgent(BaseAgent):
    """Vanilla per-agent DDPG (single critic, no policy delay, OU noise)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        name: str = "agent",
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 64,
        noise_std: float = 0.2,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = name
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.max_grad_norm = max_grad_norm

        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        self.ou_noise = OUNoise(act_dim)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs[None, :]).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()
        if not deterministic:
            action = np.clip(action + self.ou_noise.sample(self.noise_std), -1.0, 1.0)
        return action.astype(np.float64)

    def reset_noise(self):
        self.ou_noise.reset()

    def update(self, buffer_data: dict) -> dict:
        obs = torch.FloatTensor(np.clip(buffer_data["obs"], -1e6, 1e6)).to(self.device)
        actions = torch.FloatTensor(buffer_data["actions"]).to(self.device)
        rewards = torch.FloatTensor(np.clip(buffer_data["rewards"], -1e6, 1e6)).to(self.device)
        next_obs = torch.FloatTensor(np.clip(buffer_data["next_obs"], -1e6, 1e6)).to(self.device)
        dones = torch.FloatTensor(buffer_data["dones"]).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            q_next = self.critic_target(next_obs, next_actions)
            q_target = rewards.unsqueeze(-1) + self.gamma * (1.0 - dones.unsqueeze(-1)) * q_next

        q_pred = self.critic(obs, actions)
        critic_loss = self.mse(q_pred, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_grad_norm": float(critic_grad_norm.item()),
            "actor_grad_norm": float(actor_grad_norm.item()),
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std().item()),
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

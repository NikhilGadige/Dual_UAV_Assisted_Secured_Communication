"""MATD3 (Multi-Agent TD3) implementation."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from madrl_updated_exp.agents.base import BaseAgent


def _orthogonal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.zeros_(m.bias.data)


class Actor(nn.Module):
    """Deterministic policy network."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.pre_tanh = nn.Linear(hidden_dim, act_dim)
        self.tanh = nn.Tanh()
        self.net = nn.Sequential(*layers, self.pre_tanh, self.tanh)
        self.apply(_orthogonal_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def forward_with_logging(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = obs
        for layer in self.net:
            h = layer(h)
            if layer is self.pre_tanh:
                pre_tanh = h
        return h, pre_tanh


class Critic(nn.Module):
    """Centralized Q-function."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim + act_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(_orthogonal_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class MATD3Agent(BaseAgent):
    """MATD3 agent with twin critics and delayed policy updates."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        name: str = "agent",
        hidden_dim: int = 256,
        n_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        policy_delay: int = 2,
        max_grad_norm: float = 10.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = name
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self.max_grad_norm = max_grad_norm
        self.total_steps = 0

        self.actor = Actor(obs_dim, act_dim, hidden_dim, n_layers).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = Critic(obs_dim, act_dim, hidden_dim, n_layers).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2 = Critic(obs_dim, act_dim, hidden_dim, n_layers).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr,
        )

        self.mse = nn.MSELoss()
        self._action_log = []

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs[None, :]).to(self.device)
        with torch.no_grad():
            action_t, pre_tanh_t = self.actor.forward_with_logging(obs_t)
            action = action_t.cpu().numpy().flatten().astype(np.float64)
            pre_tanh = pre_tanh_t.cpu().numpy().flatten().astype(np.float64)
        if not deterministic:
            noise = np.random.randn(len(action)) * self.exploration_noise
            action = np.clip(action + noise, -1.0, 1.0)
        self._action_log.append({
            "pre_tanh": pre_tanh,
            "post_tanh": action.copy(),
        })
        return action

    def get_action_log(self) -> list[dict]:
        return self._action_log

    def clear_action_log(self):
        self._action_log = []

    def compute_saturation(self) -> dict:
        if not self._action_log:
            return {"fraction_saturated": 0.0, "mean_pre_tanh": 0.0, "mean_post_tanh": 0.0}
        all_post = np.concatenate([e["post_tanh"] for e in self._action_log])
        all_pre = np.concatenate([e["pre_tanh"] for e in self._action_log])
        frac_sat = float(np.mean(np.abs(all_post) > 0.95))
        return {
            "fraction_saturated": frac_sat,
            "mean_pre_tanh": float(np.mean(np.abs(all_pre))),
            "mean_post_tanh": float(np.mean(np.abs(all_post))),
            "std_post_tanh": float(np.std(all_post)),
        }

    def update(self, buffer_data: dict) -> dict:
        obs = torch.FloatTensor(np.clip(buffer_data["obs"], -1e6, 1e6)).to(self.device)
        actions = torch.FloatTensor(buffer_data["actions"]).to(self.device)
        rewards = torch.FloatTensor(np.clip(buffer_data["rewards"], -1e6, 1e6)).to(self.device)
        next_obs = torch.FloatTensor(np.clip(buffer_data["next_obs"], -1e6, 1e6)).to(self.device)
        dones = torch.FloatTensor(buffer_data["dones"]).to(self.device)

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(-1.0, 1.0)

            q1_next = self.critic1_target(next_obs, next_actions)
            q2_next = self.critic2_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next)
            q_target = rewards.unsqueeze(-1) + self.gamma * (1.0 - dones.unsqueeze(-1)) * q_next

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = self.mse(q1, q_target) + self.mse(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            self.max_grad_norm,
        )
        self.critic_optimizer.step()

        self.total_steps += 1
        actor_loss = torch.tensor(0.0)
        actor_grad_norm = torch.tensor(0.0)
        if self.total_steps % self.policy_delay == 0:
            actor_loss = -self.critic1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            for target_p, source_p in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_p.data.copy_(self.tau * source_p.data + (1.0 - self.tau) * target_p.data)
            for target_p, source_p in zip(
                self.critic1_target.parameters(), self.critic1.parameters(),
            ):
                target_p.data.copy_(self.tau * source_p.data + (1.0 - self.tau) * target_p.data)
            for target_p, source_p in zip(
                self.critic2_target.parameters(), self.critic2.parameters(),
            ):
                target_p.data.copy_(self.tau * source_p.data + (1.0 - self.tau) * target_p.data)

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
            "critic1": self.critic1.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

    def train_mode(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

"""MAPPO (Multi-Agent PPO) implementation."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from madrl_exp.agents.base import BaseAgent


def _orthogonal_init_default(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
        nn.init.zeros_(m.bias.data)


class ActorCritic(nn.Module):
    """Shared backbone with separate actor and critic heads.

    Supports logit temperature scaling and logit clipping for saturation reduction.
    Uses orthogonal init with per-layer gains (actor_out=0.01, critic=1.0, hidden=sqrt(2)).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, temperature: float = 1.0,
                 clip_logits: bool = False):
        super().__init__()
        self.temperature = temperature
        self.clip_logits = clip_logits

        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal init with per-layer gains
        self.apply(_orthogonal_init_default)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

        # Weight normalization on actor output (decomposes weight into g * v/||v||)
        self.actor_mean = nn.utils.weight_norm(self.actor_mean, name="weight")
        # After weight_norm: weight_g = ||w|| (initialized from ortho init gain=0.01),
        # weight_v = w / ||w|| (unit direction). Training will update both.

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply clipping then temperature scaling before tanh."""
        if self.clip_logits:
            logits = torch.clamp(logits, -10.0, 10.0)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.actor_mean(h)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        value = self.critic(h)
        return mean, std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, value = self.forward(obs)
        if deterministic:
            scaled = self._scale_logits(mean)
            action = torch.tanh(scaled)
            log_prob = None
            pre_tanh = scaled
        else:
            dist = torch.distributions.Normal(mean, std)
            raw = dist.rsample()
            scaled = self._scale_logits(raw)
            action = torch.tanh(scaled)
            log_prob = dist.log_prob(raw).sum(dim=-1)
            log_prob -= torch.log(1.0 - action.pow(2) + 1e-10).sum(dim=-1)
            pre_tanh = scaled
        return action, log_prob, value.squeeze(-1), pre_tanh

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        # Reverse: action = tanh(clip(raw) / T) if clip else tanh(raw / T)
        raw = torch.atanh(torch.clamp(action, -0.999999, 0.999999))
        if self.temperature != 1.0:
            raw = raw * self.temperature
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-10).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.squeeze(-1)


class MAPPOAgent(BaseAgent):
    """MAPPO agent with centralized value function."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        name: str = "agent",
        hidden_dim: int = 256,
        n_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        target_kl: float = 0.01,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        temperature: float = 1.0,
        clip_logits: bool = False,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = name
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.weight_decay = weight_decay
        self.batch_size = 256

        self.model = ActorCritic(obs_dim, act_dim, hidden_dim, n_layers,
                                 temperature, clip_logits).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        self._action_log = []
        self._train_pre_tanh_log = []

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs[None, :]).to(self.device)
        with torch.no_grad():
            action, log_prob, value, pre_tanh = self.model.get_action(obs_t, deterministic)
        action_np = action.cpu().numpy().flatten().astype(np.float64)
        pre_tanh_np = pre_tanh.cpu().numpy().flatten().astype(np.float64)
        self._action_log.append({
            "pre_tanh": pre_tanh_np,
            "post_tanh": action_np,
        })
        if not deterministic:
            self._train_pre_tanh_log.append(pre_tanh_np)
        return action_np

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

    def get_value(self, obs: np.ndarray) -> float:
        obs_t = torch.FloatTensor(obs[None, :]).to(self.device)
        with torch.no_grad():
            _, _, value, _ = self.model.get_action(obs_t, deterministic=True)
        return float(value.cpu().numpy())

    def update(self, buffer_data: dict) -> dict:
        obs = torch.FloatTensor(np.clip(buffer_data["obs"], -1e6, 1e6)).to(self.device)
        actions = torch.FloatTensor(buffer_data["actions"]).to(self.device)
        rewards = torch.FloatTensor(np.clip(buffer_data["rewards"], -1e6, 1e6)).to(self.device)
        dones = torch.FloatTensor(buffer_data["dones"]).to(self.device)
        values = torch.FloatTensor(buffer_data["values"]).to(self.device)

        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        adv_std = advantages.std()
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        # Advantage stats
        adv_mean = float(advantages.mean().item())
        adv_std_val = float(adv_std.item())
        adv_max = float(advantages.max().item())

        # Output layer weight norms
        with torch.no_grad():
            output_weight_norm = self.model.actor_mean.weight.norm().item()
            output_bias_norm = self.model.actor_mean.bias.norm().item()
            # Weight normalization parameters
            if hasattr(self.model.actor_mean, "weight_g"):
                output_weight_g = float(self.model.actor_mean.weight_g.mean().item())
                output_weight_v_norm = float(self.model.actor_mean.weight_v.norm().item())
            else:
                output_weight_g = output_weight_norm
                output_weight_v_norm = 0.0

        # Pre-tanh stats from training log
        if self._train_pre_tanh_log:
            all_pre = np.concatenate(self._train_pre_tanh_log)
            pre_tanh_mean = float(np.mean(np.abs(all_pre)))
            pre_tanh_std = float(np.std(all_pre))
            pre_tanh_max = float(np.max(np.abs(all_pre)))
            self._train_pre_tanh_log = []
        else:
            pre_tanh_mean = 0.0
            pre_tanh_std = 0.0
            pre_tanh_max = 0.0

        n = len(obs)
        idx = np.arange(n)

        approx_kl = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_grad_norm = 0.0
        n_minibatch_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                b_obs = obs[batch]
                b_actions = actions[batch]
                b_advantages = advantages[batch]
                b_returns = returns[batch]

                log_prob, entropy, value = self.model.evaluate(b_obs, b_actions)

                ratio = torch.exp(log_prob - log_prob.detach())
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.mse(value, b_returns)

                l2_penalty = self.weight_decay * self.model.actor_mean.weight.pow(2).sum()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean() + l2_penalty

                self.optimizer.zero_grad()
                loss.backward()
                gn = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_grad_norm += gn.item()
                n_minibatch_updates += 1

                with torch.no_grad():
                    kl = (ratio - 1.0 - torch.log(torch.clamp(ratio, min=1e-8))).mean().item()
                    approx_kl = max(approx_kl, kl)

        n_updates = max(n_minibatch_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": approx_kl,
            "grad_norm": total_grad_norm / n_updates,
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std().item()),
            "pre_tanh_mean": pre_tanh_mean,
            "pre_tanh_std": pre_tanh_std,
            "pre_tanh_max": pre_tanh_max,
            "adv_mean": adv_mean,
            "adv_std": adv_std_val,
            "adv_max": adv_max,
            "output_weight_norm": output_weight_norm,
            "output_bias_norm": output_bias_norm,
            "output_weight_g": output_weight_g,
            "output_weight_v_norm": output_weight_v_norm,
        }

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1] * (1.0 - dones[t])
            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, device=self.device)

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

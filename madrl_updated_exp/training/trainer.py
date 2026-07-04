"""Centralized training loop for MADRL."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

from madrl_updated_exp.configs import MADRLConfig
from madrl_updated_exp.environment import ISACMultiAgentEnv
from madrl_updated_exp.agents.mappo import MAPPOAgent
from madrl_updated_exp.agents.matd3 import MATD3Agent


class MARLTrainer:
    """Trains multiple agents in the ISAC environment."""

    def __init__(self, cfg: MADRLConfig):
        self.cfg = cfg
        self.env = ISACMultiAgentEnv(cfg.env, cfg.reward, cfg.training.seed)
        self.agents = {}
        self._init_agents()
        self._make_dirs()
        self.history = defaultdict(list)
        self.buffer = {name: [] for name in self.env.agent_names}
        self.loss_history_rows = []

        np.random.seed(cfg.training.seed)

    def _init_agents(self):
        cfg = self.cfg
        for ac in cfg.agents:
            name = ac.name
            obs_dim = self.env.observation_spaces[name].shape[0]
            act_dim = self.env.action_spaces[name].shape[0]

            if cfg.training.algorithm == "mappo":
                agent = MAPPOAgent(
                    obs_dim=obs_dim, act_dim=act_dim, name=name,
                    hidden_dim=ac.hidden_dim, n_layers=ac.n_layers,
                    lr=ac.lr, gamma=ac.gamma,
                    clip_epsilon=ac.clip_epsilon, target_kl=ac.target_kl,
                    entropy_coef=ac.entropy_coef, value_coef=ac.value_coef,
                    max_grad_norm=ac.max_grad_norm, ppo_epochs=ac.ppo_epochs,
                    temperature=ac.temperature, clip_logits=ac.clip_logits,
                    weight_decay=ac.weight_decay, device="cpu",
                )
            else:
                agent = MATD3Agent(
                    obs_dim=obs_dim, act_dim=act_dim, name=name,
                    hidden_dim=ac.hidden_dim, n_layers=ac.n_layers,
                    lr=ac.lr, gamma=ac.gamma, tau=ac.tau,
                    batch_size=ac.batch_size,
                    policy_noise=ac.policy_noise, noise_clip=ac.noise_clip,
                    exploration_noise=ac.exploration_noise,
                    max_grad_norm=ac.max_grad_norm,
                    device="cpu",
                )
            self.agents[name] = agent

    def _make_dirs(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            self.cfg.output_root, f"{self.cfg.training.algorithm}_{ts}",
        )
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.log_dir = os.path.join(self.run_dir, "logs")
        self.csv_dir = os.path.join(self.run_dir, "csv")
        self.eval_dir = os.path.join(self.run_dir, "evaluation")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

    def train(self):
        cfg = self.cfg
        n_ep = cfg.training.n_episodes
        max_steps = cfg.training.max_steps_per_episode
        algo = cfg.training.algorithm

        for ep in range(n_ep):
            obs, _ = self.env.reset()
            ep_rewards = {name: [] for name in self.agents}
            ep_infos = []

            for step in range(max_steps):
                actions = {}
                for name, agent in self.agents.items():
                    agent.train_mode()
                    act = agent.act(obs[name])
                    actions[name] = act

                next_obs, rewards, terminated, truncated, info = self.env.step(actions)

                for name in self.agents:
                    obs_safe = np.clip(obs[name].copy(), -1e6, 1e6)
                    next_obs_safe = np.clip(next_obs[name].copy(), -1e6, 1e6)
                    reward_safe = np.clip(rewards[name], -1e6, 1e6)
                    self.buffer[name].append({
                        "obs": obs_safe,
                        "action": actions[name].copy(),
                        "reward": reward_safe,
                        "next_obs": next_obs_safe,
                        "done": terminated[name] or truncated[name],
                    })
                    ep_rewards[name].append(reward_safe)
                ep_infos.append(info)

                obs = next_obs

                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    break

            for name, agent in self.agents.items():
                if len(self.buffer[name]) >= agent.batch_size:
                    batch = self._sample_buffer(name)
                    stats = agent.update(batch)
                    for k, v in stats.items():
                        self.history[f"{name}/{k}"].append(v)
                    loss_row = {"episode": ep + 1, "agent": name}
                    for k, v in stats.items():
                        loss_row[k] = v
                    self.loss_history_rows.append(loss_row)

            avg_reward = {name: float(np.mean(rw)) for name, rw in ep_rewards.items()}
            avg_secrecy = float(np.mean([inf["secrecy"] for inf in ep_infos]))
            avg_sensing = float(np.mean([inf["sensing"] for inf in ep_infos]))
            avg_violation = float(np.mean([inf["violation"] for inf in ep_infos]))

            for name in self.agents:
                self.history[f"{name}/reward"].append(avg_reward[name])
            self.history["secrecy"].append(avg_secrecy)
            self.history["sensing"].append(avg_sensing)
            self.history["violation"].append(avg_violation)

            if (ep + 1) % cfg.training.log_interval == 0:
                self._log(ep, n_ep, avg_reward, avg_secrecy, avg_sensing)

            if (ep + 1) % cfg.training.eval_interval == 0:
                eval_metrics = self.evaluate(n_episodes=cfg.training.n_eval_episodes)
                self._log_eval(ep, eval_metrics)
                for k, v in eval_metrics.items():
                    self.history[f"eval/{k}"].append(v)

            if (ep + 1) % cfg.training.save_interval == 0:
                self.save_checkpoint(ep + 1)

        self.save_checkpoint(n_ep)
        self.save_history()
        self.save_csvs()
        print(f"Training complete. Results in {self.run_dir}")

    def _sample_buffer(self, name: str) -> dict:
        B = self.agents[name].batch_size
        data = self.buffer[name]
        if len(data) < B:
            return self._buffer_to_dict(data)
        idx = np.random.choice(len(data), B, replace=False)
        batch = [data[i] for i in idx]
        return self._buffer_to_dict(batch)

    def _buffer_to_dict(self, batch: list) -> dict:
        return {
            "obs": np.array([b["obs"] for b in batch]),
            "actions": np.array([b["action"] for b in batch]),
            "rewards": np.array([b["reward"] for b in batch]),
            "next_obs": np.array([b["next_obs"] for b in batch]),
            "dones": np.array([b["done"] for b in batch]),
            "values": np.zeros(len(batch)),
        }

    def evaluate(self, n_episodes: int = 5) -> dict:
        for agent in self.agents.values():
            agent.eval_mode()

        rewards = []
        secrecies = []
        sensings = []
        violations = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_rew = []
            for _ in range(self.cfg.training.max_steps_per_episode):
                actions = {}
                for name, agent in self.agents.items():
                    act = agent.act(obs[name], deterministic=True)
                    actions[name] = act
                obs, rew_dict, terminated, truncated, info = self.env.step(actions)
                ep_rew.append(float(rew_dict[self.env.agent_names[0]]))
                if terminated.get("__all__", False):
                    break
            rewards.append(float(np.mean(ep_rew)))
            secrecies.append(float(info.get("secrecy", 0.0)))
            sensings.append(float(info.get("sensing", 0.0)))
            violations.append(float(info.get("violation", 0.0)))

        for agent in self.agents.values():
            agent.train_mode()

        return {
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_secrecy": float(np.mean(secrecies)),
            "avg_sensing": float(np.mean(sensings)),
            "avg_violation": float(np.mean(violations)),
        }

    def _log(self, ep, n_ep, avg_reward, avg_secrecy, avg_sensing):
        msg = f"Ep {ep+1}/{n_ep} | "
        for name, rew in avg_reward.items():
            msg += f"{name}: {rew:.4f} | "
        msg += f"Secrecy: {avg_secrecy:.4f} | Sensing: {avg_sensing:.4f}"
        print(msg)

    def _log_eval(self, ep, metrics):
        msg = f"  Eval @ ep {ep+1} | Reward: {metrics['avg_reward']:.4f} +/- {metrics['std_reward']:.4f}"
        msg += f" | Secrecy: {metrics['avg_secrecy']:.4f} | Sensing: {metrics['avg_sensing']:.4f}"
        print(msg)

    def save_checkpoint(self, episode: int):
        for name, agent in self.agents.items():
            path = os.path.join(self.ckpt_dir, f"{name}_ep{episode}.pt")
            agent.save(path)

    def save_history(self):
        path = os.path.join(self.log_dir, "history.json")
        serializable = {}
        for k, v in self.history.items():
            serializable[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def save_csvs(self):
        names = list(self.agents.keys())
        n_ep = len(self.history.get(f"{names[0]}/reward", []))

        rows_reward = []
        rows_secrecy = []
        rows_sensing = []
        rows_constraint = []
        for ep in range(n_ep):
            row_rew = {"episode": ep + 1}
            row_sec = {"episode": ep + 1}
            row_sen = {"episode": ep + 1}
            row_con = {"episode": ep + 1}
            for name in names:
                vals = self.history.get(f"{name}/reward", [])
                row_rew[name] = vals[ep] if ep < len(vals) else 0.0
            row_sec["secrecy"] = self.history["secrecy"][ep] if ep < len(self.history["secrecy"]) else 0.0
            row_sen["sensing"] = self.history["sensing"][ep] if ep < len(self.history["sensing"]) else 0.0
            row_con["violation"] = self.history["violation"][ep] if ep < len(self.history["violation"]) else 0.0
            rows_reward.append(row_rew)
            rows_secrecy.append(row_sec)
            rows_sensing.append(row_sen)
            rows_constraint.append(row_con)

        self._write_csv(os.path.join(self.csv_dir, "episode_rewards.csv"), rows_reward)
        self._write_csv(os.path.join(self.csv_dir, "episode_secrecy.csv"), rows_secrecy)
        self._write_csv(os.path.join(self.csv_dir, "episode_sensing.csv"), rows_sensing)
        self._write_csv(os.path.join(self.csv_dir, "episode_constraints.csv"), rows_constraint)
        self.save_loss_csv()

    def save_loss_csv(self):
        if not self.loss_history_rows:
            return
        fieldnames = list(self.loss_history_rows[0].keys())
        path = os.path.join(self.csv_dir, "loss_history.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.loss_history_rows)

    def _write_csv(self, path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

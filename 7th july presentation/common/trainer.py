"""Generic multi-agent training loop for the sensing study.

Mirrors madrl_updated_exp/training/trainer.py's structure (same growing
replay buffer / per-episode update pattern, generic across any agent that
implements the BaseAgent interface: act(), update(), save(), train_mode(),
eval_mode()) but logs CRB + Pd instead of secrecy + sensing, since this
folder is about the sensing-only half of the ISAC problem.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np

from common.sensing_env import MultiAgentSensingEnv, SensingEnvConfig


class SensingTrainer:
    def __init__(
        self,
        env_cfg: SensingEnvConfig,
        agents: dict,
        n_episodes: int,
        output_root: str,
        run_name: str,
        eval_interval: int = 20,
        n_eval_episodes: int = 5,
        log_interval: int = 10,
        save_interval: int = 50,
    ):
        self.env = MultiAgentSensingEnv(env_cfg)
        self.agents = agents
        self.n_episodes = n_episodes
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.log_interval = log_interval
        self.save_interval = save_interval

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_root, f"{run_name}_{ts}")
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.csv_dir = os.path.join(self.run_dir, "csv")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        self.buffer = {name: [] for name in self.env.agent_names}
        self.history = defaultdict(list)
        self.rows: list[dict] = []

    def train(self):
        for ep in range(1, self.n_episodes + 1):
            obs, _ = self.env.reset()
            for agent in self.agents.values():
                if hasattr(agent, "reset_noise"):
                    agent.reset_noise()

            ep_crb, ep_pd, ep_reward = [], [], []
            for _ in range(self.env.cfg.steps_per_episode):
                actions = {}
                for name, agent in self.agents.items():
                    agent.train_mode()
                    actions[name] = agent.act(obs[name])

                next_obs, rewards, terminated, truncated, info = self.env.step(actions)

                for name in self.agents:
                    self.buffer[name].append({
                        "obs": obs[name].copy(),
                        "action": actions[name].copy(),
                        "reward": float(np.clip(rewards[name], -1e6, 1e6)),
                        "next_obs": next_obs[name].copy(),
                        "done": bool(terminated[name] or truncated[name]),
                    })
                ep_crb.append(info["crb_mean"])
                ep_pd.append(info["pd_mean"])
                ep_reward.append(info["reward"])

                obs = next_obs
                if truncated.get("__all__", False):
                    break

            loss_stats = {}
            for name, agent in self.agents.items():
                if len(self.buffer[name]) >= agent.batch_size:
                    batch = self._sample_buffer(name, agent.batch_size)
                    loss_stats[name] = agent.update(batch)

            avg_crb = float(np.mean(ep_crb))
            avg_pd = float(np.mean(ep_pd))
            avg_reward = float(np.mean(ep_reward))
            self.history["crb"].append(avg_crb)
            self.history["pd"].append(avg_pd)
            self.history["reward"].append(avg_reward)

            row = {"episode": ep, "avg_crb": avg_crb, "avg_pd": avg_pd, "avg_reward": avg_reward}
            for name, stats in loss_stats.items():
                for k, v in stats.items():
                    row[f"{name}/{k}"] = v
            self.rows.append(row)

            if ep % self.log_interval == 0 or ep == 1 or ep == self.n_episodes:
                print(f"Ep {ep:4d}/{self.n_episodes} | avg_CRB={avg_crb:.5f} | avg_Pd={avg_pd:.3f} | reward={avg_reward:.3f}")

            if ep % self.save_interval == 0 or ep == self.n_episodes:
                self.save_checkpoints(ep)

        self.save_csv()
        return self.run_dir

    def _sample_buffer(self, name: str, batch_size: int) -> dict:
        data = self.buffer[name]
        idx = np.random.choice(len(data), min(batch_size, len(data)), replace=False)
        batch = [data[i] for i in idx]
        return {
            "obs": np.array([b["obs"] for b in batch]),
            "actions": np.array([b["action"] for b in batch]),
            "rewards": np.array([b["reward"] for b in batch]),
            "next_obs": np.array([b["next_obs"] for b in batch]),
            "dones": np.array([b["done"] for b in batch]),
            "values": np.zeros(len(batch)),
        }

    def save_checkpoints(self, episode: int):
        for name, agent in self.agents.items():
            agent.save(os.path.join(self.ckpt_dir, f"{name}_ep{episode}.pt"))

    def save_csv(self):
        path = os.path.join(self.csv_dir, "training_log.csv")
        if not self.rows:
            return
        fieldnames = sorted({k for r in self.rows for k in r.keys()}, key=lambda k: (k != "episode", k))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.rows)
        with open(os.path.join(self.run_dir, "history.json"), "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in self.history.items()}, f, indent=2)
        return path

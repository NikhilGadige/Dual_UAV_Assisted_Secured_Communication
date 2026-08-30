#!/usr/bin/env python3
"""
Final System Model RL Convergence Analysis Script.
Trains and compares:
- MAPPO (Proposed Multi-Agent PPO)
- MATD3PG (Multi-Agent Twin Delayed DDPG)
- SASAC (Single-Agent Soft Actor-Critic)
- SATD3PG (Single-Agent Twin Delayed DDPG)
- Random Walk (Baseline)
on the Week 9 Semantic-Aware ISAC Network with PD-NOMA and Elevation-Dependent Path Loss.
Logs metrics to CSV and plots convergence curves under output_final/.
"""

import os
import sys
import time
import csv
import random
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Insert the Week 9 work directory into path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Week9 work'))
from system_model import Week9SystemModel

# Setup output directory
OUTPUT_DIR = "output_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LAMBDA1_ASSR = 0.5
LAMBDA2_PD = 0.5
LAMBDA1_PD_SENSING = 0.5
LAMBDA2_PD_SENSING = 0.5
CRB_THRESHOLD = 1.0e8


def robust_max_reference(values):
    """Return the maximum secrecy value for stable ASSR scaling."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    ref = float(np.max(arr))
    return ref if ref > 1e-12 else 1.0


def compute_assr_pd_utility(secrecy_vals, pd_target_vals, pd_eaves_vals, crb_target_vals):
    secrecy_arr = np.asarray(secrecy_vals, dtype=float)
    pd_target_arr = np.asarray(pd_target_vals, dtype=float)
    pd_eaves_arr = np.asarray(pd_eaves_vals, dtype=float)
    crb_target_arr = np.asarray(crb_target_vals, dtype=float)

    r_ref = robust_max_reference(secrecy_arr)
    assr_vals = np.clip(secrecy_arr / r_ref, 0.0, 1.0)
    pd_combined_vals = LAMBDA1_PD_SENSING * pd_eaves_arr + LAMBDA2_PD_SENSING * pd_target_arr
    crb_feasible_vals = (crb_target_arr <= CRB_THRESHOLD).astype(float)
    utility_vals = (LAMBDA1_ASSR * assr_vals + LAMBDA2_PD * pd_combined_vals) * crb_feasible_vals
    return assr_vals, pd_combined_vals, utility_vals, crb_feasible_vals, r_ref

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 1. Custom Gymnasium-Style Environment for Week 9 System Model ---
class Week9ISACEnv:
    def __init__(self, seed=42, eve_uncertainty_radius=15.0):
        self.sys_model = Week9SystemModel(
            p_bs_tx=2.0,
            p_jam_tx=0.5,
            n_ris_elements=64,
            noma_a_far=0.7,
            eve_uncertainty_radius=eve_uncertainty_radius,
        )
        self.eve_uncertainty_radius = eve_uncertainty_radius
        self.lambda1_pd = LAMBDA1_PD_SENSING
        self.lambda2_pd = LAMBDA2_PD_SENSING
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Scenario limits
        self.x_min, self.x_max = 0.0, 200.0
        self.y_min, self.y_max = 0.0, 200.0
        self.z_min, self.z_max = 30.0, 100.0
        
        # Max movement increments per step
        self.max_uav_step = 5.0  # meters
        self.dt = 1.0  # seconds
        
        # Mobile User Random Walk parameters
        self.user_speed = 3.0  # m/s
        self.user_heading = 0.0
        
        # Define positions of static nodes
        self.pos_BS = np.array([0.0, 0.0, 15.0])
        self.pos_T = np.array([40.0, 45.0, 1.5])
        self.pos_E1 = np.array([70.0, 80.0, 0.0])
        self.pos_E2 = np.array([90.0, 100.0, 0.0])
        self.pos_E3 = np.array([110.0, 110.0, 0.0])
        
        # Define dynamic nodes
        self.pos_R = np.array([50.0, 50.0, 80.0])
        self.pos_J = np.array([100.0, 120.0, 60.0])
        self.pos_U = np.array([150.0, 180.0, 1.5])
        
        # Define configurable powers
        self.p_bs_tx = 2.0
        self.p_jam_tx = 0.5
        self.noma_a_far = 0.7
        
        # Dimensions
        self.state_dim = 24
        # 3 actions per agent, 3 agents
        # Agent 1 (ris_uav): dx, dy, dz for R
        # Agent 2 (jammer_uav): dx, dy, dz for J
        # Agent 3 (power_alloc): p_bs_tx_act, p_jam_tx_act, noma_a_far_act
        self.action_dim = 9
        self._objective_secrecy_history = []

    def reset_objective_tracking(self):
        self._objective_secrecy_history = []
        
    def reset(self):
        # Reset positions
        self.pos_R = np.array([50.0, 50.0, 80.0])
        self.pos_J = np.array([100.0, 120.0, 60.0])
        self.pos_U = np.array([150.0, 180.0, 1.5])
        
        self.user_heading = self.rng.uniform(0.0, 2.0 * np.pi)
        
        # Reset powers
        self.p_bs_tx = 2.0
        self.p_jam_tx = 0.5
        self.noma_a_far = 0.7
        
        # Sync system model
        self.sys_model.nodes["R"].position = self.pos_R
        self.sys_model.nodes["J"].position = self.pos_J
        self.sys_model.nodes["U"].position = self.pos_U
        self.sys_model.p_bs_tx = self.p_bs_tx
        self.sys_model.p_jam_tx = self.p_jam_tx
        self.sys_model.noma_engine.power_alloc_far = self.noma_a_far
        self.sys_model.eve_uncertainty_radius = self.eve_uncertainty_radius
        
        return self._get_obs()
        
    def _get_obs(self):
        obs = np.array([
            self.pos_R[0] / 100.0 - 1.0, self.pos_R[1] / 100.0 - 1.0, (self.pos_R[2] - 65.0) / 35.0,
            self.pos_J[0] / 100.0 - 1.0, self.pos_J[1] / 100.0 - 1.0, (self.pos_J[2] - 65.0) / 35.0,
            self.pos_T[0] / 100.0 - 1.0, self.pos_T[1] / 100.0 - 1.0, (self.pos_T[2] - 65.0) / 35.0,
            self.pos_U[0] / 100.0 - 1.0, self.pos_U[1] / 100.0 - 1.0, (self.pos_U[2] - 65.0) / 35.0,
            self.pos_E1[0] / 100.0 - 1.0, self.pos_E1[1] / 100.0 - 1.0, (self.pos_E1[2] - 65.0) / 35.0,
            self.pos_E2[0] / 100.0 - 1.0, self.pos_E2[1] / 100.0 - 1.0, (self.pos_E2[2] - 65.0) / 35.0,
            self.pos_E3[0] / 100.0 - 1.0, self.pos_E3[1] / 100.0 - 1.0, (self.pos_E3[2] - 65.0) / 35.0,
            (self.p_bs_tx - 2.75) / 2.25, (self.p_jam_tx - 0.525) / 0.475, (self.noma_a_far - 0.75) / 0.20
        ], dtype=np.float32)
        return obs
        
    def step(self, action):
        # Handle dict format (multi-agent) or flat vector format (single-agent)
        if isinstance(action, dict):
            act_R = action.get("ris_uav", np.zeros(3))
            act_J = action.get("jammer_uav", np.zeros(3))
            act_P = action.get("power_alloc", np.zeros(3))
        else:
            act_R = action[0:3]
            act_J = action[3:6]
            act_P = action[6:9]
            
        # Update dynamic positions (R and J)
        self.pos_R += np.clip(act_R, -1.0, 1.0) * self.max_uav_step
        self.pos_R[0] = np.clip(self.pos_R[0], self.x_min, self.x_max)
        self.pos_R[1] = np.clip(self.pos_R[1], self.y_min, self.y_max)
        self.pos_R[2] = np.clip(self.pos_R[2], self.z_min, self.z_max)
        
        self.pos_J += np.clip(act_J, -1.0, 1.0) * self.max_uav_step
        self.pos_J[0] = np.clip(self.pos_J[0], self.x_min, self.x_max)
        self.pos_J[1] = np.clip(self.pos_J[1], self.y_min, self.y_max)
        self.pos_J[2] = np.clip(self.pos_J[2], self.z_min, self.z_max)
        
        # Update powers
        # Map actions in [-1, 1] to physical values
        self.p_bs_tx = float(0.5 + (np.clip(act_P[0], -1.0, 1.0) + 1.0) * 2.25)   # range [0.5, 5.0] W
        self.p_jam_tx = float(0.05 + (np.clip(act_P[1], -1.0, 1.0) + 1.0) * 0.475) # range [0.05, 1.0] W
        self.noma_a_far = float(0.55 + (np.clip(act_P[2], -1.0, 1.0) + 1.0) * 0.20) # range [0.55, 0.95]
        
        # Mobile User Random Walk update
        self.user_heading += self.rng.normal(0.0, 0.5)
        self.pos_U[0] += self.user_speed * np.cos(self.user_heading) * self.dt
        self.pos_U[1] += self.user_speed * np.sin(self.user_heading) * self.dt
        
        # Reflect at boundaries
        if self.pos_U[0] < self.x_min:
            self.pos_U[0] = self.x_min
            self.user_heading = np.pi - self.user_heading
        elif self.pos_U[0] > self.x_max:
            self.pos_U[0] = self.x_max
            self.user_heading = np.pi - self.user_heading
            
        if self.pos_U[1] < self.y_min:
            self.pos_U[1] = self.y_min
            self.user_heading = -self.user_heading
        elif self.pos_U[1] > self.y_max:
            self.pos_U[1] = self.y_max
            self.user_heading = -self.user_heading
            
        # Update system model positions and configurations
        self.sys_model.nodes["R"].position = self.pos_R
        self.sys_model.nodes["J"].position = self.pos_J
        self.sys_model.nodes["U"].position = self.pos_U
        self.sys_model.p_bs_tx = self.p_bs_tx
        self.sys_model.p_jam_tx = self.p_jam_tx
        self.sys_model.noma_engine.power_alloc_far = self.noma_a_far
        self.sys_model.eve_uncertainty_radius = self.eve_uncertainty_radius
        
        # Evaluate system performance
        results = self.sys_model.evaluate_system()
        
        # Extract variables for the shared objective
        secrecy_total = results["secrecy_performance"]["secrecy_rate_total"] # suts/s
        pd_target = results["sensing_performance"]["Target_T"]["sensing_accuracy"]
        pd_eaves = np.mean([eve["sensing_accuracy"] for eve in results["sensing_performance"]["Eavesdroppers"].values()])
        crb_target = results["sensing_performance"]["Target_T"]["crb"]

        self._objective_secrecy_history.append(float(secrecy_total))
        secrecy_ref = robust_max_reference(self._objective_secrecy_history)
        assr = float(np.clip(secrecy_total / secrecy_ref, 0.0, 1.0))
        pd_combined = float(self.lambda1_pd * pd_eaves + self.lambda2_pd * pd_target)

        # Shared objective for every algorithm:
        # reward = (0.5 * ASSR + 0.5 * Pd) with CRB as a feasibility constraint.
        crb_satisfied = crb_target <= CRB_THRESHOLD
        reward = float((LAMBDA1_ASSR * assr + LAMBDA2_PD * pd_combined) * float(crb_satisfied))

        pd_satisfied = pd_target >= 0.0 and all(eve["sensing_accuracy"] >= 0.0 for eve in results["sensing_performance"]["Eavesdroppers"].values())
        joint_satisfied = crb_satisfied and pd_satisfied
        
        info = {
            "reward": reward,
            "assr": assr,
            "pd_combined": pd_combined,
            "secrecy_total": secrecy_total,
            "secrecy_target": results["secrecy_performance"]["secrecy_rate_target"],
            "secrecy_user": results["secrecy_performance"]["secrecy_rate_user"],
            "pd_target": pd_target,
            "pd_eaves": pd_eaves,
            "crb_target": crb_target,
            "crb_feasible": float(crb_satisfied),
            "p_bs_tx": self.p_bs_tx,
            "p_jam_tx": self.p_jam_tx,
            "noma_a_far": self.noma_a_far,
            "joint_satisfied": float(joint_satisfied)
        }
        
        obs = self._get_obs()
        return obs, reward, False, False, info

# --- 2. Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(reward, dtype=np.float32).reshape(-1, 1),
                np.array(next_state, dtype=np.float32),
                np.array(done, dtype=np.float32).reshape(-1, 1))
                
    def __len__(self):
        return len(self.buffer)

# --- 3. Neural Network Architectures ---
class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    def forward(self, state):
        return self.net(state)

class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        feats = self.trunk(state)
        mean = self.mean(feats)
        log_std = torch.clamp(self.log_std(feats), -5.0, 2.0)
        return mean, log_std

    def sample(self, state, reparameterize=True):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        if reparameterize:
            x_t = normal.rsample()
        else:
            x_t = normal.sample()
            
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)
        
    def deterministic(self, state):
        mean, _ = self(state)
        return torch.tanh(mean)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        return self.net(state)

# --- 4. Training Loop Helpers ---

# Helper to save metrics history
def save_history_to_csv(history, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    secrecy_vals = [row[2] for row in history]
    pd_target_vals = [row[5] for row in history]
    pd_eaves_vals = [row[6] for row in history]
    crb_target_vals = [row[7] for row in history]
    assr_vals, pd_combined_vals, utility_vals, crb_feasible_vals, _ = compute_assr_pd_utility(
        secrecy_vals, pd_target_vals, pd_eaves_vals, crb_target_vals,
    )
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "secrecy_total", "secrecy_target", "secrecy_user",
            "pd_target", "pd_eaves", "crb_target", "p_bs_tx", "p_jam_tx",
            "noma_a_far", "constraint_satisfaction", "assr", "pd_combined",
            "combined_utility", "crb_feasible"
        ])
        for idx, row in enumerate(history):
            writer.writerow(list(row) + [
                float(assr_vals[idx]),
                float(pd_combined_vals[idx]),
                float(utility_vals[idx]),
                float(crb_feasible_vals[idx]),
            ])
    print(f"Saved convergence history to {filepath}")

# Helper to generate individual algorithm convergence plots
def plot_individual_convergence(history, algo_name):
    episodes = [row[0] for row in history]
    secrecy_vals = [row[2] for row in history]
    pd_target_vals = [row[5] for row in history]
    pd_eaves_vals = [row[6] for row in history]
    crb_target_vals = [row[7] for row in history]
    assr_vals, pd_combined_vals, utility_vals, _, _ = compute_assr_pd_utility(
        secrecy_vals, pd_target_vals, pd_eaves_vals, crb_target_vals,
    )
    
    # Calculate rolling 100 average
    def rolling_average(data, window=100):
        res = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            res.append(np.mean(data[start:i+1]))
        return res
        
    assr_roll = rolling_average(assr_vals, 100)
    pd_roll = rolling_average(pd_combined_vals, 100)
    utility_roll = rolling_average(utility_vals, 100)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # ASSR plot
    axs[0].plot(episodes, assr_vals, color='green', alpha=0.3, label='Episodic ASSR')
    axs[0].plot(episodes, assr_roll, color='green', linestyle='--', linewidth=2, label='Rolling 100 ASSR')
    axs[0].set_ylabel('ASSR')
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_title(f'{algo_name} Objective Convergence Curves')
    
    # Pd plot
    axs[1].plot(episodes, pd_combined_vals, color='red', alpha=0.3, label='Episodic Pd')
    axs[1].plot(episodes, pd_roll, color='red', linestyle='--', linewidth=2, label='Rolling 100 Pd')
    axs[1].set_ylabel('Pd')
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].grid(True)
    axs[1].legend()
    
    # Combined utility plot
    axs[2].plot(episodes, utility_vals, color='purple', alpha=0.3, label='Episodic 0.5*ASSR + 0.5*Pd')
    axs[2].plot(episodes, utility_roll, color='purple', linestyle='--', linewidth=2, label='Rolling 100 Combined')
    axs[2].set_ylabel('Combined Utility')
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].set_xlabel('Episode')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{algo_name.lower().replace(' ', '_')}_convergence.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved convergence plot to {plot_path}")

    # Dedicated combined utility plot
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, utility_vals, color='purple', alpha=0.3, label='Episodic 0.5*ASSR + 0.5*Pd')
    plt.plot(episodes, utility_roll, color='purple', linewidth=2, label='Rolling 100 Combined Utility')
    plt.title(f'{algo_name} Combined Utility Convergence')
    plt.xlabel('Episode')
    plt.ylabel(r'Utility (($0.5$ ASSR $+\ 0.5$ Pd) $\times$ CRB-feasible)')
    plt.grid(True)
    plt.legend()
    
    utility_plot_path = os.path.join(OUTPUT_DIR, f"{algo_name.lower().replace(' ', '_')}_utility_convergence.png")
    plt.savefig(utility_plot_path, dpi=150)
    plt.close()
    print(f"Saved utility convergence plot to {utility_plot_path}")

# (A) Random Walk Baseline
def run_random_walk(env, num_episodes=150, steps_per_episode=50):
    print("\n--- Running Random Walk Baseline ---")
    env.reset_objective_tracking()
    history = []
    
    for ep in range(num_episodes):
        env.reset()
        ep_reward = 0.0
        metrics = {"secrecy_total": [], "secrecy_target": [], "secrecy_user": [],
                   "pd_target": [], "pd_eaves": [], "crb_target": [],
                   "p_bs_tx": [], "p_jam_tx": [], "noma_a_far": [], "joint_satisfied": []}
                   
        for step in range(steps_per_episode):
            # Sample random action
            action = np.random.uniform(-1.0, 1.0, 9)
            obs, reward, _, _, info = env.step(action)
            ep_reward += reward
            for k in metrics:
                metrics[k].append(info[k])
                
        avg_row = [
            ep + 1, ep_reward,
            np.mean(metrics["secrecy_total"]), np.mean(metrics["secrecy_target"]), np.mean(metrics["secrecy_user"]),
            np.mean(metrics["pd_target"]), np.mean(metrics["pd_eaves"]), np.mean(metrics["crb_target"]),
            np.mean(metrics["p_bs_tx"]), np.mean(metrics["p_jam_tx"]), np.mean(metrics["noma_a_far"]),
            np.mean(metrics["joint_satisfied"])
        ]
        history.append(avg_row)
        
        if (ep + 1) % 20 == 0:
            roll_reward = np.mean([row[1] for row in history[-100:]])
            roll_secrecy = np.mean([row[2] for row in history[-100:]])
            print(f"Episode {ep+1}/{num_episodes} | Reward: {ep_reward:.2f} (Rolling100: {roll_reward:.2f}) | "
                  f"Secrecy Rate: {avg_row[2]/1e6:.6f} M-suts/s (Rolling100: {roll_secrecy/1e6:.6f} M-suts/s) "
                  f"({avg_row[2]:.2f} suts/s (Rolling100: {roll_secrecy:.2f} suts/s))")
            
    save_history_to_csv(history, "random_walk_history.csv")
    plot_individual_convergence(history, "Random Walk")
    return history

# (B) SATD3PG
def train_satd3pg(env, num_episodes=150, steps_per_episode=50):
    print("\n--- Training SATD3PG ---")
    env.reset_objective_tracking()
    history = []
    
    # Hyperparameters
    batch_size = 64
    gamma = 0.95
    tau = 0.01
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    
    # Models
    actor = DeterministicActor(env.state_dim, env.action_dim)
    actor_target = DeterministicActor(env.state_dim, env.action_dim)
    actor_target.load_state_dict(actor.state_dict())
    
    critic1 = QNetwork(env.state_dim, env.action_dim)
    critic2 = QNetwork(env.state_dim, env.action_dim)
    critic1_target = QNetwork(env.state_dim, env.action_dim)
    critic2_target = QNetwork(env.state_dim, env.action_dim)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())
    
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic1_opt = optim.Adam(critic1.parameters(), lr=3e-4)
    critic2_opt = optim.Adam(critic2.parameters(), lr=3e-4)
    
    buffer = ReplayBuffer(capacity=50000)
    
    total_steps = 0
    
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        metrics = {"secrecy_total": [], "secrecy_target": [], "secrecy_user": [],
                   "pd_target": [], "pd_eaves": [], "crb_target": [],
                   "p_bs_tx": [], "p_jam_tx": [], "noma_a_far": [], "joint_satisfied": []}
                   
        for step in range(steps_per_episode):
            total_steps += 1
            
            # Action selection with exploration noise
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_t).numpy()[0]
            action = action + np.random.normal(0, 0.15, size=env.action_dim)
            action = np.clip(action, -1.0, 1.0)
            
            next_state, reward, _, _, info = env.step(action)
            buffer.add(state, action, reward, next_state, False)
            
            state = next_state
            ep_reward += reward
            for k in metrics:
                metrics[k].append(info[k])
                
            # Perform training step
            if len(buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = buffer.sample(batch_size)
                
                states_bt = torch.FloatTensor(states_b)
                actions_bt = torch.FloatTensor(actions_b)
                rewards_bt = torch.FloatTensor(rewards_b)
                next_states_bt = torch.FloatTensor(next_states_b)
                
                # Target actions with target smoothing noise
                with torch.no_grad():
                    noise = torch.randn_like(actions_bt) * policy_noise
                    noise = torch.clamp(noise, -noise_clip, noise_clip)
                    next_actions_bt = torch.clamp(actor_target(next_states_bt) + noise, -1.0, 1.0)
                    
                    target_q1 = critic1_target(next_states_bt, next_actions_bt)
                    target_q2 = critic2_target(next_states_bt, next_actions_bt)
                    target_q = rewards_bt + gamma * torch.min(target_q1, target_q2)
                    
                # Update Critics
                current_q1 = critic1(states_bt, actions_bt)
                current_q2 = critic2(states_bt, actions_bt)
                
                critic1_loss = nn.MSELoss()(current_q1, target_q)
                critic2_loss = nn.MSELoss()(current_q2, target_q)
                
                critic1_opt.zero_grad()
                critic1_loss.backward()
                critic1_opt.step()
                
                critic2_opt.zero_grad()
                critic2_loss.backward()
                critic2_opt.step()
                
                # Delayed policy updates
                if total_steps % policy_freq == 0:
                    actor_loss = -critic1(states_bt, actor(states_bt)).mean()
                    
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()
                    
                    # Soft updates of target networks
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                    for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                    for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                        
        avg_row = [
            ep + 1, ep_reward,
            np.mean(metrics["secrecy_total"]), np.mean(metrics["secrecy_target"]), np.mean(metrics["secrecy_user"]),
            np.mean(metrics["pd_target"]), np.mean(metrics["pd_eaves"]), np.mean(metrics["crb_target"]),
            np.mean(metrics["p_bs_tx"]), np.mean(metrics["p_jam_tx"]), np.mean(metrics["noma_a_far"]),
            np.mean(metrics["joint_satisfied"])
        ]
        history.append(avg_row)
        
        if (ep + 1) % 20 == 0:
            roll_reward = np.mean([row[1] for row in history[-100:]])
            roll_secrecy = np.mean([row[2] for row in history[-100:]])
            print(f"Episode {ep+1}/{num_episodes} | Reward: {ep_reward:.2f} (Rolling100: {roll_reward:.2f}) | "
                  f"Secrecy Rate: {avg_row[2]/1e6:.6f} M-suts/s (Rolling100: {roll_secrecy/1e6:.6f} M-suts/s) "
                  f"({avg_row[2]:.2f} suts/s (Rolling100: {roll_secrecy:.2f} suts/s))")
            
    save_history_to_csv(history, "satd3pg_history.csv")
    plot_individual_convergence(history, "SATD3PG")
    return history

# (C) SASAC
def train_sasac(env, num_episodes=150, steps_per_episode=50):
    print("\n--- Training SASAC ---")
    env.reset_objective_tracking()
    history = []
    
    # Hyperparameters
    batch_size = 64
    gamma = 0.95
    tau = 0.01
    alpha_entropy = 0.2  # Fixed entropy scale
    
    # Models
    actor = GaussianActor(env.state_dim, env.action_dim)
    critic1 = QNetwork(env.state_dim, env.action_dim)
    critic2 = QNetwork(env.state_dim, env.action_dim)
    critic1_target = QNetwork(env.state_dim, env.action_dim)
    critic2_target = QNetwork(env.state_dim, env.action_dim)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())
    
    actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
    critic1_opt = optim.Adam(critic1.parameters(), lr=3e-4)
    critic2_opt = optim.Adam(critic2.parameters(), lr=3e-4)
    
    buffer = ReplayBuffer(capacity=50000)
    
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        metrics = {"secrecy_total": [], "secrecy_target": [], "secrecy_user": [],
                   "pd_target": [], "pd_eaves": [], "crb_target": [],
                   "p_bs_tx": [], "p_jam_tx": [], "noma_a_far": [], "joint_satisfied": []}
                   
        for step in range(steps_per_episode):
            # Select action
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_t, _ = actor.sample(state_t, reparameterize=False)
                action = action_t.numpy()[0]
                
            next_state, reward, _, _, info = env.step(action)
            buffer.add(state, action, reward, next_state, False)
            
            state = next_state
            ep_reward += reward
            for k in metrics:
                metrics[k].append(info[k])
                
            if len(buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, _ = buffer.sample(batch_size)
                
                states_bt = torch.FloatTensor(states_b)
                actions_bt = torch.FloatTensor(actions_b)
                rewards_bt = torch.FloatTensor(rewards_b)
                next_states_bt = torch.FloatTensor(next_states_b)
                
                # Critic loss calculation
                with torch.no_grad():
                    next_actions_bt, next_log_pi = actor.sample(next_states_bt, reparameterize=True)
                    target_q1 = critic1_target(next_states_bt, next_actions_bt)
                    target_q2 = critic2_target(next_states_bt, next_actions_bt)
                    target_q = rewards_bt + gamma * (torch.min(target_q1, target_q2) - alpha_entropy * next_log_pi)
                    
                current_q1 = critic1(states_bt, actions_bt)
                current_q2 = critic2(states_bt, actions_bt)
                
                critic1_loss = nn.MSELoss()(current_q1, target_q)
                critic2_loss = nn.MSELoss()(current_q2, target_q)
                
                critic1_opt.zero_grad()
                critic1_loss.backward()
                critic1_opt.step()
                
                critic2_opt.zero_grad()
                critic2_loss.backward()
                critic2_opt.step()
                
                # Actor update
                curr_actions_bt, log_pi = actor.sample(states_bt, reparameterize=True)
                q1 = critic1(states_bt, curr_actions_bt)
                q2 = critic2(states_bt, curr_actions_bt)
                actor_loss = (alpha_entropy * log_pi - torch.min(q1, q2)).mean()
                
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                
                # Soft target updates
                for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                    
        avg_row = [
            ep + 1, ep_reward,
            np.mean(metrics["secrecy_total"]), np.mean(metrics["secrecy_target"]), np.mean(metrics["secrecy_user"]),
            np.mean(metrics["pd_target"]), np.mean(metrics["pd_eaves"]), np.mean(metrics["crb_target"]),
            np.mean(metrics["p_bs_tx"]), np.mean(metrics["p_jam_tx"]), np.mean(metrics["noma_a_far"]),
            np.mean(metrics["joint_satisfied"])
        ]
        history.append(avg_row)
        
        if (ep + 1) % 20 == 0:
            roll_reward = np.mean([row[1] for row in history[-100:]])
            roll_secrecy = np.mean([row[2] for row in history[-100:]])
            print(f"Episode {ep+1}/{num_episodes} | Reward: {ep_reward:.2f} (Rolling100: {roll_reward:.2f}) | "
                  f"Secrecy Rate: {avg_row[2]/1e6:.6f} M-suts/s (Rolling100: {roll_secrecy/1e6:.6f} M-suts/s) "
                  f"({avg_row[2]:.2f} suts/s (Rolling100: {roll_secrecy:.2f} suts/s))")
            
    save_history_to_csv(history, "sasac_history.csv")
    plot_individual_convergence(history, "SASAC")
    return history

# (D) MATD3PG (Multi-Agent Twin Delayed DDPG)
def train_matd3pg(env, num_episodes=150, steps_per_episode=50):
    print("\n--- Training MATD3PG (Proposed Baseline) ---")
    env.reset_objective_tracking()
    history = []
    
    # 3 agents: ris_uav (3-dim action), jammer_uav (3-dim action), power_alloc (3-dim action)
    agent_dims = [3, 3, 3]
    agent_names = ["ris_uav", "jammer_uav", "power_alloc"]
    
    # Hyperparameters
    batch_size = 64
    gamma = 0.95
    tau = 0.01
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    
    # Individual Actor Networks
    actors = [DeterministicActor(env.state_dim, dim) for dim in agent_dims]
    actors_target = [DeterministicActor(env.state_dim, dim) for dim in agent_dims]
    for a, at in zip(actors, actors_target):
        at.load_state_dict(a.state_dict())
        
    # Twin Centralized Critic Networks
    # Centralized Critic takes global state (24-dim) and concatenated actions of all agents (9-dim)
    critic1 = QNetwork(env.state_dim, sum(agent_dims))
    critic2 = QNetwork(env.state_dim, sum(agent_dims))
    critic1_target = QNetwork(env.state_dim, sum(agent_dims))
    critic2_target = QNetwork(env.state_dim, sum(agent_dims))
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())
    
    actor_opts = [optim.Adam(actor.parameters(), lr=3e-4) for actor in actors]
    critic1_opt = optim.Adam(critic1.parameters(), lr=3e-4)
    critic2_opt = optim.Adam(critic2.parameters(), lr=3e-4)
    
    buffer = ReplayBuffer(capacity=50000)
    total_steps = 0
    
    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        metrics = {"secrecy_total": [], "secrecy_target": [], "secrecy_user": [],
                   "pd_target": [], "pd_eaves": [], "crb_target": [],
                   "p_bs_tx": [], "p_jam_tx": [], "noma_a_far": [], "joint_satisfied": []}
                   
        for step in range(steps_per_episode):
            total_steps += 1
            
            # Action selection for each agent
            state_t = torch.FloatTensor(state).unsqueeze(0)
            actions = []
            for i, actor in enumerate(actors):
                with torch.no_grad():
                    act_indiv = actor(state_t).numpy()[0]
                # Exploration noise
                act_indiv = act_indiv + np.random.normal(0, 0.15, size=agent_dims[i])
                act_indiv = np.clip(act_indiv, -1.0, 1.0)
                actions.append(act_indiv)
                
            # Concatenate for env step
            joint_action = np.concatenate(actions)
            
            next_state, reward, _, _, info = env.step(joint_action)
            buffer.add(state, joint_action, reward, next_state, False)
            
            state = next_state
            ep_reward += reward
            for k in metrics:
                metrics[k].append(info[k])
                
            if len(buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, _ = buffer.sample(batch_size)
                
                states_bt = torch.FloatTensor(states_b)
                actions_bt = torch.FloatTensor(actions_b)
                rewards_bt = torch.FloatTensor(rewards_b)
                next_states_bt = torch.FloatTensor(next_states_b)
                
                # Centralized Critics update
                with torch.no_grad():
                    # Centralized target action: individual target actions with smoothing noise concatenated
                    next_actions_indiv = []
                    for i, act_targ in enumerate(actors_target):
                        target_act = act_targ(next_states_bt)
                        noise = torch.randn_like(target_act) * policy_noise
                        noise = torch.clamp(noise, -noise_clip, noise_clip)
                        next_actions_indiv.append(torch.clamp(target_act + noise, -1.0, 1.0))
                    next_actions_bt = torch.cat(next_actions_indiv, dim=-1)
                    
                    target_q1 = critic1_target(next_states_bt, next_actions_bt)
                    target_q2 = critic2_target(next_states_bt, next_actions_bt)
                    target_q = rewards_bt + gamma * torch.min(target_q1, target_q2)
                    
                current_q1 = critic1(states_bt, actions_bt)
                current_q2 = critic2(states_bt, actions_bt)
                
                critic1_loss = nn.MSELoss()(current_q1, target_q)
                critic2_loss = nn.MSELoss()(current_q2, target_q)
                
                critic1_opt.zero_grad()
                critic1_loss.backward()
                critic1_opt.step()
                
                critic2_opt.zero_grad()
                critic2_loss.backward()
                critic2_opt.step()
                
                # Delayed multi-agent actor updates
                if total_steps % policy_freq == 0:
                    for i in range(len(actors)):
                        # Evaluate joint actions replacing only agent i's action with current policy output
                        joint_actions_eval = []
                        for j in range(len(actors)):
                            if j == i:
                                joint_actions_eval.append(actors[j](states_bt))
                            else:
                                # Slice actions_bt to get agent j's actions from replay buffer
                                start_idx = sum(agent_dims[:j])
                                end_idx = start_idx + agent_dims[j]
                                joint_actions_eval.append(actions_bt[:, start_idx:end_idx])
                        
                        joint_actions_eval_t = torch.cat(joint_actions_eval, dim=-1)
                        actor_loss = -critic1(states_bt, joint_actions_eval_t).mean()
                        
                        actor_opts[i].zero_grad()
                        actor_loss.backward()
                        actor_opts[i].step()
                        
                    # Soft target updates
                    for i in range(len(actors)):
                        for param, target_param in zip(actors[i].parameters(), actors_target[i].parameters()):
                            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                    for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                    for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                        
        avg_row = [
            ep + 1, ep_reward,
            np.mean(metrics["secrecy_total"]), np.mean(metrics["secrecy_target"]), np.mean(metrics["secrecy_user"]),
            np.mean(metrics["pd_target"]), np.mean(metrics["pd_eaves"]), np.mean(metrics["crb_target"]),
            np.mean(metrics["p_bs_tx"]), np.mean(metrics["p_jam_tx"]), np.mean(metrics["noma_a_far"]),
            np.mean(metrics["joint_satisfied"])
        ]
        history.append(avg_row)
        
        if (ep + 1) % 20 == 0:
            roll_reward = np.mean([row[1] for row in history[-100:]])
            roll_secrecy = np.mean([row[2] for row in history[-100:]])
            print(f"Episode {ep+1}/{num_episodes} | Reward: {ep_reward:.2f} (Rolling100: {roll_reward:.2f}) | "
                  f"Secrecy Rate: {avg_row[2]/1e6:.6f} M-suts/s (Rolling100: {roll_secrecy/1e6:.6f} M-suts/s) "
                  f"({avg_row[2]:.2f} suts/s (Rolling100: {roll_secrecy:.2f} suts/s))")
            
    save_history_to_csv(history, "matd3pg_history.csv")
    plot_individual_convergence(history, "MATD3PG")
    return history

# (E) MAPPO (Proposed Multi-Agent PPO)
def train_mappo(env, num_episodes=150, steps_per_episode=50, episodes_per_update=8,
                 entropy_coef=0.01, max_grad_norm=0.5, actor_lr=2.5e-4, critic_lr=2.5e-4):
    """
    MAPPO training loop.

    Stability note: earlier revisions updated the policy from a single
    episode's 50-step on-policy rollout, with no gradient clipping and no
    entropy bonus. That produced very high-variance policy updates (visible
    as large oscillations in the rolling-average utility, unlike the
    off-policy, replay-buffer-based SASAC/MATD3PG baselines which update
    every step from a large, diverse buffer). This version accumulates
    `episodes_per_update` episodes into a larger on-policy rollout before
    each PPO update (reduces gradient variance), clips gradients to
    `max_grad_norm` (bounds the size of any single update), and adds a small
    entropy bonus (keeps exploration from collapsing prematurely) --
    standard PPO stabilization practice for small on-policy batch sizes.
    """
    print("\n--- Training MAPPO (Proposed Algorithm) ---")
    env.reset_objective_tracking()
    history = []

    agent_dims = [3, 3, 3]

    # Models
    # 3 Gaussian Actor Networks
    actors = [GaussianActor(env.state_dim, dim) for dim in agent_dims]
    # Centralized Critic Network (evaluates state value)
    critic = ValueNetwork(env.state_dim)

    actor_opts = [optim.Adam(actor.parameters(), lr=actor_lr) for actor in actors]
    critic_opt = optim.Adam(critic.parameters(), lr=critic_lr)

    # MAPPO Hyperparameters
    gamma = 0.95
    gae_lambda = 0.95
    ppo_epochs = 5
    batch_size = 64
    clip_epsilon = 0.2

    # Rollout buffer, accumulated across `episodes_per_update` episodes
    buf_states, buf_actions, buf_rewards, buf_log_probs, buf_values = [], [], [], [], []
    buf_episode_bounds = []  # (start_idx, end_idx, bootstrap_value) per episode, for per-episode GAE

    for ep in range(num_episodes):
        # Collect trajectory for 1 episode
        ep_states, ep_actions, ep_rewards, ep_log_probs, ep_values = [], [], [], [], []

        state = env.reset()
        ep_reward = 0.0
        metrics = {"secrecy_total": [], "secrecy_target": [], "secrecy_user": [],
                   "pd_target": [], "pd_eaves": [], "crb_target": [],
                   "p_bs_tx": [], "p_jam_tx": [], "noma_a_far": [], "joint_satisfied": []}

        for step in range(steps_per_episode):
            state_t = torch.FloatTensor(state).unsqueeze(0)

            # Centralized value prediction
            with torch.no_grad():
                val = critic(state_t).item()

            # Sample actions and log_probs for each agent
            act_list = []
            log_prob_list = []

            for i, actor in enumerate(actors):
                with torch.no_grad():
                    act_indiv_t, log_prob_indiv_t = actor.sample(state_t, reparameterize=False)
                    act_list.append(act_indiv_t.numpy()[0])
                    log_prob_list.append(log_prob_indiv_t.item())

            joint_action = np.concatenate(act_list)
            next_state, reward, _, _, info = env.step(joint_action)

            ep_states.append(state)
            ep_actions.append(joint_action)
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob_list)  # individual log probabilities
            ep_values.append(val)

            state = next_state
            ep_reward += reward
            for k in metrics:
                metrics[k].append(info[k])

        # Bootstrap value for GAE at the end of this episode's trajectory
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            last_value = critic(state_t).item()

        buf_episode_bounds.append((len(buf_states), len(buf_states) + steps_per_episode, last_value))
        buf_states.extend(ep_states)
        buf_actions.extend(ep_actions)
        buf_rewards.extend(ep_rewards)
        buf_log_probs.extend(ep_log_probs)
        buf_values.extend(ep_values)

        avg_row = [
            ep + 1, ep_reward,
            np.mean(metrics["secrecy_total"]), np.mean(metrics["secrecy_target"]), np.mean(metrics["secrecy_user"]),
            np.mean(metrics["pd_target"]), np.mean(metrics["pd_eaves"]), np.mean(metrics["crb_target"]),
            np.mean(metrics["p_bs_tx"]), np.mean(metrics["p_jam_tx"]), np.mean(metrics["noma_a_far"]),
            np.mean(metrics["joint_satisfied"])
        ]
        history.append(avg_row)

        # Only run a PPO update once every `episodes_per_update` episodes
        # (or on the final episode), using the accumulated rollout buffer.
        is_update_step = ((ep + 1) % episodes_per_update == 0) or (ep + 1 == num_episodes)
        if is_update_step and buf_states:
            # Per-episode GAE (each episode's advantages/returns computed over
            # its own trajectory, then concatenated into one training batch)
            returns, advantages = [], []
            for start_idx, end_idx, bootstrap_value in buf_episode_bounds:
                seg_rewards = buf_rewards[start_idx:end_idx]
                seg_values = buf_values[start_idx:end_idx] + [bootstrap_value]
                gae = 0.0
                seg_returns, seg_advantages = [], []
                for t in reversed(range(len(seg_rewards))):
                    delta = seg_rewards[t] + gamma * seg_values[t + 1] - seg_values[t]
                    gae = delta + gamma * gae_lambda * gae
                    seg_advantages.insert(0, gae)
                    seg_returns.insert(0, gae + seg_values[t])
                returns.extend(seg_returns)
                advantages.extend(seg_advantages)

            # Convert list to tensors
            states_t = torch.FloatTensor(np.array(buf_states))
            actions_t = torch.FloatTensor(np.array(buf_actions))
            old_log_probs_t = torch.FloatTensor(np.array(buf_log_probs))
            returns_t = torch.FloatTensor(np.array(returns)).unsqueeze(-1)
            advantages_t = torch.FloatTensor(np.array(advantages)).unsqueeze(-1)

            # Normalize advantages (now computed over a much larger, more
            # representative batch than a single 50-step episode)
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # PPO Update epochs
            dataset_size = len(buf_states)
            for epoch in range(ppo_epochs):
                indices = np.arange(dataset_size)
                np.random.shuffle(indices)

                for start in range(0, dataset_size, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    if len(batch_idx) == 0:
                        continue

                    states_b = states_t[batch_idx]
                    actions_b = actions_t[batch_idx]
                    old_log_probs_b = old_log_probs_t[batch_idx]
                    returns_b = returns_t[batch_idx]
                    advantages_b = advantages_t[batch_idx]

                    # Centralized Critic Update
                    values_pred = critic(states_b)
                    critic_loss = nn.MSELoss()(values_pred, returns_b)

                    critic_opt.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                    critic_opt.step()

                    # Multi-Agent Policy updates
                    for i in range(len(actors)):
                        # Compute log probabilities of actions_b for agent i
                        start_idx = sum(agent_dims[:i])
                        end_idx = start_idx + agent_dims[i]
                        act_indiv_b = actions_b[:, start_idx:end_idx]

                        mean, log_std = actors[i](states_b)
                        std = log_std.exp()
                        normal = torch.distributions.Normal(mean, std)

                        # Target policy actions are tanh-scaled, mapping back to compute log prob
                        # Since act_indiv_b is tanh(x), we compute arctanh(act_indiv_b)
                        # For stability: arctanh(y) = 0.5 * log((1+y)/(1-y))
                        y_clamped = torch.clamp(act_indiv_b, -0.999, 0.999)
                        x_t_b = 0.5 * torch.log((1.0 + y_clamped) / (1.0 - y_clamped))

                        log_prob_indiv_b = normal.log_prob(x_t_b) - torch.log(1.0 - act_indiv_b.pow(2) + 1e-6)
                        log_prob_indiv_b = log_prob_indiv_b.sum(dim=-1, keepdim=True)
                        entropy_b = normal.entropy().sum(dim=-1, keepdim=True)

                        # We compute ratio of joint policies
                        # For simplicity, we approximate agent i's ratio
                        ratio = torch.exp(log_prob_indiv_b - old_log_probs_b[:, i:i + 1])
                        surr1 = ratio * advantages_b
                        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_b
                        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy_b.mean()

                        actor_opts[i].zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actors[i].parameters(), max_grad_norm)
                        actor_opts[i].step()

            # Clear the rollout buffer after the update (on-policy)
            buf_states, buf_actions, buf_rewards, buf_log_probs, buf_values = [], [], [], [], []
            buf_episode_bounds = []

        if (ep + 1) % 20 == 0:
            roll_reward = np.mean([row[1] for row in history[-100:]])
            roll_secrecy = np.mean([row[2] for row in history[-100:]])
            print(f"Episode {ep+1}/{num_episodes} | Reward: {ep_reward:.2f} (Rolling100: {roll_reward:.2f}) | "
                  f"Secrecy Rate: {avg_row[2]/1e6:.6f} M-suts/s (Rolling100: {roll_secrecy/1e6:.6f} M-suts/s) "
                  f"({avg_row[2]:.2f} suts/s (Rolling100: {roll_secrecy:.2f} suts/s))")

    save_history_to_csv(history, "mappo_history.csv")
    plot_individual_convergence(history, "MAPPO")
    return history

# --- 5. Generate Combined Comparison Plots ---
def generate_comparison_plots(mappo_hist, matd3pg_hist, sac_hist, td3pg_hist, rw_hist):
    print("\n--- Generating Combined Comparison Plots ---")
    
    episodes = [r[0] for r in mappo_hist]
    
    # Helper for rolling average calculation
    def rolling_average(data, window=100):
        res = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            res.append(np.mean(data[start:i+1]))
        return res
        
    hists = {
        'MAPPO (Proposed)': ('purple', mappo_hist),
        'MATD3PG': ('blue', matd3pg_hist),
        'SASAC': ('green', sac_hist),
        'SATD3PG': ('orange', td3pg_hist),
        'Random Walk': ('grey', rw_hist)
    }
    
    # Compute a global secrecy normalization reference across all algorithms to ensure fair comparison
    all_secrecy_vals = []
    for name, (color, hist) in hists.items():
        all_secrecy_vals.extend([r[2] for r in hist])
    global_r_ref = robust_max_reference(all_secrecy_vals)
    
    # 1. ASSR comparison plot
    plt.figure(figsize=(10, 6))
    for name, (color, hist) in hists.items():
        secrecy_vals = np.asarray([r[2] for r in hist], dtype=float)
        assr_vals = np.clip(secrecy_vals / global_r_ref, 0.0, 1.0)
        roll_vals = rolling_average(assr_vals, 100)
        linestyle = '--' if name == 'Random Walk' else '-'
        plt.plot(episodes, assr_vals, color=color, alpha=0.15, linestyle=linestyle)
        plt.plot(episodes, roll_vals, label=name, color=color, linewidth=2, linestyle=linestyle)
        
    plt.title('ASSR Convergence Comparison (Rolling 100)')
    plt.xlabel('Episode')
    plt.ylabel('ASSR')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    
    comparison_assr_path = os.path.join(OUTPUT_DIR, "assr_comparison.png")
    plt.savefig(comparison_assr_path, dpi=150)
    plt.close()
    
    # 2. Combined Pd comparison plot
    plt.figure(figsize=(10, 6))
    for name, (color, hist) in hists.items():
        pd_target_vals = np.asarray([r[5] for r in hist], dtype=float)
        pd_eaves_vals = np.asarray([r[6] for r in hist], dtype=float)
        pd_combined_vals = LAMBDA1_PD_SENSING * pd_eaves_vals + LAMBDA2_PD_SENSING * pd_target_vals
        roll_vals = rolling_average(pd_combined_vals, 100)
        linestyle = '--' if name == 'Random Walk' else '-'
        plt.plot(episodes, pd_combined_vals, color=color, alpha=0.15, linestyle=linestyle)
        plt.plot(episodes, roll_vals, label=name, color=color, linewidth=2, linestyle=linestyle)
        
    plt.title('Pd Convergence Comparison (Rolling 100)')
    plt.xlabel('Episode')
    plt.ylabel('Pd')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    
    comparison_pd_path = os.path.join(OUTPUT_DIR, "pd_comparison.png")
    plt.savefig(comparison_pd_path, dpi=150)
    plt.close()
    
    # 3. Combined utility comparison plot
    plt.figure(figsize=(10, 6))
    for name, (color, hist) in hists.items():
        secrecy_vals = np.asarray([r[2] for r in hist], dtype=float)
        pd_target_vals = np.asarray([r[5] for r in hist], dtype=float)
        pd_eaves_vals = np.asarray([r[6] for r in hist], dtype=float)
        crb_target_vals = np.asarray([r[7] for r in hist], dtype=float)
        
        assr_vals = np.clip(secrecy_vals / global_r_ref, 0.0, 1.0)
        pd_combined_vals = LAMBDA1_PD_SENSING * pd_eaves_vals + LAMBDA2_PD_SENSING * pd_target_vals
        crb_feasible_vals = (crb_target_vals <= CRB_THRESHOLD).astype(float)
        
        utility_vals = (LAMBDA1_ASSR * assr_vals + LAMBDA2_PD * pd_combined_vals) * crb_feasible_vals
        roll_vals = rolling_average(utility_vals, 100)
        
        linestyle = '--' if name == 'Random Walk' else '-'
        plt.plot(episodes, utility_vals, color=color, alpha=0.15, linestyle=linestyle)
        plt.plot(episodes, roll_vals, label=name, color=color, linewidth=2, linestyle=linestyle)
        
    plt.title('Combined Utility Convergence Comparison (Rolling 100)')
    plt.xlabel('Episode')
    plt.ylabel(r'Utility (($0.5$ ASSR $+\ 0.5$ Pd) $\times$ CRB-feasible)')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    
    comparison_utility_path = os.path.join(OUTPUT_DIR, "convergence_comparison.png")
    plt.savefig(comparison_utility_path, dpi=150)
    plt.close()
    
    print("Comparison plots saved to:")
    print(f"  - {comparison_assr_path}")
    print(f"  - {comparison_pd_path}")
    print(f"  - {comparison_utility_path}")

# --- 6. Main Runner Function ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RL Convergence Study on Week 9 System Model")
    parser.add_argument("--episodes", type=int, default=150, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps per episode")
    parser.add_argument("--lambda1", type=float, default=0.5, help="Weight for eavesdropper detection probability in Pd")
    parser.add_argument("--lambda2", type=float, default=0.5, help="Weight for target detection probability in Pd")
    args = parser.parse_args()

    global LAMBDA1_PD_SENSING, LAMBDA2_PD_SENSING
    LAMBDA1_PD_SENSING = args.lambda1
    LAMBDA2_PD_SENSING = args.lambda2

    set_seed(42)
    env = Week9ISACEnv(seed=42)
    
    episodes = args.episodes
    steps = args.steps
    
    print(f"===========================================================")
    print(f"Starting Convergence Study on Week 9 System Model")
    print(f"Episodes: {episodes} | Steps per Episode: {steps}")
    print(f"===========================================================")
    
    # Run Random Walk
    rw_hist = run_random_walk(env, num_episodes=episodes, steps_per_episode=steps)
    
    # Train SATD3PG
    td3_hist = train_satd3pg(env, num_episodes=episodes, steps_per_episode=steps)
    
    # Train SASAC
    sac_hist = train_sasac(env, num_episodes=episodes, steps_per_episode=steps)
    
    # Train MATD3PG
    matd3pg_hist = train_matd3pg(env, num_episodes=episodes, steps_per_episode=steps)
    
    # Train MAPPO
    mappo_hist = train_mappo(env, num_episodes=episodes, steps_per_episode=steps)
    
    # Generate Comparisons
    generate_comparison_plots(mappo_hist, matd3pg_hist, sac_hist, td3_hist, rw_hist)
    
    print("\n===========================================================")
    print("Convergence Study Completed Successfully!")
    print(f"Outputs written to: {os.path.abspath(OUTPUT_DIR)}")
    print("===========================================================")

if __name__ == "__main__":
    main()

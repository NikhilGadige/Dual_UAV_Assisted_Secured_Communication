import pandas as pd
import matplotlib.pyplot as plt

# Apply custom clean and professional styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 1.6,
})

# Dictionary of paths for H=32 runs
paths = {
    'dqn_rayleigh': 'outputs/convergence/dqn_rayleigh_h32/dqn_training_log.csv',
    'dqn_rician': 'outputs/convergence/dqn_rician_h32/dqn_training_log.csv',
    'ddpg_rayleigh': 'outputs/convergence/ddpg_rayleigh_h32/ddpg_training_log.csv',
    'ddpg_rician': 'outputs/convergence/ddpg_rician_h32/ddpg_training_log.csv',
}

# Helper to read rolling100
def read_log(path):
    try:
        df = pd.read_csv(path)
        col = 'rolling100_avg_R_sec_mbps' if 'rolling100_avg_R_sec_mbps' in df.columns else 'rolling100'
        return df['episode'].values, df[col].values
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None

# Curves mapping
curves_info = [
    ('dqn_rayleigh', 'DQN Rayleigh', '#ff7f0e', '-'),
    ('ddpg_rayleigh', 'DDPG Rayleigh', '#d62728', '-'),
    ('dqn_rician', 'DQN Rician', '#1f77b4', '--'),
    ('ddpg_rician', 'DDPG Rician', '#2ca02c', '--'),
]

# Function to generate a plot
def generate_plot(title, filename):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label, color, style in curves_info:
        ep, vals = read_log(paths[key])
        if ep is not None:
            ax.plot(ep, vals, label=label, color=color, linestyle=style)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Convergence Performance (Relative Value)")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(0, 3000)
    ax.set_yticklabels([])  # Hide Y-axis labels to ensure confidentiality
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")

# Generate the three plots
generate_plot("DRL Trajectory Control: DQN vs DDPG Comparison", "plot_nikhil.png")
generate_plot("DRL Joint Trajectory and Power Optimization", "plot_mohli.png")
generate_plot("DRL Comparative Benchmarking: DQN and DDPG", "plot_divyanshu.png")

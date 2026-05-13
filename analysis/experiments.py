import csv
from pathlib import Path

from analysis.baselines import (
    distance_greedy_policy,
    evaluate_policy,
    print_summary,
    random_policy,
)


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def run_baseline_experiments(
    episodes_per_seed: int = 20,
    seeds: list[int] | None = None,
    output_dir: str = "outputs",
) -> dict:
    if seeds is None:
        seeds = [7, 21, 42, 84, 168]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policies = [
        ("Random Walk", random_policy),
        ("Distance-Greedy", distance_greedy_policy),
    ]

    summary_rows: list[dict] = []
    episode_rows: list[dict] = []

    for policy_name, policy_fn in policies:
        for seed in seeds:
            summary = evaluate_policy(
                policy_name,
                policy_fn,
                episodes=episodes_per_seed,
                seed=seed,
                return_episode_metrics=True,
            )

            summary_rows.append(
                {
                    "policy": summary["policy"],
                    "seed": seed,
                    "episodes": summary["episodes"],
                    "mean_episode_reward_bps_step": summary["mean_episode_reward_bps_step"],
                    "mean_episode_secrecy_throughput_bps_step": summary[
                        "mean_episode_secrecy_throughput_bps_step"
                    ],
                    "mean_episode_secrecy_mbits": summary["mean_episode_secrecy_mbits"],
                    "mean_avg_R_legit_mbps": summary["mean_avg_R_legit_mbps"],
                    "mean_avg_R_eve_mbps": summary["mean_avg_R_eve_mbps"],
                    "mean_avg_R_sec_mbps": summary["mean_avg_R_sec_mbps"],
                }
            )

            for i, m in enumerate(summary["episode_metrics"], start=1):
                episode_rows.append(
                    {
                        "policy": policy_name,
                        "seed": seed,
                        "episode_idx": i,
                        "steps": m["steps"],
                        "episode_secrecy_mbits": m["episode_secrecy_mbits"],
                        "avg_R_legit_mbps": m["avg_R_legit_mbps"],
                        "avg_R_eve_mbps": m["avg_R_eve_mbps"],
                        "avg_R_sec_mbps": m["avg_R_sec_mbps"],
                    }
                )

    summary_csv = out_dir / "baseline_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    episodes_csv = out_dir / "baseline_episodes.csv"
    with episodes_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)

    aggregate = _aggregate_across_seeds(summary_rows)
    _print_aggregate(aggregate, episodes_per_seed, seeds)
    _plot_results(out_dir, summary_rows, episode_rows)

    return {
        "summary_csv": str(summary_csv.resolve()),
        "episodes_csv": str(episodes_csv.resolve()),
        "aggregate": aggregate,
    }


def _aggregate_across_seeds(summary_rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in summary_rows:
        grouped.setdefault(row["policy"], []).append(row)

    aggregate: dict[str, dict] = {}
    for policy, rows in grouped.items():
        n = len(rows)
        aggregate[policy] = {
            "mean_avg_R_sec_mbps": sum(r["mean_avg_R_sec_mbps"] for r in rows) / n,
            "mean_avg_R_legit_mbps": sum(r["mean_avg_R_legit_mbps"] for r in rows) / n,
            "mean_avg_R_eve_mbps": sum(r["mean_avg_R_eve_mbps"] for r in rows) / n,
            "mean_episode_secrecy_mbits": sum(r["mean_episode_secrecy_mbits"] for r in rows) / n,
        }
    return aggregate


def _print_aggregate(aggregate: dict, episodes: int, seeds: list[int]) -> None:
    print(f"\nBaseline experiment complete | episodes/seed={episodes} | seeds={seeds}")
    for policy, metrics in aggregate.items():
        print(f"\nPolicy: {policy}")
        print(f"  Mean avg R_sec     : {metrics['mean_avg_R_sec_mbps']:.4f} Mbps")
        print(f"  Mean avg R_legit   : {metrics['mean_avg_R_legit_mbps']:.4f} Mbps")
        print(f"  Mean avg R_eve     : {metrics['mean_avg_R_eve_mbps']:.4f} Mbps")
        print(f"  Mean secrecy/ep    : {metrics['mean_episode_secrecy_mbits']:.4f} Mbits")


def _plot_results(out_dir: Path, summary_rows: list[dict], episode_rows: list[dict]) -> None:
    plt = _safe_import_matplotlib()
    if plt is None:
        print("\nmatplotlib not installed, skipping plots.")
        return

    policies = sorted({r["policy"] for r in summary_rows})
    mean_r_sec = []
    for p in policies:
        vals = [r["mean_avg_R_sec_mbps"] for r in summary_rows if r["policy"] == p]
        mean_r_sec.append(sum(vals) / len(vals))

    fig1 = plt.figure(figsize=(7, 4))
    ax1 = fig1.add_subplot(111)
    ax1.bar(policies, mean_r_sec)
    ax1.set_ylabel("Avg Secrecy Rate (Mbps)")
    ax1.set_title("Baseline Comparison Across Seeds")
    ax1.grid(axis="y", alpha=0.25)
    fig1.tight_layout()
    fig1.savefig(out_dir / "baseline_bar_avg_rsec_mbps.png", dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(7, 4))
    ax2 = fig2.add_subplot(111)
    data = []
    for p in policies:
        vals = [r["episode_secrecy_mbits"] for r in episode_rows if r["policy"] == p]
        data.append(vals)
    ax2.boxplot(data, tick_labels=policies)
    ax2.set_ylabel("Episode Secrecy Payload (Mbits)")
    ax2.set_title("Episode-Level Distribution")
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(out_dir / "baseline_box_episode_secrecy_mbits.png", dpi=150)
    plt.close(fig2)

    print("\nSaved plots:")
    print(f"  - {(out_dir / 'baseline_bar_avg_rsec_mbps.png').resolve()}")
    print(f"  - {(out_dir / 'baseline_box_episode_secrecy_mbits.png').resolve()}")


if __name__ == "__main__":
    results = run_baseline_experiments(episodes_per_seed=20)

    print("\nSaved CSV files:")
    print(f"  - {results['summary_csv']}")
    print(f"  - {results['episodes_csv']}")

    print("\nSingle-seed snapshot (seed=42):")
    print_summary(
        evaluate_policy(
            "Random Walk", random_policy, episodes=20, seed=42, return_episode_metrics=False
        )
    )
    print_summary(
        evaluate_policy(
            "Distance-Greedy",
            distance_greedy_policy,
            episodes=20,
            seed=42,
            return_episode_metrics=False,
        )
    )

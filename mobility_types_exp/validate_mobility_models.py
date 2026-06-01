import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mobility_types_exp.run_mobility_experiments import run_experiment
from mobility_types_exp.configs import MOBILITY_MODELS, ALGORITHMS, CHANNELS, OUTPUT_ROOT

SUMMARY_DIR = Path(OUTPUT_ROOT)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def run_validation(episodes=100, seed=42, device="cpu"):
    results = []

    print(f"\n{'='*70}")
    print(f"  MOBILITY MODEL VALIDATION - {len(MOBILITY_MODELS)} models x {len(ALGORITHMS)} algos x {len(CHANNELS)} channels")
    print(f"  Total experiments: {len(MOBILITY_MODELS) * len(ALGORITHMS) * len(CHANNELS)}")
    print(f"  Episodes per experiment: {episodes}")
    print(f"{'='*70}\n")

    total = len(MOBILITY_MODELS) * len(ALGORITHMS) * len(CHANNELS)
    completed = 0

    for mobility_name in MOBILITY_MODELS:
        for algo in ALGORITHMS:
            for channel in CHANNELS:
                completed += 1
                print(f"\n[{completed}/{total}] Starting...")

                t0 = time.time()
                result = run_experiment(
                    mobility_name=mobility_name,
                    algorithm=algo,
                    channel=channel,
                    episodes=episodes,
                    seed=seed,
                    device=device,
                )
                elapsed = time.time() - t0
                result["elapsed_sec"] = round(elapsed, 1)
                results.append(result)

                print(f"  OK [{completed}/{total}] {mobility_name}/{algo}/{channel} - "
                      f"{result['Final_Rolling100_Secrecy']} Mbps - {elapsed:.1f}s")

    return results


def write_summary_csv(results, output_path):
    fieldnames = [
        "Mobility_Model", "Algorithm", "Channel",
        "Final_Rolling100_Secrecy", "Best_Secrecy",
        "Average_Reward", "Convergence_Episode",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "Mobility_Model": r["mobility_model"],
                "Algorithm": r["algorithm"],
                "Channel": r["channel"],
                "Final_Rolling100_Secrecy": r["Final_Rolling100_Secrecy"],
                "Best_Secrecy": r["Best_Secrecy"],
                "Average_Reward": r["Average_Reward"],
                "Convergence_Episode": r["Convergence_Episode"],
            })
    print(f"\nSummary CSV written to: {output_path}")


def generate_analysis(results, output_path):
    lines = []
    lines.append("# Mobility Model Analysis Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Models: {len(MOBILITY_MODELS)} | Algorithms: {len(ALGORITHMS)} | Channels: {len(CHANNELS)}\n")

    # -- Aggregate by mobility model --
    from collections import defaultdict
    model_rolls = defaultdict(list)
    for r in results:
        model_rolls[r["mobility_model"]].append(r["Final_Rolling100_Secrecy"])

    avg_by_model = {m: sum(vals) / len(vals) for m, vals in model_rolls.items()}
    ranked = sorted(avg_by_model, key=avg_by_model.get, reverse=True)

    lines.append("## 1. Ranking of Mobility Models (by avg Final_Rolling100_Secrecy)\n")
    lines.append("| Rank | Mobility Model | Avg Final Rolling100 Secrecy (Mbps) |")
    lines.append("|------|---------------|-------------------------------------|")
    for i, m in enumerate(ranked, 1):
        lines.append(f"| {i} | {m} | {avg_by_model[m]:.4f} |")

    lines.append("")
    hardest_model = ranked[-1]
    best_model = ranked[0]
    lines.append(f"## 2. Hardest Mobility Model")
    lines.append(f"**{hardest_model}** - lowest average secrecy ({avg_by_model[hardest_model]:.4f} Mbps).\n")

    lines.append(f"## 3. Highest Secrecy Mobility Model")
    lines.append(f"**{best_model}** - highest average secrecy ({avg_by_model[best_model]:.4f} Mbps).\n")

    # -- Convergence by model --
    model_conv = defaultdict(list)
    for r in results:
        model_conv[r["mobility_model"]].append(r["Convergence_Episode"])
    avg_conv = {m: sum(v) / len(v) for m, v in model_conv.items()}
    fastest_conv = sorted(avg_conv, key=avg_conv.get)[0]

    lines.append(f"## 4. Fastest Converging Mobility Model")
    lines.append(f"**{fastest_conv}** - average convergence at episode {avg_conv[fastest_conv]:.0f}.\n")
    lines.append("| Mobility Model | Avg Convergence Episode |")
    lines.append("|---------------|------------------------|")
    for m in sorted(avg_conv, key=avg_conv.get):
        lines.append(f"| {m} | {avg_conv[m]:.0f} |")

    # -- Per-algorithm observations --
    lines.append("\n## 5. Per-Algorithm Observations\n")
    for algo in ALGORITHMS:
        algo_results = [r for r in results if r["algorithm"] == algo]
        if not algo_results:
            continue
        algo_rolls = [r["Final_Rolling100_Secrecy"] for r in algo_results]
        algo_models = defaultdict(list)
        for r in algo_results:
            algo_models[r["mobility_model"]].append(r["Final_Rolling100_Secrecy"])
        algo_avg = {m: sum(v) / len(v) for m, v in algo_models.items()}
        best_for_algo = sorted(algo_avg, key=algo_avg.get, reverse=True)[0]

        lines.append(f"### {algo.upper()}")
        lines.append(f"- Avg Final Rolling100: {sum(algo_rolls) / len(algo_rolls):.4f} Mbps (over {len(algo_results)} runs)")
        lines.append(f"- Best mobility: **{best_for_algo}** ({algo_avg[best_for_algo]:.4f} Mbps)")
        lines.append(f"- Worst mobility: **{sorted(algo_avg, key=algo_avg.get)[0]}** ({algo_avg[sorted(algo_avg, key=algo_avg.get)[0]]:.4f} Mbps)")
        lines.append("")

    # -- Channel effect --
    lines.append("## 6. Channel Model Effect\n")
    for channel in CHANNELS:
        ch_results = [r for r in results if r["channel"] == channel]
        ch_rolls = [r["Final_Rolling100_Secrecy"] for r in ch_results]
        lines.append(f"- **{channel.title()}**: {sum(ch_rolls) / len(ch_rolls):.4f} Mbps avg over {len(ch_results)} runs")

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Analysis report written to: {output_path}")
    return content


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate mobility models with short runs")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    t_start = time.time()

    results = run_validation(
        episodes=args.episodes,
        seed=args.seed,
        device=args.device,
    )

    summary_csv = str(SUMMARY_DIR / "mobility_summary.csv")
    write_summary_csv(results, summary_csv)

    analysis_md = str(SUMMARY_DIR / "mobility_analysis.md")
    generate_analysis(results, analysis_md)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  VALIDATION COMPLETE")
    print(f"  Total experiments: {len(results)}")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  Summary:  {summary_csv}")
    print(f"  Analysis: {analysis_md}")
    print(f"{'='*70}")

    print("\n--- mobility_summary.csv Preview ---")
    with open(summary_csv, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 12:
                print("  ...")
                break
            print(f"  {line.strip()}")

    return results


if __name__ == "__main__":
    main()

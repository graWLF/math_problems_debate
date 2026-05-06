"""
LogiQA deney sonuçlarını görselleştirir.

Usage:
    uv run python experiments/logiqa_analysis.py [--results RESULTS_DIR]

RESULTS_DIR verilmezse experiments/results/ altındaki en son logiqa_groq_* klasörünü kullanır.
Grafikler experiments/analysis/logiqa/ altına kaydedilir.
"""
import json
import argparse
import os
import os.path as osp
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_BASE = osp.join(osp.dirname(__file__), "results")
OUTPUT_DIR = osp.join(osp.dirname(__file__), "analysis", "logiqa")

PROTOCOL_LABELS = {
    "Blind":                "Blind\n(baseline)",
    "Debate_t0_n2":         "Debate\nsequential n=2",
    "Debate_t1_n2":         "Debate\nsimultaneous n=2",
    "Debate_t0_n4":         "Debate\nsequential n=4",
    "Debate_t1_n4":         "Debate\nsimultaneous n=4",
    "Consultancy_t0_n2":    "Consultancy\nclient-first n=2",
    "Consultancy_t1_n2":    "Consultancy\nconsultant-first n=2",
    "Consultancy_t0_n4":    "Consultancy\nclient-first n=4",
    "Consultancy_t1_n4":    "Consultancy\nconsultant-first n=4",
}

COLORS = {
    "Blind":       "#7f7f7f",
    "Debate":      "#1f77b4",
    "Consultancy": "#2ca02c",
}


def get_color(protocol_key: str) -> str:
    if "Debate" in protocol_key:
        return COLORS["Debate"]
    if "Consultancy" in protocol_key:
        return COLORS["Consultancy"]
    return COLORS["Blind"]


def find_latest_results() -> str:
    dirs = sorted(glob.glob(osp.join(RESULTS_BASE, "logiqa_groq_*")))
    if not dirs:
        raise FileNotFoundError("No logiqa_groq_* results found. Run init_exp.py first.")
    return dirs[-1]


def load_stats(results_dir: str) -> list[dict]:
    path = osp.join(results_dir, "all_stats.json")
    if not osp.exists(path):
        # Fallback: collect individual stats.json files
        entries = []
        for stats_file in sorted(glob.glob(osp.join(results_dir, "*", "*", "stats.json"))):
            parts = stats_file.split(os.sep)
            protocol_dir = parts[-3]
            with open(stats_file) as f:
                stats = json.load(f)
            entries.append({"protocol_dir": protocol_dir, "stats": stats})
        return entries

    with open(path) as f:
        data = json.load(f)

    entries = []
    for item in data:
        cfg = item["config"]
        protocol = cfg["protocol"]
        init_kw = cfg.get("init_kwargs", {})
        simultaneous = init_kw.get("simultaneous", None)
        num_turns = init_kw.get("num_turns", None)
        consultant_first = init_kw.get("consultant_goes_first", None)

        if simultaneous is None and consultant_first is None:
            key = protocol
        elif consultant_first is not None:
            t = 1 if consultant_first else 0
            key = f"{protocol}_t{t}_n{num_turns}"
        else:
            t = 1 if simultaneous else 0
            key = f"{protocol}_t{t}_n{num_turns}"

        entries.append({"protocol_dir": key, "stats": item["stats"]})
    return entries


def plot_judge_accuracy(entries, output_dir):
    labels, means, stds, colors = [], [], [], []
    for e in entries:
        key = e["protocol_dir"]
        s = e["stats"]
        labels.append(PROTOCOL_LABELS.get(key, key))
        means.append(s["jse_b0_mean"]["accuracy"])
        stds.append(s["jse_b0_std"]["accuracy"])
        colors.append(get_color(key))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random baseline (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Judge Accuracy", fontsize=11)
    ax.set_title("Judge Accuracy per Protocol\n(LogiQA — Llama 70B debater vs 8B judge)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    legend_patches = [
        mpatches.Patch(color=COLORS["Blind"], label="Blind"),
        mpatches.Patch(color=COLORS["Debate"], label="Debate"),
        mpatches.Patch(color=COLORS["Consultancy"], label="Consultancy"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color="red", linestyle="--", label="Random (50%)")
    ], fontsize=9)

    plt.tight_layout()
    path = osp.join(output_dir, "judge_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_asd(entries, output_dir):
    labels, means, stds, colors = [], [], [], []
    for e in entries:
        key = e["protocol_dir"]
        s = e["stats"]
        labels.append(PROTOCOL_LABELS.get(key, key))
        means.append(s["asd_mean"]["accuracy"])
        stds.append(s["asd_std"]["accuracy"])
        colors.append(get_color(key))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, means, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Agent Score Difference (ASD)", fontsize=11)
    ax.set_title("ASD per Protocol — Does truth-telling pay off?\n(Positive = arguing for truth is advantageous)", fontsize=12)

    legend_patches = [
        mpatches.Patch(color=COLORS["Blind"], label="Blind"),
        mpatches.Patch(color=COLORS["Debate"], label="Debate"),
        mpatches.Patch(color=COLORS["Consultancy"], label="Consultancy"),
    ]
    ax.legend(handles=legend_patches, fontsize=9)

    plt.tight_layout()
    path = osp.join(output_dir, "asd.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_summary_table(entries, output_dir):
    print("\n── Özet Tablo ──────────────────────────────────────────")
    print(f"{'Protokol':<35} {'Judge Acc':>10} {'ASD acc':>10} {'ASD log':>10}")
    print("─" * 70)
    for e in entries:
        key = e["protocol_dir"]
        s = e["stats"]
        label = PROTOCOL_LABELS.get(key, key).replace("\n", " ")
        print(
            f"{label:<35}"
            f" {s['jse_b0_mean']['accuracy']:>10.3f}"
            f" {s['asd_mean']['accuracy']:>10.3f}"
            f" {s['asd_mean']['log']:>10.3f}"
        )
    print("─" * 70)


def main(results_dir: str | None):
    if results_dir is None:
        results_dir = find_latest_results()
    print(f"Results: {results_dir}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    entries = load_stats(results_dir)

    if not entries:
        print("No stats found.")
        return

    plot_summary_table(entries, OUTPUT_DIR)
    plot_judge_accuracy(entries, OUTPUT_DIR)
    plot_asd(entries, OUTPUT_DIR)
    print(f"\nGrafikler kaydedildi: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None, help="Results directory path")
    args = parser.parse_args()
    main(args.results)

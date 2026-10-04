from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PHASE 8 — RESULT VISUALIZATIONS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

METRICS_JSON = (
    ROOT_DIR
    / "phase8_evaluation"
    / "phase8_metrics.json"
)

PER_TARGET_CSV = (
    ROOT_DIR
    / "phase8_evaluation"
    / "phase8_per_target.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "phase8_evaluation"
)


def load_metrics():
    with open(
        METRICS_JSON,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_per_target():
    data = np.genfromtxt(
        PER_TARGET_CSV,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    return data


def plot_similarity_trajectory(metrics):
    systems = [
        ("baseline", "Baseline"),
        ("original", "Original Learned"),
        ("ranking", "Final Learned"),
    ]

    plt.figure(figsize=(10, 6))

    for key, label in systems:

        trajectory = np.asarray(
            metrics["systems"][key]["mean_trajectory"],
            dtype=np.float64,
        )

        rounds = np.arange(
            1,
            len(trajectory) + 1,
        )

        plt.plot(
            rounds,
            trajectory,
            marker="o",
            markersize=4,
            label=label,
        )

    plt.xlabel("Round")
    plt.ylabel("Mean Face Similarity")

    plt.title(
        "Mean Face Similarity Across 20 Interactive Rounds"
    )

    plt.xticks(
        np.arange(1, 21)
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "phase8_similarity_trajectory.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


def plot_final_r20_comparison(data):
    sessions = data["session_id"]

    baseline = data["final_vs_baseline_r20"]

    # Reconstruct actual R20 values using the saved differences
    # and the values stored in the JSON trajectory.
    #
    # For an exact per-target R20 comparison, use the original
    # evaluation artifact directly in the next analysis stage.
    #
    # Here we visualize the paired Final-vs-Baseline difference.

    final_vs_baseline = np.asarray(
        baseline,
        dtype=np.float64,
    )

    x = np.arange(
        len(sessions)
    )

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        x,
        final_vs_baseline,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xlabel("Evaluation Target")
    plt.ylabel(
        "Final Learned − Baseline R20 Similarity"
    )

    plt.title(
        "Per-Target Final Learned Improvement over Baseline"
    )

    plt.xticks(
        x,
        [
            str(session)
            for session in sessions
        ],
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "phase8_final_vs_baseline_per_target.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


def main():
    print("=" * 60)
    print("PHASE 8 — RESULT VISUALIZATIONS")
    print("=" * 60)

    if not METRICS_JSON.exists():
        raise FileNotFoundError(
            f"Missing metrics file:\n{METRICS_JSON}"
        )

    if not PER_TARGET_CSV.exists():
        raise FileNotFoundError(
            f"Missing per-target file:\n{PER_TARGET_CSV}"
        )

    metrics = load_metrics()
    data = load_per_target()

    plot_similarity_trajectory(
        metrics
    )

    plot_final_r20_comparison(
        data
    )

    print("\nVisualization generation complete.")


if __name__ == "__main__":
    main()
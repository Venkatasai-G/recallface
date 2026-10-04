from pathlib import Path

import numpy as np
import torch
import csv
import json

# ============================================================
# PHASE 8 — EVALUATION & METRICS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    ROOT_DIR
    / "phase8_evaluation"
    / "phase7_baseline_original_ranking_10_targets.pt"
)

METRICS_JSON = (
    ROOT_DIR
    / "phase8_evaluation"
    / "phase8_metrics.json"
)

METRICS_CSV = (
    ROOT_DIR
    / "phase8_evaluation"
    / "phase8_per_target.csv"
)

SYSTEMS = [
    ("baseline", "BASELINE"),
    ("original", "ORIGINAL LEARNED"),
    ("ranking", "FINAL LEARNED"),
]


def load_results():
    """Load the saved Phase 7 three-way evaluation artifact."""
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation artifact not found:\n{EVALUATION_FILE}"
        )

    data = torch.load(
        EVALUATION_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, dict):
        raise TypeError("Evaluation artifact must contain a dictionary.")

    required_keys = {"summary", "results", "evaluation_config"}

    missing = required_keys - set(data.keys())

    if missing:
        raise KeyError(
            f"Evaluation artifact is missing keys: {sorted(missing)}"
        )

    return data


def collect_trajectories(results, system_key):
    """Collect round-by-round similarity and validity trajectories."""

    similarity = []
    validity = []
    session_ids = []

    for session in results:
        session_id = session["session_id"]
        system_result = session[system_key]

        round_similarity = np.asarray(
            system_result["round_best_similarity"],
            dtype=np.float64,
        )

        round_validity = np.asarray(
            system_result["round_validity"],
            dtype=np.float64,
        )

        if len(round_similarity) != 20:
            raise ValueError(
                f"Session {session_id}, {system_key}: "
                f"expected 20 similarity values, "
                f"got {len(round_similarity)}."
            )

        if len(round_validity) != 20:
            raise ValueError(
                f"Session {session_id}, {system_key}: "
                f"expected 20 validity values, "
                f"got {len(round_validity)}."
            )

        similarity.append(round_similarity)
        validity.append(round_validity)
        session_ids.append(session_id)

    return (
        np.asarray(similarity),
        np.asarray(validity),
        session_ids,
    )


def summarize_system(similarity, validity):
    """Calculate Phase 8 summary metrics for one system."""

    r1 = similarity[:, 0]
    r20 = similarity[:, -1]

    improvement = r20 - r1

    mean_trajectory = similarity.mean(axis=0)

    return {
        "num_sessions": int(similarity.shape[0]),
        "num_rounds": int(similarity.shape[1]),

        "mean_r1": float(np.mean(r1)),
        "mean_r20": float(np.mean(r20)),

        "mean_improvement": float(np.mean(improvement)),
        "median_improvement": float(np.median(improvement)),
        "std_improvement": float(np.std(improvement, ddof=1)),
        "min_improvement": float(np.min(improvement)),
        "max_improvement": float(np.max(improvement)),

        "success_rate": float(np.mean(improvement > 0.0)),

        "mean_validity": float(np.mean(validity)),
        "min_validity": float(np.min(validity)),

        "mean_trajectory": mean_trajectory.tolist(),
    }


def print_system_summary(label, metrics):
    """Print a readable Phase 8 summary."""

    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")

    print(f"Sessions              : {metrics['num_sessions']}")
    print(f"Rounds                : {metrics['num_rounds']}")

    print(f"Mean R1 similarity    : {metrics['mean_r1']:.6f}")
    print(f"Mean R20 similarity   : {metrics['mean_r20']:.6f}")

    print(f"Mean improvement      : {metrics['mean_improvement']:+.6f}")
    print(f"Median improvement    : {metrics['median_improvement']:+.6f}")
    print(f"Std improvement       : {metrics['std_improvement']:.6f}")
    print(f"Minimum improvement   : {metrics['min_improvement']:+.6f}")
    print(f"Maximum improvement   : {metrics['max_improvement']:+.6f}")

    print(f"Success rate          : {metrics['success_rate']:.4f}")
    print(f"Mean validity         : {metrics['mean_validity']:.6f}")
    print(f"Minimum validity      : {metrics['min_validity']:.6f}")

    print("\nMean similarity trajectory:")

    for round_idx, value in enumerate(
        metrics["mean_trajectory"],
        start=1,
    ):
        print(
            f"  R{round_idx:02d}: {value:.6f}"
        )


def main():
    print("=" * 60)
    print("PHASE 8 — EVALUATION & METRICS")
    print("=" * 60)

    print(f"\nEvaluation artifact:")
    print(EVALUATION_FILE)

    data = load_results()

    results = data["results"]

    print(f"\nLoaded sessions: {len(results)}")

    all_metrics = {}

    for system_key, label in SYSTEMS:

        similarity, validity, session_ids = collect_trajectories(
            results,
            system_key,
        )

        metrics = summarize_system(
            similarity,
            validity,
        )

        all_metrics[system_key] = metrics

        print_system_summary(
            label,
            metrics,
        )

    # --------------------------------------------------------
    # FINAL MODEL COMPARISONS
    # --------------------------------------------------------

    baseline = all_metrics["baseline"]
    original = all_metrics["original"]
    final = all_metrics["ranking"]

    print(f"\n{'=' * 60}")
    print("FINAL LEARNED COMPARISONS")
    print(f"{'=' * 60}")

    print("\nFINAL LEARNED vs BASELINE")

    print(
        f"Mean R20 difference    : "
        f"{final['mean_r20'] - baseline['mean_r20']:+.6f}"
    )

    print(
        f"Mean improvement diff  : "
        f"{final['mean_improvement'] - baseline['mean_improvement']:+.6f}"
    )

    print("\nFINAL LEARNED vs ORIGINAL LEARNED")

    print(
        f"Mean R20 difference    : "
        f"{final['mean_r20'] - original['mean_r20']:+.6f}"
    )

    print(
        f"Mean improvement diff  : "
        f"{final['mean_improvement'] - original['mean_improvement']:+.6f}"
    )

        # --------------------------------------------------------
    # PER-TARGET ANALYSIS
    # --------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("PER-TARGET ANALYSIS")
    print(f"{'=' * 60}")

    print(
        f"\n{'Session':<15}"
        f"{'Base Δ':>12}"
        f"{'Original Δ':>14}"
        f"{'Final Δ':>12}"
        f"{'Final-Base':>14}"
        f"{'Final-Orig':>14}"
    )

    print("-" * 81)

    baseline_sim, _, _ = collect_trajectories(
        results,
        "baseline",
    )

    original_sim, _, _ = collect_trajectories(
        results,
        "original",
    )

    final_sim, _, _ = collect_trajectories(
        results,
        "ranking",
    )

    final_vs_baseline = []
    final_vs_original = []

    for i, session_id in enumerate(session_ids):

        baseline_delta = (
            baseline_sim[i, -1]
            - baseline_sim[i, 0]
        )

        original_delta = (
            original_sim[i, -1]
            - original_sim[i, 0]
        )

        final_delta = (
            final_sim[i, -1]
            - final_sim[i, 0]
        )

        final_baseline = (
            final_sim[i, -1]
            - baseline_sim[i, -1]
        )

        final_original = (
            final_sim[i, -1]
            - original_sim[i, -1]
        )

        final_vs_baseline.append(final_baseline)
        final_vs_original.append(final_original)

        print(
            f"{str(session_id):<15}"
            f"{baseline_delta:+.6f}"
            f"{original_delta:+.6f}"
            f"{final_delta:+.6f}"
            f"{final_baseline:+.6f}"
            f"{final_original:+.6f}"
        )

    final_vs_baseline = np.asarray(
        final_vs_baseline,
        dtype=np.float64,
    )

    final_vs_original = np.asarray(
        final_vs_original,
        dtype=np.float64,
    )

    print("\nFinal Learned vs Baseline:")
    print(
        f"  Higher R20 : "
        f"{np.sum(final_vs_baseline > 0)}/10"
    )
    print(
        f"  Equal R20  : "
        f"{np.sum(final_vs_baseline == 0)}/10"
    )
    print(
        f"  Lower R20  : "
        f"{np.sum(final_vs_baseline < 0)}/10"
    )

    print("\nFinal Learned vs Original Learned:")
    print(
        f"  Higher R20 : "
        f"{np.sum(final_vs_original > 0)}/10"
    )
    print(
        f"  Equal R20  : "
        f"{np.sum(final_vs_original == 0)}/10"
    )
    print(
        f"  Lower R20  : "
        f"{np.sum(final_vs_original < 0)}/10"
    )

        # --------------------------------------------------------
    # ROUND-BY-ROUND CONVERGENCE ANALYSIS
    # --------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("ROUND-BY-ROUND CONVERGENCE ANALYSIS")
    print(f"{'=' * 60}")

    print(
        f"\n{'System':<22}"
        f"{'Best Round':>12}"
        f"{'Best Mean':>14}"
        f"{'R1':>12}"
        f"{'R5':>12}"
        f"{'R10':>12}"
        f"{'R15':>12}"
        f"{'R20':>12}"
    )

    print("-" * 116)

    convergence_metrics = {}

    for system_key, label in SYSTEMS:

        similarity, _, _ = collect_trajectories(
            results,
            system_key,
        )

        mean_trajectory = similarity.mean(axis=0)

        best_round_index = int(
            np.argmax(mean_trajectory)
        )

        best_round = best_round_index + 1
        best_mean = mean_trajectory[best_round_index]

        convergence_metrics[system_key] = {
            "best_round": best_round,
            "best_mean_similarity": float(best_mean),
            "r1": float(mean_trajectory[0]),
            "r5": float(mean_trajectory[4]),
            "r10": float(mean_trajectory[9]),
            "r15": float(mean_trajectory[14]),
            "r20": float(mean_trajectory[19]),
        }

        print(
            f"{label:<22}"
            f"{best_round:>12}"
            f"{best_mean:>14.6f}"
            f"{mean_trajectory[0]:>12.6f}"
            f"{mean_trajectory[4]:>12.6f}"
            f"{mean_trajectory[9]:>12.6f}"
            f"{mean_trajectory[14]:>12.6f}"
            f"{mean_trajectory[19]:>12.6f}"
        )

    print("\nBest-round analysis:")

    for system_key, label in SYSTEMS:

        metrics = convergence_metrics[system_key]

        gain_to_best = (
            metrics["best_mean_similarity"]
            - metrics["r1"]
        )

        gain_to_r20 = (
            metrics["r20"]
            - metrics["r1"]
        )

        print(f"\n{label}")
        print(
            f"  Best mean similarity : "
            f"{metrics['best_mean_similarity']:.6f}"
        )
        print(
            f"  Best round           : "
            f"R{metrics['best_round']}"
        )
        print(
            f"  R1 → best gain       : "
            f"{gain_to_best:+.6f}"
        )
        print(
            f"  R1 → R20 gain        : "
            f"{gain_to_r20:+.6f}"
        )

    final_metrics = convergence_metrics["ranking"]

    if final_metrics["best_round"] != 20:

        difference = (
            final_metrics["best_mean_similarity"]
            - final_metrics["r20"]
        )

        print(
            "\nFinal Learned observation:"
        )

        print(
            f"  R20 is not the peak mean-similarity round."
        )

        print(
            f"  Peak occurs at R{final_metrics['best_round']} "
            f"with mean similarity "
            f"{final_metrics['best_mean_similarity']:.6f}."
        )

        print(
            f"  Peak-to-R20 difference: "
            f"{difference:+.6f}"
        )

    else:

        print(
            "\nFinal Learned observation:"
        )

        print(
            "  R20 is the peak mean-similarity round."
        )

        # --------------------------------------------------------
    # PAIRED STATISTICAL COMPARISON
    # --------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("PAIRED STATISTICAL COMPARISON")
    print(f"{'=' * 60}")

    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError:
        raise ImportError(
            "SciPy is required for statistical analysis. "
            "Install it with: pip install scipy"
        )

    baseline_r20 = baseline_sim[:, -1]
    original_r20 = original_sim[:, -1]
    final_r20 = final_sim[:, -1]

    comparisons = [
        (
            "FINAL LEARNED vs BASELINE",
            final_r20,
            baseline_r20,
        ),
        (
            "FINAL LEARNED vs ORIGINAL LEARNED",
            final_r20,
            original_r20,
        ),
    ]

    for label, final_values, comparison_values in comparisons:

        differences = final_values - comparison_values

        mean_difference = np.mean(differences)
        median_difference = np.median(differences)
        std_difference = np.std(
            differences,
            ddof=1,
        )

        minimum_difference = np.min(differences)
        maximum_difference = np.max(differences)

        wins = int(np.sum(differences > 0))
        ties = int(np.sum(differences == 0))
        losses = int(np.sum(differences < 0))

        t_stat, t_p = ttest_rel(
            final_values,
            comparison_values,
        )

        try:
            w_stat, w_p = wilcoxon(
                final_values,
                comparison_values,
                alternative="two-sided",
            )
        except ValueError:
            w_stat = np.nan
            w_p = np.nan

        print(f"\n{label}")
        print("-" * 60)

        print(
            f"Mean paired difference   : "
            f"{mean_difference:+.6f}"
        )

        print(
            f"Median paired difference : "
            f"{median_difference:+.6f}"
        )

        print(
            f"Std paired difference    : "
            f"{std_difference:.6f}"
        )

        print(
            f"Minimum difference       : "
            f"{minimum_difference:+.6f}"
        )

        print(
            f"Maximum difference       : "
            f"{maximum_difference:+.6f}"
        )

        print(
            f"Wins / Ties / Losses    : "
            f"{wins} / {ties} / {losses}"
        )

        print(
            f"Paired t-test statistic  : "
            f"{t_stat:.6f}"
        )

        print(
            f"Paired t-test p-value    : "
            f"{t_p:.6f}"
        )

        print(
            f"Wilcoxon statistic       : "
            f"{w_stat:.6f}"
        )

        print(
            f"Wilcoxon p-value         : "
            f"{w_p:.6f}"
        )

        if t_p < 0.05:
            print(
                "Paired t-test result     : "
                "Statistically significant (p < 0.05)"
            )
        else:
            print(
                "Paired t-test result     : "
                "Not statistically significant (p >= 0.05)"
            )

        if not np.isnan(w_p):
            if w_p < 0.05:
                print(
                    "Wilcoxon result          : "
                    "Statistically significant (p < 0.05)"
                )
            else:
                print(
                    "Wilcoxon result          : "
                    "Not statistically significant (p >= 0.05)"
                )

        # --------------------------------------------------------
    # SAVE PHASE 8 METRICS
    # --------------------------------------------------------

    metrics_output = {
        "evaluation_artifact": str(EVALUATION_FILE),
        "num_targets": len(results),
        "num_rounds": 20,

        "systems": all_metrics,

        "convergence": convergence_metrics,

        "paired_comparisons": {
            "final_vs_baseline": {
                "mean_difference": float(
                    np.mean(final_vs_baseline)
                ),
                "median_difference": float(
                    np.median(final_vs_baseline)
                ),
                "std_difference": float(
                    np.std(final_vs_baseline, ddof=1)
                ),
                "wins": int(
                    np.sum(final_vs_baseline > 0)
                ),
                "ties": int(
                    np.sum(final_vs_baseline == 0)
                ),
                "losses": int(
                    np.sum(final_vs_baseline < 0)
                ),
            },
            "final_vs_original": {
                "mean_difference": float(
                    np.mean(final_vs_original)
                ),
                "median_difference": float(
                    np.median(final_vs_original)
                ),
                "std_difference": float(
                    np.std(final_vs_original, ddof=1)
                ),
                "wins": int(
                    np.sum(final_vs_original > 0)
                ),
                "ties": int(
                    np.sum(final_vs_original == 0)
                ),
                "losses": int(
                    np.sum(final_vs_original < 0)
                ),
            },
        },
    }

    with open(
        METRICS_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics_output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # SAVE PER-TARGET CSV
    # --------------------------------------------------------

    with open(
        METRICS_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "session_id",
                "baseline_improvement",
                "original_improvement",
                "final_improvement",
                "final_vs_baseline_r20",
                "final_vs_original_r20",
            ]
        )

        for i, session_id in enumerate(session_ids):

            writer.writerow(
                [
                    session_id,
                    float(
                        baseline_sim[i, -1]
                        - baseline_sim[i, 0]
                    ),
                    float(
                        original_sim[i, -1]
                        - original_sim[i, 0]
                    ),
                    float(
                        final_sim[i, -1]
                        - final_sim[i, 0]
                    ),
                    float(final_vs_baseline[i]),
                    float(final_vs_original[i]),
                ]
            )

    print("\nSaved Phase 8 artifacts:")

    print(
        f"  JSON: {METRICS_JSON}"
    )

    print(
        f"  CSV : {METRICS_CSV}"
    )
    
    print("\nPhase 8 per-target analysis complete.")


if __name__ == "__main__":
    main()
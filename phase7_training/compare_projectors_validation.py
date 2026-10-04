# compare_projectors_validation.py:-


from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# ============================================================
# Project root
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phase4_projector import DirectionProjector
from phase7_training.dataset import NavigationTransitionDataset


# ============================================================
# Configuration
# ============================================================

SEED = 2026

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_DIR = Path(
    "/kaggle/working/recallface/data/simulated_sessions/pilot_503"
)

ORIGINAL_CHECKPOINT = Path(
    "/kaggle/working/recallface/checkpoints/FINAL_TRAINING/projector_best.pt"
)

RANKING_CHECKPOINT = Path(
    "/kaggle/working/recallface/checkpoints/PHASE7_RANKING_OBJECTIVE/projector_best.pt"
)


# ============================================================
# Reproducibility
# ============================================================

def build_session_split():

    files = sorted(
        DATASET_DIR.glob("session_*.pt")
    )

    if len(files) != 503:
        raise ValueError(
            f"Expected 503 sessions, found {len(files)}"
        )

    rng = random.Random(SEED)

    shuffled = files.copy()
    rng.shuffle(shuffled)

    split_index = int(
        len(shuffled) * 0.80
    )

    train_files = sorted(
        shuffled[:split_index],
        key=lambda p: p.name,
    )

    val_files = sorted(
        shuffled[split_index:],
        key=lambda p: p.name,
    )

    return train_files, val_files


# ============================================================
# Model loading
# ============================================================

def load_projector(checkpoint_path: Path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = DirectionProjector().to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# Spearman correlation
# ============================================================

def rankdata(values):

    n = len(values)

    sorted_indices = sorted(
        range(n),
        key=lambda i: values[i]
    )

    ranks = [0.0] * n

    i = 0

    while i < n:

        j = i + 1

        while (
            j < n
            and values[
                sorted_indices[j]
            ] == values[
                sorted_indices[i]
            ]
        ):
            j += 1

        average_rank = (
            (i + 1) + j
        ) / 2.0

        for k in range(i, j):

            ranks[
                sorted_indices[k]
            ] = average_rank

        i = j

    return ranks


def spearman_correlation(
    x,
    y,
):

    if len(x) < 2:
        return 0.0

    x_rank = rankdata(x)
    y_rank = rankdata(y)

    x_mean = sum(x_rank) / len(x_rank)
    y_mean = sum(y_rank) / len(y_rank)

    numerator = 0.0
    x_denominator = 0.0
    y_denominator = 0.0

    for xr, yr in zip(
        x_rank,
        y_rank,
    ):

        dx = xr - x_mean
        dy = yr - y_mean

        numerator += dx * dy
        x_denominator += dx * dx
        y_denominator += dy * dy

    denominator = (
        x_denominator
        * y_denominator
    ) ** 0.5

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


# ============================================================
# Evaluate one projector
# ============================================================

@torch.no_grad()
def evaluate_projector(
    projector,
    dataset,
):

    weighted_cosines = []

    alignment_similarities = []

    predicted_best_ranks = []

    rank1_count = 0

    transition_count = 0

    for index in range(
        len(dataset)
    ):

        item = dataset[index]

        current_latent = (
            item["current_latent"]
            .float()
            .to(DEVICE)
            .unsqueeze(0)
        )

        candidate_latents = (
            item["candidate_latents"]
            .float()
            .to(DEVICE)
        )

        similarity_scores = (
            item["similarity_scores"]
            .float()
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Predicted Projector direction
        # ----------------------------------------------------

        predicted_direction = projector(
            current_latent
        )[0]

        # ----------------------------------------------------
        # Candidate movements
        # ----------------------------------------------------

        observed_deltas = (
            candidate_latents
            - current_latent[0]
        )

        # ----------------------------------------------------
        # Valid candidates
        # ----------------------------------------------------

        valid_mask = (
            similarity_scores > 0
        )

        valid_indices = torch.where(
            valid_mask
        )[0]

        if valid_indices.numel() < 2:
            continue

        valid_deltas = (
            observed_deltas[
                valid_indices
            ]
        )

        valid_similarities = (
            similarity_scores[
                valid_indices
            ]
        )

        # ----------------------------------------------------
        # Similarity-weighted direction
        # ----------------------------------------------------

        valid_scores = (
            valid_similarities
        )

        preference_weights = torch.softmax(
            valid_scores / 0.10,
            dim=0,
        )

        weighted_direction = (
            valid_deltas
            * preference_weights.unsqueeze(1)
        ).sum(dim=0)

        # ----------------------------------------------------
        # Projector → weighted cosine
        # ----------------------------------------------------

        predicted_norm = F.normalize(
            predicted_direction.unsqueeze(0),
            p=2,
            dim=1,
            eps=1e-8,
        )[0]

        weighted_norm = F.normalize(
            weighted_direction.unsqueeze(0),
            p=2,
            dim=1,
            eps=1e-8,
        )[0]

        cosine = torch.sum(
            predicted_norm
            * weighted_norm
        ).item()

        weighted_cosines.append(
            cosine
        )

        # ----------------------------------------------------
        # Candidate alignment with Projector direction
        # ----------------------------------------------------

        normalized_candidates = F.normalize(
            valid_deltas,
            p=2,
            dim=1,
            eps=1e-8,
        )

        predicted_alignment = (
            normalized_candidates
            * predicted_norm.unsqueeze(0)
        ).sum(dim=1)

        alignment_values = (
            predicted_alignment
            .detach()
            .cpu()
            .tolist()
        )

        similarity_values = (
            valid_similarities
            .detach()
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # Spearman: Projector alignment vs similarity
        # ----------------------------------------------------

        correlation = spearman_correlation(
            alignment_values,
            similarity_values,
        )

        alignment_similarities.append(
            correlation
        )

        # ----------------------------------------------------
        # Actual best candidate
        # ----------------------------------------------------

        actual_best_position = int(
            torch.argmax(
                valid_similarities
            ).item()
        )

        # ----------------------------------------------------
        # Rank actual best candidate
        # according to Projector alignment
        # ----------------------------------------------------

        sorted_positions = sorted(
            range(
                len(alignment_values)
            ),
            key=lambda i:
                alignment_values[i],
            reverse=True,
        )

        predicted_rank = (
            sorted_positions.index(
                actual_best_position
            ) + 1
        )

        predicted_best_ranks.append(
            predicted_rank
        )

        if predicted_rank == 1:
            rank1_count += 1

        transition_count += 1

    return {
        "weighted_cosine": (
            sum(weighted_cosines)
            / len(weighted_cosines)
        ),

        "alignment_similarity": (
            sum(alignment_similarities)
            / len(alignment_similarities)
        ),

        "mean_predicted_best_rank": (
            sum(predicted_best_ranks)
            / len(predicted_best_ranks)
        ),

        "rank1_rate": (
            rank1_count
            / transition_count
        ),

        "transitions": transition_count,
    }


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========== PROJECTOR VALIDATION COMPARISON =========="
    )

    print(
        "Device:",
        DEVICE,
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Session split
    # --------------------------------------------------------

    _, val_files = build_session_split()

    print(
        "Validation sessions:",
        len(val_files),
    )

    dataset = NavigationTransitionDataset(
        val_files
    )

    print(
        "Validation transitions:",
        len(dataset),
    )

    # --------------------------------------------------------
    # Load original
    # --------------------------------------------------------

    original_projector, original_meta = (
        load_projector(
            ORIGINAL_CHECKPOINT
        )
    )

    # --------------------------------------------------------
    # Load ranking-aware
    # --------------------------------------------------------

    ranking_projector, ranking_meta = (
        load_projector(
            RANKING_CHECKPOINT
        )
    )

    print(
        "\nOriginal checkpoint epoch:",
        original_meta.get("epoch"),
    )

    print(
        "Ranking checkpoint epoch:",
        ranking_meta.get("epoch"),
    )

    # --------------------------------------------------------
    # Evaluate original
    # --------------------------------------------------------

    print(
        "\nEvaluating ORIGINAL..."
    )

    original_results = evaluate_projector(
        original_projector,
        dataset,
    )

    # --------------------------------------------------------
    # Evaluate ranking-aware
    # --------------------------------------------------------

    print(
        "Evaluating RANKING-AWARE..."
    )

    ranking_results = evaluate_projector(
        ranking_projector,
        dataset,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n========== ORIGINAL vs RANKING-AWARE =========="
    )

    print("\nORIGINAL")

    print(
        f"Projector → weighted cosine: "
        f"{original_results['weighted_cosine']:+.6f}"
    )

    print(
        f"Projector alignment → similarity: "
        f"{original_results['alignment_similarity']:+.6f}"
    )

    print(
        f"Mean predicted-best rank: "
        f"{original_results['mean_predicted_best_rank']:.3f}"
    )

    print(
        f"Rank-1 rate: "
        f"{original_results['rank1_rate']:.4f}"
    )

    print("\nRANKING-AWARE")

    print(
        f"Projector → weighted cosine: "
        f"{ranking_results['weighted_cosine']:+.6f}"
    )

    print(
        f"Projector alignment → similarity: "
        f"{ranking_results['alignment_similarity']:+.6f}"
    )

    print(
        f"Mean predicted-best rank: "
        f"{ranking_results['mean_predicted_best_rank']:.3f}"
    )

    print(
        f"Rank-1 rate: "
        f"{ranking_results['rank1_rate']:.4f}"
    )

    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    cosine_diff = (
        ranking_results["weighted_cosine"]
        - original_results["weighted_cosine"]
    )

    alignment_diff = (
        ranking_results["alignment_similarity"]
        - original_results["alignment_similarity"]
    )

    rank_diff = (
        ranking_results["mean_predicted_best_rank"]
        - original_results["mean_predicted_best_rank"]
    )

    rank1_diff = (
        ranking_results["rank1_rate"]
        - original_results["rank1_rate"]
    )

    print(
        "\nCHANGE (RANKING-AWARE - ORIGINAL)"
    )

    print(
        f"Weighted cosine change: "
        f"{cosine_diff:+.6f}"
    )

    print(
        f"Alignment→similarity change: "
        f"{alignment_diff:+.6f}"
    )

    print(
        f"Mean predicted-best-rank change: "
        f"{rank_diff:+.3f}"
    )

    print(
        f"Rank-1-rate change: "
        f"{rank1_diff:+.4f}"
    )

    print(
        "\n=============================================="
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import random
import sys
from pathlib import Path


# ============================================================
# Project root
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase4_projector import DirectionProjector
from phase5_sampler import ExplorationSampler
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

# IMPORTANT:
# Keep the original Phase 7 checkpoints untouched.
CHECKPOINT_DIR = Path(
    "/kaggle/working/recallface/checkpoints/PHASE7_RANKING_OBJECTIVE"
)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
MAX_GRAD_NORM = 1.0


# ------------------------------------------------------------
# Ranking-aware loss weights
# ------------------------------------------------------------

COSINE_LOSS_WEIGHT = 1.0
RANKING_LOSS_WEIGHT = 1.0
NLL_LOSS_WEIGHT = 1.0


# ------------------------------------------------------------
# Loss temperatures
# ------------------------------------------------------------

PREFERENCE_TEMPERATURE = 0.10
RANKING_TEMPERATURE = 0.10


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Session split
# ============================================================

def build_session_split():
    files = sorted(
        DATASET_DIR.glob("session_*.pt")
    )

    if len(files) != 503:
        raise ValueError(
            f"Expected 503 session files, found {len(files)}"
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

    train_ids = {
        p.name for p in train_files
    }

    val_ids = {
        p.name for p in val_files
    }

    if train_ids & val_ids:
        raise RuntimeError(
            "Train/validation session overlap detected."
        )

    return train_files, val_files


# ============================================================
# DataLoaders
# ============================================================

def build_dataloaders():

    train_files, val_files = build_session_split()

    train_dataset = NavigationTransitionDataset(
        train_files
    )

    val_dataset = NavigationTransitionDataset(
        val_files
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    )


# ============================================================
# Loss
# ============================================================

def compute_navigation_loss(
    projector: DirectionProjector,
    sampler: ExplorationSampler,
    batch: dict,
) -> tuple[torch.Tensor, dict[str, float]]:

    # --------------------------------------------------------
    # Input tensors
    # --------------------------------------------------------

    current_latent = batch[
        "current_latent"
    ].float().to(DEVICE)

    candidate_latents = batch[
        "candidate_latents"
    ].float().to(DEVICE)

    similarity_scores = batch[
        "similarity_scores"
    ].float().to(DEVICE)

    round_number = batch[
        "round_number"
    ].float().to(DEVICE).unsqueeze(1)

    # --------------------------------------------------------
    # Projector
    # --------------------------------------------------------

    predicted_direction = projector(
        current_latent
    )

    # --------------------------------------------------------
    # Sampler
    # --------------------------------------------------------

    predicted_variance = sampler(
        current_latent,
        round_number,
    )

    predicted_variance = torch.clamp(
        predicted_variance,
        min=1e-6,
        max=10.0,
    )

    # --------------------------------------------------------
    # Candidate movements
    # --------------------------------------------------------

    observed_deltas = (
        candidate_latents
        - current_latent.unsqueeze(1)
    )

    # --------------------------------------------------------
    # Valid candidate mask
    # --------------------------------------------------------

    valid_mask = (
        similarity_scores > 0
    )

    if not torch.all(
        valid_mask.any(dim=1)
    ):
        raise RuntimeError(
            "At least one training example has "
            "no valid candidates."
        )

    # --------------------------------------------------------
    # Similarity-weighted target distribution
    # --------------------------------------------------------

    masked_scores = similarity_scores.masked_fill(
        ~valid_mask,
        -1e9,
    )

    preference_weights = torch.softmax(
        masked_scores / PREFERENCE_TEMPERATURE,
        dim=1,
    )

    # --------------------------------------------------------
    # Similarity-weighted direction target
    # --------------------------------------------------------

    weighted_direction = (
        observed_deltas
        * preference_weights.unsqueeze(-1)
    ).sum(dim=1)

    # --------------------------------------------------------
    # 1. Preference-direction cosine loss
    # --------------------------------------------------------

    predicted_direction_normalized = F.normalize(
        predicted_direction,
        p=2,
        dim=1,
        eps=1e-8,
    )

    weighted_direction_normalized = F.normalize(
        weighted_direction,
        p=2,
        dim=1,
        eps=1e-8,
    )

    cosine_alignment = (
        predicted_direction_normalized
        * weighted_direction_normalized
    ).sum(dim=1)

    preference_cosine_loss = (
        1.0 - cosine_alignment
    ).mean()

    # --------------------------------------------------------
    # 2. Candidate-ranking loss
    # --------------------------------------------------------
    #
    # The predicted direction should rank candidate
    # movements according to witness similarity.
    #
    # Higher cosine alignment with the predicted direction
    # should correspond to higher witness similarity.
    # --------------------------------------------------------

    candidate_directions_normalized = F.normalize(
        observed_deltas,
        p=2,
        dim=2,
        eps=1e-8,
    )

    predicted_candidate_alignment = (
        candidate_directions_normalized
        * predicted_direction_normalized.unsqueeze(1)
    ).sum(dim=2)

    predicted_candidate_alignment = (
        predicted_candidate_alignment.masked_fill(
            ~valid_mask,
            -1e9,
        )
    )

    target_distribution = (
        preference_weights.detach()
    )

    predicted_distribution = torch.softmax(
        predicted_candidate_alignment
        / RANKING_TEMPERATURE,
        dim=1,
    )

    ranking_loss = -(
        target_distribution
        * torch.log(
            predicted_distribution + 1e-8
        )
    ).sum(dim=1).mean()

    # --------------------------------------------------------
    # 3. Similarity-weighted Gaussian NLL
    # --------------------------------------------------------

    mean = predicted_direction.unsqueeze(1)

    variance = predicted_variance.unsqueeze(1)

    per_candidate_nll = 0.5 * (
        torch.log(variance)
        + (
            (observed_deltas - mean) ** 2
            / variance
        )
    )

    # Average across latent dimensions.
    per_candidate_nll = (
        per_candidate_nll.mean(dim=2)
    )

    weighted_nll = (
        per_candidate_nll
        * preference_weights
    ).sum(dim=1).mean()

    # --------------------------------------------------------
    # 4. Combined ranking-aware loss
    # --------------------------------------------------------

    total_loss = (
        COSINE_LOSS_WEIGHT
        * preference_cosine_loss

        + RANKING_LOSS_WEIGHT
        * ranking_loss

        + NLL_LOSS_WEIGHT
        * weighted_nll
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {
        "total_loss": float(
            total_loss.detach().item()
        ),

        "preference_cosine_loss": float(
            preference_cosine_loss
            .detach()
            .item()
        ),

        "ranking_loss": float(
            ranking_loss
            .detach()
            .item()
        ),

        "nll_loss": float(
            weighted_nll
            .detach()
            .item()
        ),

        "mean_variance": float(
            predicted_variance
            .detach()
            .mean()
            .item()
        ),
    }

    return total_loss, metrics


# ============================================================
# One epoch
# ============================================================

def train_one_epoch(
    projector: DirectionProjector,
    sampler: ExplorationSampler,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
) -> dict[str, float]:

    projector.train()
    sampler.train()

    totals = {
        "total_loss": 0.0,
        "preference_cosine_loss": 0.0,
        "ranking_loss": 0.0,
        "nll_loss": 0.0,
        "mean_variance": 0.0,
    }

    batches = 0

    for batch in loader:

        optimizer.zero_grad(
            set_to_none=True
        )

        loss, metrics = compute_navigation_loss(
            projector,
            sampler,
            batch,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training loss became NaN/Inf."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(projector.parameters())
            + list(sampler.parameters()),
            MAX_GRAD_NORM,
        )

        optimizer.step()

        for key in totals:
            totals[key] += metrics[key]

        batches += 1

    return {
        key: value / batches
        for key, value in totals.items()
    }


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    projector: DirectionProjector,
    sampler: ExplorationSampler,
    loader: DataLoader,
) -> dict[str, float]:

    projector.eval()
    sampler.eval()

    totals = {
        "total_loss": 0.0,
        "preference_cosine_loss": 0.0,
        "ranking_loss": 0.0,
        "nll_loss": 0.0,
        "mean_variance": 0.0,
    }

    batches = 0

    for batch in loader:

        loss, metrics = compute_navigation_loss(
            projector,
            sampler,
            batch,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Validation loss became NaN/Inf."
            )

        for key in totals:
            totals[key] += metrics[key]

        batches += 1

    return {
        key: value / batches
        for key, value in totals.items()
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    print(
        "========== PHASE 7 RANKING-AWARE TRAINING =========="
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

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = build_dataloaders()

    print(
        "Training sessions     :",
        402,
    )

    print(
        "Validation sessions   :",
        101,
    )

    print(
        "Training transitions  :",
        len(train_dataset),
    )

    print(
        "Validation transitions:",
        len(val_dataset),
    )

    print(
        "Checkpoint directory  :",
        CHECKPOINT_DIR,
    )

    # --------------------------------------------------------
    # Loss configuration
    # --------------------------------------------------------

    print(
        "\nLoss configuration:"
    )

    print(
        "Preference cosine weight:",
        COSINE_LOSS_WEIGHT,
    )

    print(
        "Ranking loss weight     :",
        RANKING_LOSS_WEIGHT,
    )

    print(
        "NLL loss weight         :",
        NLL_LOSS_WEIGHT,
    )

    print(
        "Preference temperature  :",
        PREFERENCE_TEMPERATURE,
    )

    print(
        "Ranking temperature     :",
        RANKING_TEMPERATURE,
    )

    # --------------------------------------------------------
    # Clean initialization
    # --------------------------------------------------------

    projector = DirectionProjector().to(
        DEVICE
    )

    sampler = ExplorationSampler().to(
        DEVICE
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        list(projector.parameters())
        + list(sampler.parameters()),
        lr=LEARNING_RATE,
    )

    print(
        "\nOptimizer: Adam"
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Epochs:",
        EPOCHS,
    )

    # --------------------------------------------------------
    # Initial validation
    # --------------------------------------------------------

    initial_val = validate(
        projector,
        sampler,
        val_loader,
    )

    print(
        "\nInitial validation:"
    )

    print(
        f"total={initial_val['total_loss']:.6f}, "
        f"cosine={initial_val['preference_cosine_loss']:.6f}, "
        f"ranking={initial_val['ranking_loss']:.6f}, "
        f"nll={initial_val['nll_loss']:.6f}, "
        f"variance={initial_val['mean_variance']:.6f}"
    )

    # IMPORTANT:
    # Initialize from the clean model's validation loss.
    # This prevents saving a checkpoint that is worse
    # than the initial model.
    best_val_loss = (
        initial_val["total_loss"]
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_metrics = train_one_epoch(
            projector,
            sampler,
            optimizer,
            train_loader,
        )

        val_metrics = validate(
            projector,
            sampler,
            val_loader,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }

        history.append(
            epoch_record
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            "Train:"
            f" total={train_metrics['total_loss']:.6f},"
            f" cosine={train_metrics['preference_cosine_loss']:.6f},"
            f" ranking={train_metrics['ranking_loss']:.6f},"
            f" nll={train_metrics['nll_loss']:.6f}"
        )

        print(
            "Validation:"
            f" total={val_metrics['total_loss']:.6f},"
            f" cosine={val_metrics['preference_cosine_loss']:.6f},"
            f" ranking={val_metrics['ranking_loss']:.6f},"
            f" nll={val_metrics['nll_loss']:.6f}"
        )

        print(
            "Mean variance:",
            f"{val_metrics['mean_variance']:.6f}",
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["total_loss"]
            < best_val_loss
        ):

            best_val_loss = (
                val_metrics["total_loss"]
            )

            CHECKPOINT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            projector_path = (
                CHECKPOINT_DIR
                / "projector_best.pt"
            )

            sampler_path = (
                CHECKPOINT_DIR
                / "sampler_best.pt"
            )

            torch.save(
                {
                    "model_state_dict":
                        projector.state_dict(),

                    "epoch":
                        epoch,

                    "validation_loss":
                        val_metrics[
                            "total_loss"
                        ],

                    "config": {
                        "seed":
                            SEED,

                        "batch_size":
                            BATCH_SIZE,

                        "learning_rate":
                            LEARNING_RATE,

                        "cosine_loss_weight":
                            COSINE_LOSS_WEIGHT,

                        "ranking_loss_weight":
                            RANKING_LOSS_WEIGHT,

                        "nll_loss_weight":
                            NLL_LOSS_WEIGHT,

                        "preference_temperature":
                            PREFERENCE_TEMPERATURE,

                        "ranking_temperature":
                            RANKING_TEMPERATURE,

                        "max_grad_norm":
                            MAX_GRAD_NORM,
                    },
                },
                projector_path,
            )

            torch.save(
                {
                    "model_state_dict":
                        sampler.state_dict(),

                    "epoch":
                        epoch,

                    "validation_loss":
                        val_metrics[
                            "total_loss"
                        ],

                    "config": {
                        "seed":
                            SEED,

                        "batch_size":
                            BATCH_SIZE,

                        "learning_rate":
                            LEARNING_RATE,

                        "cosine_loss_weight":
                            COSINE_LOSS_WEIGHT,

                        "ranking_loss_weight":
                            RANKING_LOSS_WEIGHT,

                        "nll_loss_weight":
                            NLL_LOSS_WEIGHT,

                        "preference_temperature":
                            PREFERENCE_TEMPERATURE,

                        "ranking_temperature":
                            RANKING_TEMPERATURE,

                        "max_grad_norm":
                            MAX_GRAD_NORM,
                    },
                },
                sampler_path,
            )

            print(
                "Saved new best ranking-aware checkpoints."
            )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = (
        CHECKPOINT_DIR
        / "training_history.pt"
    )

    torch.save(
        history,
        history_path,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "PHASE 7 RANKING-AWARE TRAINING COMPLETE"
    )

    print(
        "Best validation loss:",
        best_val_loss,
    )

    print(
        "Checkpoint directory:",
        CHECKPOINT_DIR,
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()
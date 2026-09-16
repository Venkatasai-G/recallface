from __future__ import annotations

import random
from pathlib import Path

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

CHECKPOINT_DIR = Path(
    "/kaggle/working/recallface/checkpoints/FINAL_TRAINING"
)

BATCH_SIZE = 32

# We will initially test with 1 epoch.
EPOCHS = 1

LEARNING_RATE = 1e-4

# Weight of the selected-movement supervision.
DIRECTION_LOSS_WEIGHT = 1.0

# Weight of the candidate-distribution NLL.
NLL_LOSS_WEIGHT = 1.0

MAX_GRAD_NORM = 1.0


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

    train_ids = {p.name for p in train_files}
    val_ids = {p.name for p in val_files}

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

    current_latent = batch[
        "current_latent"
    ].float().to(DEVICE)

    target_direction = batch[
        "target_direction"
    ].float().to(DEVICE)

    candidate_latents = batch[
        "candidate_latents"
    ].float().to(DEVICE)

    round_number = batch[
        "round_number"
    ].float().to(DEVICE).unsqueeze(1)

    # --------------------------------------------------------
    # Projector prediction
    # --------------------------------------------------------

    predicted_direction = projector(
        current_latent
    )

    # --------------------------------------------------------
    # Sampler prediction
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
    # 1. Selected-candidate direction loss
    # --------------------------------------------------------

    direction_loss = F.mse_loss(
        predicted_direction,
        target_direction,
    )

    # --------------------------------------------------------
    # 2. Candidate-distribution negative log likelihood
    # --------------------------------------------------------

    observed_deltas = (
        candidate_latents
        - current_latent.unsqueeze(1)
    )

    mean = predicted_direction.unsqueeze(1)

    variance = predicted_variance.unsqueeze(1)

    nll = 0.5 * (
        torch.log(variance)
        + (
            (observed_deltas - mean) ** 2
            / variance
        )
    )

    nll_loss = nll.mean()

    # --------------------------------------------------------
    # Combined loss
    # --------------------------------------------------------

    total_loss = (
        DIRECTION_LOSS_WEIGHT * direction_loss
        + NLL_LOSS_WEIGHT * nll_loss
    )

    metrics = {
        "total_loss": float(total_loss.detach().item()),
        "direction_loss": float(
            direction_loss.detach().item()
        ),
        "nll_loss": float(
            nll_loss.detach().item()
        ),
        "mean_variance": float(
            predicted_variance.detach().mean().item()
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
        "direction_loss": 0.0,
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
        "direction_loss": 0.0,
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

    print("========== PHASE 7 TRAINING ==========")
    print("Device:", DEVICE)

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
        "Training sessions     : 402"
    )

    print(
        "Validation sessions   : 101"
    )

    print(
        "Training transitions  :",
        len(train_dataset),
    )

    print(
        "Validation transitions:",
        len(val_dataset),
    )

    # --------------------------------------------------------
    # Clean initialization
    # --------------------------------------------------------

    projector = DirectionProjector().to(DEVICE)

    sampler = ExplorationSampler().to(DEVICE)

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

    print("\nInitial validation:")
    print(
        f"total={initial_val['total_loss']:.6f}, "
        f"direction={initial_val['direction_loss']:.6f}, "
        f"nll={initial_val['nll_loss']:.6f}, "
        f"variance={initial_val['mean_variance']:.6f}"
    )

    best_val_loss = initial_val["total_loss"]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    for epoch in range(1, EPOCHS + 1):

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

        history.append(epoch_record)

        print(f"\nEpoch {epoch}/{EPOCHS}")

        print(
            "Train:"
            f" total={train_metrics['total_loss']:.6f},"
            f" direction={train_metrics['direction_loss']:.6f},"
            f" nll={train_metrics['nll_loss']:.6f}"
        )

        print(
            "Validation:"
            f" total={val_metrics['total_loss']:.6f},"
            f" direction={val_metrics['direction_loss']:.6f},"
            f" nll={val_metrics['nll_loss']:.6f}"
        )

        print(
            "Mean variance:",
            f"{val_metrics['mean_variance']:.6f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_metrics["total_loss"] < best_val_loss:

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
                    "epoch": epoch,
                    "validation_loss":
                        val_metrics["total_loss"],
                    "config": {
                        "seed": SEED,
                        "batch_size": BATCH_SIZE,
                        "learning_rate":
                            LEARNING_RATE,
                        "direction_loss_weight":
                            DIRECTION_LOSS_WEIGHT,
                        "nll_loss_weight":
                            NLL_LOSS_WEIGHT,
                    },
                },
                projector_path,
            )

            torch.save(
                {
                    "model_state_dict":
                        sampler.state_dict(),
                    "epoch": epoch,
                    "validation_loss":
                        val_metrics["total_loss"],
                    "config": {
                        "seed": SEED,
                        "batch_size": BATCH_SIZE,
                        "learning_rate":
                            LEARNING_RATE,
                        "direction_loss_weight":
                            DIRECTION_LOSS_WEIGHT,
                        "nll_loss_weight":
                            NLL_LOSS_WEIGHT,
                    },
                },
                sampler_path,
            )

            print(
                "Saved new best checkpoints."
            )

    history_path = (
        CHECKPOINT_DIR
        / "training_history.pt"
    )

    torch.save(
        history,
        history_path,
    )

    print(
        "\nPHASE 7 TRAINING TEST COMPLETE"
    )

    print(
        "Best validation loss:",
        best_val_loss,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_DIR = Path(
    "/kaggle/working/recallface/data/simulated_sessions/pilot_503"
)

BATCH_SIZE = 8


# ============================================================
# Build datasets
# ============================================================

def build_datasets():
    session_files = sorted(
        DATASET_DIR.glob("session_*.pt")
    )

    if len(session_files) != 503:
        raise ValueError(
            f"Expected 503 sessions, found {len(session_files)}"
        )

    # Reproduce exactly the split we already verified.
    import random

    rng = random.Random(2026)

    shuffled_files = session_files.copy()
    rng.shuffle(shuffled_files)

    split_index = int(
        len(shuffled_files) * 0.80
    )

    train_files = sorted(
        shuffled_files[:split_index],
        key=lambda p: p.name,
    )

    val_files = sorted(
        shuffled_files[split_index:],
        key=lambda p: p.name,
    )

    train_dataset = NavigationTransitionDataset(
        train_files
    )

    val_dataset = NavigationTransitionDataset(
        val_files
    )

    return train_dataset, val_dataset


# ============================================================
# One-batch sanity test
# ============================================================

def main():
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_dataset, val_dataset = build_datasets()

    print(
        "Training transitions:",
        len(train_dataset)
    )

    print(
        "Validation transitions:",
        len(val_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    batch = next(iter(train_loader))

    print("\nBatch shapes:")
    print(
        "current_latent:",
        tuple(batch["current_latent"].shape)
    )

    print(
        "target_direction:",
        tuple(batch["target_direction"].shape)
    )

    print(
        "candidate_latents:",
        tuple(batch["candidate_latents"].shape)
    )

    print(
        "similarity_scores:",
        tuple(batch["similarity_scores"].shape)
    )

    # --------------------------------------------------------
    # Move required tensors to GPU
    # --------------------------------------------------------

    current_latent = batch[
        "current_latent"
    ].float().to(DEVICE)

    target_direction = batch[
        "target_direction"
    ].float().to(DEVICE)

    round_number = batch[
        "round_number"
    ].float().to(DEVICE).unsqueeze(1)

    candidate_latents = batch[
        "candidate_latents"
    ].float().to(DEVICE)

    # --------------------------------------------------------
    # Create clean models
    # --------------------------------------------------------

    projector = DirectionProjector().to(DEVICE)

    sampler = ExplorationSampler().to(DEVICE)

    projector.train()
    sampler.train()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    predicted_direction = projector(
        current_latent
    )

    predicted_variance = sampler(
        current_latent,
        round_number,
    )

    print("\nModel outputs:")
    print(
        "predicted_direction:",
        tuple(predicted_direction.shape)
    )

    print(
        "predicted_variance:",
        tuple(predicted_variance.shape)
    )

    print(
        "variance minimum:",
        predicted_variance.min().item()
    )

    print(
        "variance maximum:",
        predicted_variance.max().item()
    )

    # --------------------------------------------------------
    # Initial direction loss
    # --------------------------------------------------------

    direction_loss = F.mse_loss(
        predicted_direction,
        target_direction,
    )

    print(
        "\nDirection loss:",
        direction_loss.item()
    )

    # --------------------------------------------------------
    # Candidate movement around current state
    # --------------------------------------------------------

    predicted_std = torch.sqrt(
        predicted_variance
    )

    # Use the observed selected candidate movement
    # only for checking dimensions at this stage.
    observed_selected = batch[
        "selected_latent"
    ].float().to(DEVICE)

    observed_movement = (
        observed_selected - current_latent
    )

    print(
        "observed_movement:",
        tuple(observed_movement.shape)
    )

    # Basic finite checks
    checks = {
        "current_latent": torch.isfinite(
            current_latent
        ).all(),

        "target_direction": torch.isfinite(
            target_direction
        ).all(),

        "predicted_direction": torch.isfinite(
            predicted_direction
        ).all(),

        "predicted_variance": torch.isfinite(
            predicted_variance
        ).all(),

        "predicted_std": torch.isfinite(
            predicted_std
        ).all(),
    }

    print("\nFinite-value checks:")

    for name, result in checks.items():
        print(
            f"{name}: {bool(result.item())}"
        )

    if not all(
        bool(result.item())
        for result in checks.values()
    ):
        raise RuntimeError(
            "A NaN/Inf value was detected."
        )

    print(
        "\nPHASE 7 SANITY TEST PASSED"
    )


if __name__ == "__main__":
    main()
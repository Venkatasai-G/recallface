from pathlib import Path

import torch

from phase4_projector.direction_projector import DirectionProjector
from phase5_sampler.exploration_sampler import (
    ExplorationSampler,
)
from phase6_simulated_data.diffae_candidate_generator import (
    DiffAECandidateGenerator,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE7_FINAL_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "PHASE7_FINAL"
)


def find_checkpoint(filename):
    """
    Find a Phase 7 FINAL checkpoint.

    Supports:
    1. Standard repository filename:
       checkpoints/PHASE7_FINAL/projector_best.pt

    2. Existing Windows filename:
       checkpoints/PHASE7_FINAL/PHASE7_FINAL_projector_best.pt
    """

    standard_path = PHASE7_FINAL_DIR / filename

    windows_path = (
        PHASE7_FINAL_DIR
        / f"PHASE7_FINAL_{filename}"
    )

    print(f"\nSearching for: {filename}")

    print("Standard path:")
    print(standard_path)
    print("Exists:", standard_path.is_file())

    if standard_path.is_file():
        print("FOUND:", standard_path)
        return standard_path

    print("\nWindows path:")
    print(windows_path)
    print("Exists:", windows_path.is_file())

    if windows_path.is_file():
        print("FOUND:", windows_path)
        return windows_path

    raise FileNotFoundError(
        "Phase 7 FINAL checkpoint not found.\n"
        f"Checked:\n"
        f"  {standard_path}\n"
        f"  {windows_path}"
    )


PROJECTOR_CHECKPOINT = find_checkpoint(
    "projector_best.pt"
)

SAMPLER_CHECKPOINT = find_checkpoint(
    "sampler_best.pt"
)


# ============================================================
# Load Phase 7 navigation models
# ============================================================

def load_navigation_models(device=None):
    """
    Load the trained Phase 7 FINAL Projector and Sampler.

    Returns:
        projector: trained DirectionProjector
        sampler: trained ExplorationSampler
        device: torch.device
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    print("\n" + "=" * 60)
    print("Loading Phase 7 FINAL navigation models")
    print("=" * 60)

    print(
        "Projector checkpoint:",
        PROJECTOR_CHECKPOINT,
    )

    print(
        "Sampler checkpoint:",
        SAMPLER_CHECKPOINT,
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Create exact Phase 7 architectures
    # --------------------------------------------------------

    projector = DirectionProjector().to(device)
    sampler = ExplorationSampler().to(device)

    # --------------------------------------------------------
    # Load checkpoints
    # --------------------------------------------------------

    projector_checkpoint = torch.load(
        PROJECTOR_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    sampler_checkpoint = torch.load(
        SAMPLER_CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Validate checkpoint structure
    # --------------------------------------------------------

    if "model_state_dict" not in projector_checkpoint:
        raise KeyError(
            "Projector checkpoint does not contain "
            "'model_state_dict'."
        )

    if "model_state_dict" not in sampler_checkpoint:
        raise KeyError(
            "Sampler checkpoint does not contain "
            "'model_state_dict'."
        )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    projector.load_state_dict(
        projector_checkpoint["model_state_dict"]
    )

    sampler.load_state_dict(
        sampler_checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    projector.eval()
    sampler.eval()

    print("\nProjector loaded successfully.")
    print("Sampler loaded successfully.")

    print(
        "Projector training epoch:",
        projector_checkpoint.get("epoch"),
    )

    print(
        "Sampler training epoch:",
        sampler_checkpoint.get("epoch"),
    )

    print(
        "Validation loss:",
        projector_checkpoint.get(
            "validation_loss"
        ),
    )

    print("=" * 60)

    return projector, sampler, device


# ============================================================
# Load Phase 6 DiffAE candidate generator
# ============================================================

def load_candidate_generator(device=None):
    """
    Load the existing Phase 6 DiffAE candidate generator.

    Phase 11 deliberately reuses the Phase 6 implementation
    instead of duplicating DiffAE loading/inference code.
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("\n" + "=" * 60)
    print("Loading Phase 6 DiffAE candidate generator")
    print("=" * 60)

    generator = DiffAECandidateGenerator(
        device=str(device)
    )

    generator.load()

    print("DiffAE candidate generator loaded successfully.")
    print("=" * 60)

    return generator
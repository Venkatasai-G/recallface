from pathlib import Path

import torch

from phase4_projector.direction_projector import DirectionProjector
from phase5_sampler.exploration_sampler import ExplorationSampler
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

# Kaggle working directory.
#
# The Phase 7 checkpoints may exist directly here depending
# on how the Kaggle environment was prepared.
KAGGLE_WORKING_DIR = Path("/kaggle/working")


# ============================================================
# Checkpoint resolution
# ============================================================

def _find_checkpoint(
    standard_name,
    windows_name,
    root_name,
):
    """
    Find a Phase 7 FINAL checkpoint.

    Supported layouts:

    1. Standard repository layout:
       checkpoints/
       └── PHASE7_FINAL/
           ├── projector_best.pt
           └── sampler_best.pt

    2. Existing Windows layout:
       checkpoints/
       └── PHASE7_FINAL/
           ├── PHASE7_FINAL_projector_best.pt
           └── PHASE7_FINAL_sampler_best.pt

    3. Kaggle working-directory layout:
       /kaggle/working/
       ├── PHASE7_FINAL_projector_best.pt
       └── PHASE7_FINAL_sampler_best.pt
    """

    candidates = [
        PHASE7_FINAL_DIR / standard_name,
        PHASE7_FINAL_DIR / windows_name,
        KAGGLE_WORKING_DIR / root_name,
    ]

    for path in candidates:
        if path.exists():
            return path

    checked_paths = "\n".join(
        f"  {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        "Phase 7 FINAL checkpoint not found.\n"
        "Checked:\n"
        f"{checked_paths}"
    )


PROJECTOR_CHECKPOINT = _find_checkpoint(
    standard_name="projector_best.pt",
    windows_name="PHASE7_FINAL_projector_best.pt",
    root_name="PHASE7_FINAL_projector_best.pt",
)

SAMPLER_CHECKPOINT = _find_checkpoint(
    standard_name="sampler_best.pt",
    windows_name="PHASE7_FINAL_sampler_best.pt",
    root_name="PHASE7_FINAL_sampler_best.pt",
)


# ============================================================
# Phase 7 model loading
# ============================================================

def load_navigation_models(device=None):
    """
    Load the trained Phase 7 Projector and Sampler.

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

    # --------------------------------------------------------
    # Verify checkpoint files
    # --------------------------------------------------------

    if not PROJECTOR_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Projector checkpoint not found:\n"
            f"{PROJECTOR_CHECKPOINT}"
        )

    if not SAMPLER_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Sampler checkpoint not found:\n"
            f"{SAMPLER_CHECKPOINT}"
        )

    print("=" * 60)
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
    # Create the exact Phase 7 architectures
    # --------------------------------------------------------

    projector = DirectionProjector().to(device)
    sampler = ExplorationSampler().to(device)

    # --------------------------------------------------------
    # Load trained checkpoint dictionaries
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

    print("Projector loaded successfully.")
    print("Sampler loaded successfully.")
    print("=" * 60)

    return projector, sampler, device


# ============================================================
# Phase 6 DiffAE candidate generator
# ============================================================

def load_candidate_generator(device=None):
    """
    Load the existing Phase 6 DiffAE candidate generator.

    Phase 11 intentionally reuses the Phase 6 implementation
    instead of duplicating DiffAE loading and inference logic.
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("=" * 60)
    print("Loading Phase 6 DiffAE candidate generator")
    print("=" * 60)

    generator = DiffAECandidateGenerator(
        device=str(device)
    )

    generator.load()

    print("DiffAE candidate generator loaded successfully.")
    print("=" * 60)

    return generator
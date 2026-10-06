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


# ============================================================
# Checkpoint locations
# ============================================================

# Normal project checkpoint directory.
PROJECT_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "PHASE7_FINAL"
)

# Kaggle working directory.
#
# IMPORTANT:
# In your current Kaggle environment the Phase 7 FINAL
# checkpoints are located directly here:
#
# /kaggle/working/PHASE7_FINAL_projector_best.pt
# /kaggle/working/PHASE7_FINAL_sampler_best.pt
#
KAGGLE_WORKING_DIR = Path("/kaggle/working")


def find_checkpoint(filename):
    """
    Find a Phase 7 FINAL checkpoint.

    Supported locations:

    1. Project standard layout:
       recallface/checkpoints/PHASE7_FINAL/<filename>

    2. Kaggle working directory:
       /kaggle/working/<filename>

    3. Existing Windows naming:
       recallface/checkpoints/PHASE7_FINAL/
       PHASE7_FINAL_<filename>
    """

    # --------------------------------------------------------
    # Standard project filename
    # --------------------------------------------------------

    standard_path = (
        PROJECT_CHECKPOINT_DIR
        / filename
    )

    # --------------------------------------------------------
    # Kaggle root filename
    # --------------------------------------------------------

    kaggle_path = (
        KAGGLE_WORKING_DIR
        / filename
    )

    # --------------------------------------------------------
    # Existing Windows filename
    # --------------------------------------------------------

    windows_path = (
        PROJECT_CHECKPOINT_DIR
        / f"PHASE7_FINAL_{filename}"
    )

    candidates = [
        standard_path,
        kaggle_path,
        windows_path,
    ]

    print(f"\nSearching for checkpoint: {filename}")

    for path in candidates:
        print(f"  Checking: {path}")
        print(f"  Exists:   {path.exists()}")

        if path.is_file():
            print(f"  FOUND:    {path}")
            return path

    checked = "\n".join(
        f"  {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        "\nPhase 7 FINAL checkpoint was not found.\n\n"
        "Checked these locations:\n"
        f"{checked}"
    )


# ============================================================
# Resolve Phase 7 checkpoints
# ============================================================

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
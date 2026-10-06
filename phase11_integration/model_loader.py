from pathlib import Path
import torch

from phase4_projector.direction_projector import DirectionProjector
from phase5_sampler.exploration_sampler import ExplorationSampler
from phase6_simulated_data.diffae_candidate_generator import DiffAECandidateGenerator


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE7_FINAL_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "PHASE7_FINAL"
)


def _find_checkpoint(preferred_name, fallback_name):
    """
    Find a Phase 7 FINAL checkpoint using the standard Kaggle
    filename first and the existing Windows filename as fallback.
    """

    preferred = PHASE7_FINAL_DIR / preferred_name

    if preferred.exists():
        return preferred

    fallback = PHASE7_FINAL_DIR / fallback_name

    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "Phase 7 FINAL checkpoint not found.\n"
        f"Checked:\n"
        f"  {preferred}\n"
        f"  {fallback}"
    )


PROJECTOR_CHECKPOINT = _find_checkpoint(
    "projector_best.pt",
    "PHASE7_FINAL_projector_best.pt",
)

SAMPLER_CHECKPOINT = _find_checkpoint(
    "sampler_best.pt",
    "PHASE7_FINAL_sampler_best.pt",
)


def load_navigation_models(device=None):
    """
    Load the trained Phase 7 Projector and Sampler.

    Returns:
        projector: trained DirectionProjector
        sampler: trained ExplorationSampler
        device: torch.device
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if not PROJECTOR_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Projector checkpoint not found:\n{PROJECTOR_CHECKPOINT}"
        )

    if not SAMPLER_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Sampler checkpoint not found:\n{SAMPLER_CHECKPOINT}"
        )

    # Create the exact same architectures used during Phase 7 training.
    projector = DirectionProjector().to(device)
    sampler = ExplorationSampler().to(device)

    # Load trained weights.
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

    projector.load_state_dict(
        projector_checkpoint["model_state_dict"]
    )

    sampler.load_state_dict(
        sampler_checkpoint["model_state_dict"]
    )

    projector.eval()
    sampler.eval()

    return projector, sampler, device


def load_candidate_generator(device=None):
    """
    Load the existing Phase 6 DiffAE candidate generator.

    Phase 11 reuses the Phase 6 DiffAE implementation instead
    of duplicating DiffAE loading/inference code.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = DiffAECandidateGenerator(device=str(device))
    generator.load()

    return generator
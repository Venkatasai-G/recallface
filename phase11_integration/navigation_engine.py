from pathlib import Path

import numpy as np
import torch

from phase11_integration.model_loader import (
    load_navigation_models,
    load_candidate_generator,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LATENT_STATS_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "diffae"
    / "checkpoints"
    / "latent.pkl"
)


class NavigationEngine:
    """
    Phase 11 navigation engine.

    Combines:
        Phase 7 trained Projector
        Phase 7 trained Sampler
        Phase 6 latent safety logic
        Phase 6 DiffAE candidate generator
    """

    def __init__(
        self,
        device=None,
        num_candidates=12,
        latent_clip=3.0,
    ):
        if not 12 <= num_candidates <= 20:
            raise ValueError(
                "num_candidates must be between 12 and 20."
            )

        self.num_candidates = num_candidates
        self.latent_clip = latent_clip

        # Load trained Phase 7 navigation models.
        (
            self.projector,
            self.sampler,
            self.device,
        ) = load_navigation_models(device)

        # Load the existing Phase 6 DiffAE generator.
        self.candidate_generator = load_candidate_generator(
            self.device
        )

        # Load empirical latent statistics used by Phase 6
        # for latent safety clamping.
        self.latent_mean, self.latent_std = self._load_latent_stats()

    def _load_latent_stats(self):
        """
        Load DiffAE latent mean and standard deviation.

        These are the same statistics used by the Phase 6
        SessionSimulator for safety clamping.
        """

        if not LATENT_STATS_PATH.exists():
            raise FileNotFoundError(
                f"Latent statistics not found:\n{LATENT_STATS_PATH}"
            )

        latent_data = torch.load(
            LATENT_STATS_PATH,
            map_location="cpu",
            weights_only=False,
        )

        if "conds_mean" not in latent_data:
            raise KeyError(
                "latent.pkl does not contain 'conds_mean'."
            )

        if "conds_std" not in latent_data:
            raise KeyError(
                "latent.pkl does not contain 'conds_std'."
            )

        latent_mean = latent_data["conds_mean"]
        latent_std = latent_data["conds_std"]

        if not isinstance(latent_mean, torch.Tensor):
            latent_mean = torch.as_tensor(latent_mean)

        if not isinstance(latent_std, torch.Tensor):
            latent_std = torch.as_tensor(latent_std)

        latent_mean = latent_mean.float()
        latent_std = latent_std.float()

        if latent_mean.shape != (512,):
            raise ValueError(
                f"Expected latent mean shape (512,), "
                f"got {tuple(latent_mean.shape)}"
            )

        if latent_std.shape != (512,):
            raise ValueError(
                f"Expected latent std shape (512,), "
                f"got {tuple(latent_std.shape)}"
            )

        if torch.any(latent_std <= 0):
            raise ValueError(
                "Latent standard deviation must be positive."
            )

        return latent_mean, latent_std

    def clamp_latents(self, latents):
        """
        Apply the same empirical-distribution safety clamp
        used by Phase 6.

        Each latent dimension is limited to ±latent_clip
        standard deviations from the empirical latent mean.
        """

        if not isinstance(latents, torch.Tensor):
            latents = torch.as_tensor(
                latents,
                dtype=torch.float32,
            )

        latents = latents.float()

        if latents.ndim != 2 or latents.shape[1] != 512:
            raise ValueError(
                "latents must have shape (N, 512)."
            )

        mean = self.latent_mean.to(latents.device)
        std = self.latent_std.to(latents.device)

        z_scores = (latents - mean) / std

        z_scores = torch.clamp(
            z_scores,
            -self.latent_clip,
            self.latent_clip,
        )

        return z_scores * std + mean

    @torch.no_grad()
    def generate_candidate_latents(
        self,
        current_latent,
        round_number,
    ):
        """
        Generate candidate latent vectors for one navigation round.

        Formula follows the Phase 6 navigation logic:

            candidate =
                current_latent
                + direction
                + noise * sqrt(variance)
        """

        if not isinstance(current_latent, torch.Tensor):
            current_latent = torch.as_tensor(
                current_latent,
                dtype=torch.float32,
            )

        current_latent = current_latent.float()

        if current_latent.ndim == 1:
            current_latent = current_latent.unsqueeze(0)

        if current_latent.shape != (1, 512):
            raise ValueError(
                "current_latent must have shape (512,) "
                "or (1, 512)."
            )

        current_latent = current_latent.to(self.device)

        round_tensor = torch.tensor(
            [[float(round_number)]],
            dtype=torch.float32,
            device=self.device,
        )

        # Phase 7 learned direction.
        direction = self.projector(current_latent)

        # Phase 7 learned exploration variance.
        variance = self.sampler(
            current_latent,
            round_tensor,
        )

        # Random exploration noise.
        noise = torch.randn(
            self.num_candidates,
            512,
            device=self.device,
        )

        # Expand current latent to all candidates.
        current_expanded = current_latent.expand(
            self.num_candidates,
            -1,
        )

        candidates = (
            current_expanded
            + direction.expand(self.num_candidates, -1)
            + noise * torch.sqrt(variance.expand(
                self.num_candidates,
                -1,
            ))
        )

        # Apply Phase 6 safety clamp.
        candidates = self.clamp_latents(candidates)

        return candidates

    @torch.no_grad()
    def generate_images(
        self,
        candidate_latents,
    ):
        """
        Decode candidate latent vectors using the existing
        Phase 6 DiffAE candidate generator.
        """

        if isinstance(candidate_latents, np.ndarray):
            candidate_latents = torch.from_numpy(
                candidate_latents
            )

        candidate_latents = candidate_latents.float()

        if candidate_latents.ndim != 2:
            raise ValueError(
                "candidate_latents must be a 2D tensor."
            )

        if candidate_latents.shape[1] != 512:
            raise ValueError(
                "candidate_latents must have shape (N, 512)."
            )

        if candidate_latents.shape[0] != self.num_candidates:
            raise ValueError(
                f"Expected {self.num_candidates} candidates, "
                f"got {candidate_latents.shape[0]}."
            )

        return self.candidate_generator.generate(
            candidate_latents
        )

    @torch.no_grad()
    def generate_round(
        self,
        current_latent,
        round_number,
    ):
        """
        Generate both candidate latents and their decoded images.

        Returns:
            candidate_latents
            candidate_images
        """

        candidate_latents = self.generate_candidate_latents(
            current_latent=current_latent,
            round_number=round_number,
        )

        candidate_images = self.generate_images(
            candidate_latents
        )

        return (
            candidate_latents,
            candidate_images,
        )
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from phase3_selection_aggregator import aggregate_selected_latents
from phase4_projector import DirectionProjector
from phase5_sampler import ExplorationSampler

from .session_data import SimulatedSession
from .simulated_witness import (
    compute_face_encoding,
    select_best_candidate_with_confidence,
)


class SessionSimulator:
    """
    Runs one simulated RecallFace reconstruction session.

    Pipeline:

        aggregation
            ↓
        projector
            ↓
        sampler
            ↓
        candidate latents
            ↓
        candidate image generation
            ↓
        simulated witness
            ↓
        selected candidate
    """

    def __init__(
        self,
        projector: DirectionProjector,
        sampler: ExplorationSampler,
        device: str = "cpu",
        num_candidates: int = 12,
    ):
        if not 12 <= num_candidates <= 20:
            raise ValueError(
                "num_candidates must be between 12 and 20."
            )

        self.projector = projector.to(device)
        self.sampler = sampler.to(device)

        self.projector.eval()
        self.sampler.eval()

        self.device = device
        self.num_candidates = num_candidates

    def generate_candidate_latents(
        self,
        current_latent: np.ndarray,
        round_number: int,
    ) -> np.ndarray:
        """
        Generate candidate latent vectors for one round.

        Returns:
            Array with shape:
            (num_candidates, 512)
        """

        current_tensor = torch.tensor(
            current_latent,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        round_tensor = torch.tensor(
            [[float(round_number)]],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            # Phase 4: project the current latent
            direction = self.projector(
                current_tensor
            )

            # Phase 5: estimate exploration variance
            variance = self.sampler(
                current_tensor,
                round_tensor,
            )

            # Controlled stochastic exploration around
            # the projected direction.
            noise = torch.randn(
                self.num_candidates,
                512,
                device=self.device,
            )

            candidate_latents = (
                direction
                + noise * torch.sqrt(
                    variance
                )
            )

        return candidate_latents.cpu().numpy().astype(
            np.float32
        )

    def run_round(
        self,
        session: SimulatedSession,
        round_number: int,
        current_latent: np.ndarray,
        candidate_image_generator: Callable[
            [np.ndarray], list[np.ndarray]
        ],
        target_encoding: np.ndarray,
    ) -> tuple[np.ndarray, int, float, list[float]]:
        """
        Run one complete simulated round.

        candidate_image_generator receives candidate latents
        and must return candidate face images.
        """

        candidate_latents = self.generate_candidate_latents(
            current_latent=current_latent,
            round_number=round_number,
        )

        candidate_images = candidate_image_generator(
            candidate_latents
        )

        candidate_encodings = [
            compute_face_encoding(image)
            for image in candidate_images
        ]

        (
            selected_index,
            similarity_scores,
            confidence,
        ) = select_best_candidate_with_confidence(
            target_encoding,
            candidate_encodings,
        )

        selected_latent = candidate_latents[
            selected_index
        ]

        session.add_round(
            round_number=round_number,
            candidate_latents=candidate_latents.tolist(),
            similarity_scores=similarity_scores,
            selected_index=selected_index,
            confidence=confidence,
        )

        return (
            selected_latent,
            selected_index,
            confidence,
            similarity_scores,
        )
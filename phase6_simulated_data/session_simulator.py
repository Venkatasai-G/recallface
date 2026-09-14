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

    Phase 6 flow:

        Round 1
            ↓
        Phase 2 starter set
            ↓
        simulated witness
            ↓
        selected candidate
            ↓
        Phase 3 aggregator
            ↓
        Phase 4 projector
            ↓
        Phase 5 sampler
            ↓
        next-round candidates
            ↓
        simulated witness
            ↓
        repeat
    """

    def __init__(
        self,
        projector: DirectionProjector,
        sampler: ExplorationSampler,
        starter_latents: np.ndarray,
        device: str = "cpu",
        num_candidates: int = 12,
    ):
        if not 12 <= num_candidates <= 20:
            raise ValueError(
                "num_candidates must be between 12 and 20."
            )

        starter_latents = np.asarray(
            starter_latents,
            dtype=np.float32,
        )

        if starter_latents.ndim != 2:
            raise ValueError(
                "starter_latents must have shape (N, 512)"
            )

        if starter_latents.shape != (num_candidates, 512):
            raise ValueError(
                f"starter_latents must have shape "
                f"({num_candidates}, 512), "
                f"got {starter_latents.shape}"
            )

        self.projector = projector.to(device)
        self.sampler = sampler.to(device)

        self.projector.eval()
        self.sampler.eval()

        self.starter_latents = starter_latents
        self.device = device
        self.num_candidates = num_candidates

    def generate_candidate_latents(
        self,
        current_latent: np.ndarray,
        round_number: int,
    ) -> np.ndarray:
        """
        Generate candidate latent vectors for rounds after Round 1.

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

            # Phase 4: project the current aggregated latent.
            direction = self.projector(
                current_tensor
            )

            # Phase 5: estimate exploration variance.
            variance = self.sampler(
                current_tensor,
                round_tensor,
            )

            # Controlled stochastic exploration.
            noise = torch.randn(
                self.num_candidates,
                512,
                device=self.device,
            )

            candidate_latents = (
                direction
                + noise * torch.sqrt(variance)
            )

        return candidate_latents.cpu().numpy().astype(
            np.float32
        )

    def get_candidate_latents(
        self,
        round_number: int,
        current_latent: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Get candidate latents for a given round.

        Round 1:
            Use the fixed Phase 2 starter set.

        Round 2+:
            Use Phase 4 projector + Phase 5 sampler.
        """

        if round_number == 1:
            return self.starter_latents.copy()

        if current_latent is None:
            raise ValueError(
                "current_latent is required for rounds after Round 1."
            )

        return self.generate_candidate_latents(
            current_latent=current_latent,
            round_number=round_number,
        )

    def aggregate_selected_latent(
        self,
        selected_latent: np.ndarray,
        confidence: float,
    ) -> np.ndarray:
        """
        Apply Phase 3 aggregation to the selected candidate.

        A single selected candidate is represented as a
        one-row latent tensor with its witness confidence
        as the aggregation weight.
        """

        selected_tensor = torch.tensor(
            selected_latent,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        weights = torch.tensor(
            [confidence],
            dtype=torch.float32,
            device=self.device,
        )

        aggregated_latent = aggregate_selected_latents(
            selected_latents=selected_tensor,
            weights=weights,
        )

        return aggregated_latent.detach().cpu().numpy().astype(
            np.float32
        )

    def run_round(
        self,
        session: SimulatedSession,
        round_number: int,
        current_latent: np.ndarray | None,
        candidate_image_generator: Callable[
            [np.ndarray], list[np.ndarray]
        ],
        target_encoding: np.ndarray,
    ) -> tuple[np.ndarray, int, float, list[float]]:
        """
        Run one complete simulated round.

        Round 1:
            Phase 2 starter set → witness.

        Round 2+:
            Phase 3 aggregation → Phase 4 projector
            → Phase 5 sampler → witness.
        """

        candidate_latents = self.get_candidate_latents(
            round_number=round_number,
            current_latent=current_latent,
        )

        candidate_images = candidate_image_generator(
            candidate_latents
        )

        candidate_encodings: list[np.ndarray | None] = []

        for image in candidate_images:
            try:
                encoding = compute_face_encoding(image)
            except ValueError:
                encoding = None

            candidate_encodings.append(encoding)

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

        # Phase 3: aggregate the selected candidate
        # using the simulated witness confidence.
        next_latent = self.aggregate_selected_latent(
            selected_latent=selected_latent,
            confidence=confidence,
        )

        session.add_round(
            round_number=round_number,
            candidate_latents=candidate_latents.tolist(),
            similarity_scores=similarity_scores,
            selected_index=selected_index,
            confidence=confidence,
        )

        return (
            next_latent,
            selected_index,
            confidence,
            similarity_scores,
        )
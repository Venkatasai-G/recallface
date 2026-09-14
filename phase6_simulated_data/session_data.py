from dataclasses import dataclass, field
from typing import List


@dataclass
class RoundRecord:
    """
    Stores all information from one simulated reconstruction round.
    """

    round_number: int

    # Candidate latent codes generated during this round.
    # Shape conceptually: (num_candidates, 512)
    candidate_latents: List[List[float]]

    # Candidate similarity scores against the target.
    # One score per candidate.
    similarity_scores: List[float]

    # Index of the candidate selected by the simulated witness.
    selected_index: int

    # Optional confidence assigned to the selected candidate.
    confidence: float | None = None


@dataclass
class SimulatedSession:
    """
    Stores a complete simulated RecallFace session.
    """

    session_id: int

    # Target latent representing the face being reconstructed.
    target_latent: List[float]

    # Complete round-by-round history.
    rounds: List[RoundRecord] = field(default_factory=list)

    def add_round(
        self,
        round_number: int,
        candidate_latents: List[List[float]],
        similarity_scores: List[float],
        selected_index: int,
        confidence: float | None = None,
    ) -> None:
        """
        Add one completed reconstruction round.
        """

        if len(candidate_latents) != len(similarity_scores):
            raise ValueError(
                "Number of candidate latents must match "
                "number of similarity scores"
            )

        if len(candidate_latents) == 0:
            raise ValueError(
                "At least one candidate is required"
            )

        if not 0 <= selected_index < len(candidate_latents):
            raise ValueError(
                "selected_index is outside the candidate range"
            )

        self.rounds.append(
            RoundRecord(
                round_number=round_number,
                candidate_latents=candidate_latents,
                similarity_scores=similarity_scores,
                selected_index=selected_index,
                confidence=confidence,
            )
        )
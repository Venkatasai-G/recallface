from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from config import (
    DEFAULT_DEVICE,
    LATENT_DIM,
    MAX_CANDIDATES,
    MAX_ROUNDS,
    RANDOM_SEED,
    SIMULATED_SESSIONS_DIR,
    STARTER_SET_SIZE,
    TARGET_SESSIONS_MAX,
)
from phase4_projector import DirectionProjector
from phase5_sampler import ExplorationSampler

from .diffae_candidate_generator import DiffAECandidateGenerator
from .session_data import SimulatedSession
from .session_simulator import SessionSimulator
from .simulated_witness import compute_face_encoding
from .target_generator import sample_target_latent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "diffae"
    / "checkpoints"
    / "last.ckpt"
)

LATENT_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "diffae"
    / "checkpoints"
    / "latent.pkl"
)

STARTER_LATENTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "starter_set"
    / "starter_latents.pt"
)


def set_session_seed(seed: int) -> None:
    """
    Make each session independently reproducible.

    Using a session-specific seed also makes resume behavior
    independent of which previous sessions were completed.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_session(
    session: SimulatedSession,
    output_path: Path,
) -> None:
    """
    Save one complete simulated session in compact PyTorch format.

    Saving one session per file provides simple resume support:
    completed session files can be skipped after a Colab restart.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rounds = []

    for record in session.rounds:
        rounds.append(
            {
                "round_number": int(record.round_number),
                "candidate_latents": torch.tensor(
                    record.candidate_latents,
                    dtype=torch.float32,
                ),
                "similarity_scores": torch.tensor(
                    record.similarity_scores,
                    dtype=torch.float32,
                ),
                "selected_index": int(record.selected_index),
                "confidence": (
                    None
                    if record.confidence is None
                    else float(record.confidence)
                ),
            }
        )

    data = {
        "session_id": int(session.session_id),
        "target_latent": torch.tensor(
            session.target_latent,
            dtype=torch.float32,
        ),
        "rounds": rounds,
    }

    temporary_path = output_path.with_suffix(".tmp")

    torch.save(
        data,
        temporary_path,
    )

    temporary_path.replace(output_path)


def build_candidate_generator(
    device: str,
) -> DiffAECandidateGenerator:
    """Create and load the pretrained DiffAE candidate generator."""

    generator = DiffAECandidateGenerator(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
    )

    generator.load()

    return generator


def build_simulator(
    device: str,
    starter_latents: np.ndarray,
    num_candidates: int,
) -> SessionSimulator:
    """Create the untrained Phase 4 + Phase 5 navigation stack."""

    projector = DirectionProjector().to(device)
    sampler = ExplorationSampler().to(device)

    projector.eval()
    sampler.eval()

    return SessionSimulator(
        projector=projector,
        sampler=sampler,
        starter_latents=starter_latents,
        device=device,
        num_candidates=num_candidates,
    )


def generate_target_encoding(
    target_latent: torch.Tensor,
    candidate_generator: DiffAECandidateGenerator,
) -> np.ndarray:
    """
    Generate the synthetic target face and compute its
    128-dimensional face-recognition encoding.
    """

    target_array = (
        target_latent
        .unsqueeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    images = candidate_generator.generate(
        target_array
    )

    return compute_face_encoding(images[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase 6 simulated RecallFace sessions."
    )

    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="Total session IDs to generate.",
    )

    parser.add_argument(
        "--start-session",
        type=int,
        default=1,
        help="First session ID to consider.",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of rounds per session.",
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=STARTER_SET_SIZE,
        help="Candidates per round.",
    )

    args = parser.parse_args()

    if args.sessions < 1:
        raise ValueError("--sessions must be at least 1.")

    if not 1 <= args.start_session:
        raise ValueError(
            "--start-session must be at least 1."
        )

    if args.rounds < 1 or args.rounds > MAX_ROUNDS:
        raise ValueError(
            f"--rounds must be between 1 and {MAX_ROUNDS}."
        )

    if not 12 <= args.candidates <= MAX_CANDIDATES:
        raise ValueError(
            f"--candidates must be between 12 and {MAX_CANDIDATES}."
        )

    final_session_id = (
        args.start_session + args.sessions - 1
    )

    if final_session_id > TARGET_SESSIONS_MAX:
        raise ValueError(
            f"Requested session ID {final_session_id} exceeds "
            f"configured maximum {TARGET_SESSIONS_MAX}."
        )

    if args.candidates != STARTER_SET_SIZE:
        raise ValueError(
            "For the current Phase 6 implementation, "
            "the number of candidates must equal the "
            "12-image Phase 2 starter set."
        )

    device = (
        DEFAULT_DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = (
        SIMULATED_SESSIONS_DIR
        / f"sessions_{args.candidates}c_{args.rounds}r"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("============================================")
    print("RecallFace Phase 6 Production Generator")
    print("============================================")
    print("Device:", device)
    print("Session range:", args.start_session, "→", final_session_id)
    print("Rounds:", args.rounds)
    print("Candidates:", args.candidates)
    print("Output:", output_dir)

    # --------------------------------------------------------
    # Load latent distribution
    # --------------------------------------------------------

    print()
    print("Loading learned latent distribution...")

    latent_data = torch.load(
        LATENT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    conds = latent_data["conds"].float()

    if conds.ndim != 2 or conds.shape[1] != LATENT_DIM:
        raise ValueError(
            f"Expected latent pool shape (N, {LATENT_DIM}), "
            f"got {tuple(conds.shape)}"
        )

    print("Latent pool:", tuple(conds.shape))

    # --------------------------------------------------------
    # Load Phase 2 starter latents
    # --------------------------------------------------------

    print()
    print("Loading Phase 2 starter set...")

    starter_latents = torch.load(
        STARTER_LATENTS_PATH,
        map_location="cpu",
        weights_only=True,
    ).numpy().astype(np.float32)

    if starter_latents.shape != (
        STARTER_SET_SIZE,
        LATENT_DIM,
    ):
        raise ValueError(
            "Invalid starter latent shape: "
            f"{starter_latents.shape}"
        )

    print(
        "Starter latents:",
        starter_latents.shape,
    )

    # --------------------------------------------------------
    # Load DiffAE once
    # --------------------------------------------------------

    print()
    print("Loading DiffAE...")

    candidate_generator = build_candidate_generator(
        device=device,
    )

    # --------------------------------------------------------
    # Build projector + sampler once
    # --------------------------------------------------------

    set_session_seed(RANDOM_SEED)
    
    simulator = build_simulator(
        device=device,
        starter_latents=starter_latents,
        num_candidates=args.candidates,
    )

    # --------------------------------------------------------
    # Generate sessions
    # --------------------------------------------------------

    completed = 0
    skipped = 0
    failed = 0

    for session_id in range(
        args.start_session,
        final_session_id + 1,
    ):

        output_path = (
            output_dir
            / f"session_{session_id:06d}.pt"
        )

        # Resume support.
        if output_path.exists():
            print(
                f"[{session_id}] already exists; skipping."
            )
            skipped += 1
            continue

        print()
        print("--------------------------------------------")
        print(
            f"Session {session_id} "
            f"({session_id - args.start_session + 1}/"
            f"{args.sessions})"
        )
        print("--------------------------------------------")

        try:
            session_seed = (
                RANDOM_SEED + session_id
            )

            set_session_seed(session_seed)

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

            target_latent = sample_target_latent(
                conds,
                seed=session_seed,
            )

            target_encoding = generate_target_encoding(
                target_latent=target_latent,
                candidate_generator=candidate_generator,
            )

            session = SimulatedSession(
                session_id=session_id,
                target_latent=(
                    target_latent
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .tolist()
                ),
            )

            current_latent = None

            # ------------------------------------------------
            # Rounds
            # ------------------------------------------------

            for round_number in range(
                1,
                args.rounds + 1,
            ):

                (
                    current_latent,
                    selected_index,
                    confidence,
                    similarity_scores,
                ) = simulator.run_round(
                    session=session,
                    round_number=round_number,
                    current_latent=current_latent,
                    candidate_image_generator=(
                        candidate_generator.generate
                    ),
                    target_encoding=target_encoding,
                )

                valid_count = sum(
                    score > 0.0
                    for score in similarity_scores
                )

                print(
                    f"  Round {round_number:02d}: "
                    f"selected={selected_index:02d} "
                    f"confidence={confidence:.6f} "
                    f"valid={valid_count}/"
                    f"{args.candidates}"
                )

            # ------------------------------------------------
            # Save complete session atomically
            # ------------------------------------------------

            save_session(
                session=session,
                output_path=output_path,
            )

            completed += 1

            print(
                f"  Saved: {output_path.name}"
            )

        except Exception as exc:
            failed += 1

            print(
                f"  ERROR in session {session_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("============================================")
    print("Phase 6 Generation Complete")
    print("============================================")
    print("Completed:", completed)
    print("Skipped:", skipped)
    print("Failed:", failed)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from phase4_projector import DirectionProjector
from phase5_sampler import ExplorationSampler

from phase6_simulated_data.diffae_candidate_generator import (
    DiffAECandidateGenerator,
)
from phase6_simulated_data.session_data import SimulatedSession
from phase6_simulated_data.session_simulator import SessionSimulator
from phase6_simulated_data.simulated_witness import (
    compute_face_encoding,
)
from phase7_training.train_navigation import (
    build_session_split,
)


# ============================================================
# Configuration
# ============================================================

SEED = 2026

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATASET_DIR = Path(
    "/kaggle/working/recallface/data/simulated_sessions/pilot_503"
)

CHECKPOINT_DIR = Path(
    "/kaggle/working/recallface/checkpoints/FINAL_TRAINING"
)

OUTPUT_DIR = Path(
    "/kaggle/working/recallface/evaluation"
)

NUM_TARGETS = 10
NUM_ROUNDS = 20
NUM_CANDIDATES = 12

DIFFAE_CHECKPOINT = Path(
    "/kaggle/working/recallface/"
    "pretrained/diffae/checkpoints/last.ckpt"
)

LATENT_FILE = Path(
    "/kaggle/working/recallface/"
    "pretrained/diffae/checkpoints/latent.pkl"
)

STARTER_LATENTS_FILE = Path(
    "/kaggle/working/recallface/"
    "data/starter_set/starter_latents.pt"
)

STARTER_XT_FILE = Path(
    "/kaggle/working/recallface/"
    "data/starter_set/starter_xT.pt"
)

PROJECTOR_CHECKPOINT = (
    CHECKPOINT_DIR / "projector_best.pt"
)

SAMPLER_CHECKPOINT = (
    CHECKPOINT_DIR / "sampler_best.pt"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Validation checks
# ============================================================

def check_required_files() -> None:
    required = [
        DATASET_DIR,
        DIFFAE_CHECKPOINT,
        LATENT_FILE,
        STARTER_LATENTS_FILE,
        STARTER_XT_FILE,
        PROJECTOR_CHECKPOINT,
        SAMPLER_CHECKPOINT,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path does not exist:\n{path}"
            )


# ============================================================
# Load latent statistics
# ============================================================

def load_latent_statistics() -> tuple[np.ndarray, np.ndarray]:
    latent_data = torch.load(
        LATENT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(latent_data, dict):
        raise ValueError(
            "Expected latent.pkl to contain a dictionary."
        )

    latent_mean = np.asarray(
        latent_data["conds_mean"],
        dtype=np.float32,
    )

    latent_std = np.asarray(
        latent_data["conds_std"],
        dtype=np.float32,
    )

    if latent_mean.shape != (512,):
        raise ValueError(
            "Expected conds_mean shape (512,), "
            f"got {latent_mean.shape}"
        )

    if latent_std.shape != (512,):
        raise ValueError(
            "Expected conds_std shape (512,), "
            f"got {latent_std.shape}"
        )

    if np.any(latent_std <= 0):
        raise ValueError(
            "latent_std must contain positive values."
        )

    return latent_mean, latent_std


# ============================================================
# Load models
# ============================================================

def load_navigation_models(
    learned: bool,
) -> tuple[
    DirectionProjector,
    ExplorationSampler,
]:
    projector = DirectionProjector().to(DEVICE)
    sampler = ExplorationSampler().to(DEVICE)

    if learned:
        projector_data = torch.load(
            PROJECTOR_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False,
        )

        sampler_data = torch.load(
            SAMPLER_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False,
        )

        projector.load_state_dict(
            projector_data["model_state_dict"]
        )

        sampler.load_state_dict(
            sampler_data["model_state_dict"]
        )

        print(
            "Loaded learned Projector checkpoint "
            f"(epoch {projector_data['epoch']})."
        )

        print(
            "Loaded learned Sampler checkpoint "
            f"(epoch {sampler_data['epoch']})."
        )

    else:
        print(
            "Using clean bootstrap "
            "Projector + Sampler."
        )

    projector.eval()
    sampler.eval()

    return projector, sampler


# ============================================================
# Candidate generator
# ============================================================

def build_candidate_generator():
    generator = DiffAECandidateGenerator(
        checkpoint_path=str(DIFFAE_CHECKPOINT),
        xT_path=str(STARTER_XT_FILE),
        device=str(DEVICE),
    )

    generator.load()

    return generator


# ============================================================
# Single system run
# ============================================================

def run_single_system(
    target_latent: np.ndarray,
    target_encoding: np.ndarray,
    starter_latents: np.ndarray,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    candidate_generator,
    learned: bool,
    seed: int,
    session_id: int,
) -> dict:

    # Reset RNG before every matched run.
    # Therefore baseline and learned receive the
    # same stochastic exploration sequence.
    set_seed(seed)

    projector, sampler = load_navigation_models(
        learned=learned
    )

    simulator = SessionSimulator(
        projector=projector,
        sampler=sampler,
        starter_latents=starter_latents,
        latent_mean=latent_mean,
        latent_std=latent_std,
        device=str(DEVICE),
        num_candidates=NUM_CANDIDATES,
        latent_clip=3.0,
    )

    session = SimulatedSession(
        session_id=session_id,
        target_latent=target_latent.tolist(),
    )

    current_latent = None

    round_best_similarity = []
    round_validity = []
    round_confidence = []

    for round_number in range(
        1,
        NUM_ROUNDS + 1,
    ):

        current_latent_input = (
            None
            if round_number == 1
            else current_latent
        )

        (
            next_latent,
            selected_index,
            confidence,
            similarity_scores,
        ) = simulator.run_round(
            session=session,
            round_number=round_number,
            current_latent=current_latent_input,
            candidate_image_generator=candidate_generator.generate,
            target_encoding=target_encoding,
        )

        scores = np.asarray(
            similarity_scores,
            dtype=np.float32,
        )

        valid_mask = scores > 0

        valid_count = int(
            valid_mask.sum()
        )

        if valid_count == 0:
            raise RuntimeError(
                f"No valid candidates in "
                f"session {session_id}, "
                f"round {round_number}."
            )

        round_best_similarity.append(
            float(scores[valid_mask].max())
        )

        round_validity.append(
            float(
                valid_count / len(scores)
            )
        )

        round_confidence.append(
            float(confidence)
        )

        current_latent = next_latent

    return {
        "session_id": session_id,
        "learned": learned,
        "round_best_similarity": (
            round_best_similarity
        ),
        "round_validity": round_validity,
        "round_confidence": round_confidence,
    }


# ============================================================
# Main evaluation
# ============================================================

def main() -> None:

    set_seed(SEED)

    print(
        "========== PHASE 7 END-TO-END EVALUATION =========="
    )

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    check_required_files()

    print("\nLoading latent statistics...")

    latent_mean, latent_std = (
        load_latent_statistics()
    )

    starter_latents = torch.load(
        STARTER_LATENTS_FILE,
        map_location="cpu",
        weights_only=False,
    )

    starter_latents = (
        starter_latents.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if starter_latents.shape != (
        NUM_CANDIDATES,
        512,
    ):
        raise ValueError(
            "Unexpected starter_latents shape: "
            f"{starter_latents.shape}"
        )

    # --------------------------------------------------------
    # Reproduce the EXACT Phase 7 validation split
    # --------------------------------------------------------

    print(
        "\nRecreating exact Phase 7 "
        "train/validation split..."
    )

    train_files, val_files = (
        build_session_split()
    )

    print(
        "Training sessions:",
        len(train_files)
    )

    print(
        "Validation sessions:",
        len(val_files)
    )

    if len(train_files) != 402:
        raise RuntimeError(
            f"Expected 402 training sessions, "
            f"got {len(train_files)}"
        )

    if len(val_files) != 101:
        raise RuntimeError(
            f"Expected 101 validation sessions, "
            f"got {len(val_files)}"
        )

    # Same deterministic ordering used after
    # the shuffled split.
    evaluation_files = val_files[
        :NUM_TARGETS
    ]

    print("\nEvaluation targets:")

    for path in evaluation_files:
        print(" ", path.name)

    # --------------------------------------------------------
    # Load DiffAE
    # --------------------------------------------------------

    print(
        "\nLoading DiffAE candidate generator..."
    )

    candidate_generator = (
        build_candidate_generator()
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    for target_number, session_path in enumerate(
        evaluation_files,
        start=1,
    ):

        session_data = torch.load(
            session_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(session_data, dict):
            raise ValueError(
                f"{session_path.name}: expected a dictionary, "
                f"got {type(session_data)}"
            )

        if "target_latent" not in session_data:
            raise KeyError(
                f"{session_path.name}: missing 'target_latent'"
            )

        target_latent = np.asarray(
            session_data["target_latent"],
            dtype=np.float32,
        )

        if target_latent.shape != (512,):
            raise ValueError(
                f"{session_path.name}: "
                f"target latent has shape "
                f"{target_latent.shape}"
            )

        # Render the target face using the SAME
        # DiffAE + fixed xT path.
        target_image = (
            candidate_generator.generate(
                target_latent[None, :]
            )[0]
        )

        target_encoding = compute_face_encoding(
            target_image
        )

        session_id = int(
            session_path.stem.split("_")[1]
        )

        run_seed = SEED + session_id

        print(
            f"\n[{target_number}/{NUM_TARGETS}] "
            f"Session {session_id}"
        )

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_result = run_single_system(
            target_latent=target_latent,
            target_encoding=target_encoding,
            starter_latents=starter_latents,
            latent_mean=latent_mean,
            latent_std=latent_std,
            candidate_generator=candidate_generator,
            learned=False,
            seed=run_seed,
            session_id=session_id,
        )

        # ----------------------------------------------------
        # Learned
        # ----------------------------------------------------

        learned_result = run_single_system(
            target_latent=target_latent,
            target_encoding=target_encoding,
            starter_latents=starter_latents,
            latent_mean=latent_mean,
            latent_std=latent_std,
            candidate_generator=candidate_generator,
            learned=True,
            seed=run_seed,
            session_id=session_id,
        )

        results.append(
            {
                "session_id": session_id,
                "baseline": baseline_result,
                "learned": learned_result,
            }
        )

        baseline_r1 = (
            baseline_result[
                "round_best_similarity"
            ][0]
        )

        baseline_r20 = (
            baseline_result[
                "round_best_similarity"
            ][-1]
        )

        learned_r1 = (
            learned_result[
                "round_best_similarity"
            ][0]
        )

        learned_r20 = (
            learned_result[
                "round_best_similarity"
            ][-1]
        )

        print(
            f"  Baseline: R1={baseline_r1:.4f}, "
            f"R20={baseline_r20:.4f}, "
            f"Δ={baseline_r20 - baseline_r1:+.4f}"
        )

        print(
            f"  Learned : R1={learned_r1:.4f}, "
            f"R20={learned_r20:.4f}, "
            f"Δ={learned_r20 - learned_r1:+.4f}"
        )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    def collect(
        system: str,
        metric: str,
    ):
        return np.asarray(
            [
                r[system][metric]
                for r in results
            ],
            dtype=np.float32,
        )

    baseline_similarity = collect(
        "baseline",
        "round_best_similarity",
    )

    learned_similarity = collect(
        "learned",
        "round_best_similarity",
    )

    baseline_validity = collect(
        "baseline",
        "round_validity",
    )

    learned_validity = collect(
        "learned",
        "round_validity",
    )

    baseline_confidence = collect(
        "baseline",
        "round_confidence",
    )

    learned_confidence = collect(
        "learned",
        "round_confidence",
    )

    baseline_r1 = baseline_similarity[:, 0]
    baseline_r20 = baseline_similarity[:, -1]

    learned_r1 = learned_similarity[:, 0]
    learned_r20 = learned_similarity[:, -1]

    baseline_improvement = (
        baseline_r20 - baseline_r1
    )

    learned_improvement = (
        learned_r20 - learned_r1
    )

    summary = {
        "num_targets": NUM_TARGETS,
        "num_rounds": NUM_ROUNDS,
        "num_candidates": NUM_CANDIDATES,
        "seed": SEED,

        "baseline_mean_r1": float(
            baseline_r1.mean()
        ),

        "baseline_mean_r20": float(
            baseline_r20.mean()
        ),

        "baseline_mean_improvement": float(
            baseline_improvement.mean()
        ),

        "learned_mean_r1": float(
            learned_r1.mean()
        ),

        "learned_mean_r20": float(
            learned_r20.mean()
        ),

        "learned_mean_improvement": float(
            learned_improvement.mean()
        ),

        "baseline_candidate_validity": float(
            baseline_validity.mean()
        ),

        "learned_candidate_validity": float(
            learned_validity.mean()
        ),

        "baseline_mean_confidence_r1": float(
            baseline_confidence[:, 0].mean()
        ),

        "baseline_mean_confidence_r20": float(
            baseline_confidence[:, -1].mean()
        ),

        "learned_mean_confidence_r1": float(
            learned_confidence[:, 0].mean()
        ),

        "learned_mean_confidence_r20": float(
            learned_confidence[:, -1].mean()
        ),
    }

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print(
        "\n========== EVALUATION SUMMARY =========="
    )

    print(
        "\nBaseline:"
    )

    print(
        f"Mean R1 similarity  : "
        f"{summary['baseline_mean_r1']:.6f}"
    )

    print(
        f"Mean R20 similarity : "
        f"{summary['baseline_mean_r20']:.6f}"
    )

    print(
        f"Mean improvement    : "
        f"{summary['baseline_mean_improvement']:+.6f}"
    )

    print(
        f"Candidate validity  : "
        f"{summary['baseline_candidate_validity']:.6f}"
    )

    print(
        "\nLearned:"
    )

    print(
        f"Mean R1 similarity  : "
        f"{summary['learned_mean_r1']:.6f}"
    )

    print(
        f"Mean R20 similarity : "
        f"{summary['learned_mean_r20']:.6f}"
    )

    print(
        f"Mean improvement    : "
        f"{summary['learned_mean_improvement']:+.6f}"
    )

    print(
        f"Candidate validity  : "
        f"{summary['learned_candidate_validity']:.6f}"
    )

    print(
        "\nConfidence:"
    )

    print(
        f"Baseline R1  : "
        f"{summary['baseline_mean_confidence_r1']:.6f}"
    )

    print(
        f"Baseline R20 : "
        f"{summary['baseline_mean_confidence_r20']:.6f}"
    )

    print(
        f"Learned R1   : "
        f"{summary['learned_mean_confidence_r1']:.6f}"
    )

    print(
        f"Learned R20  : "
        f"{summary['learned_mean_confidence_r20']:.6f}"
    )

    # --------------------------------------------------------
    # Save evaluation artifact
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "phase7_learned_vs_baseline_10_targets.pt"
    )

    torch.save(
        {
            "summary": summary,
            "results": results,
        },
        output_path,
    )

    print(
        "\nSaved evaluation results:"
    )

    print(output_path)

    print(
        "\n========== EVALUATION COMPLETE =========="
    )


if __name__ == "__main__":
    main()
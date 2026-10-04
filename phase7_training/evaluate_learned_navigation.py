# evaluate_learned_navigation.py


from __future__ import annotations



import random

import sys

from pathlib import Path



import numpy as np

import torch





# ============================================================

# Project root

# ============================================================



ROOT_DIR = Path(__file__).resolve().parent.parent



if str(ROOT_DIR) not in sys.path:

    sys.path.insert(0, str(ROOT_DIR))





from phase4_projector import DirectionProjector

from phase5_sampler import ExplorationSampler



from phase6_simulated_data.diffae_candidate_generator import (

    DiffAECandidateGenerator,

)



from phase6_simulated_data.session_data import (

    SimulatedSession,

)



from phase6_simulated_data.session_simulator import (

    SessionSimulator,

)



from phase6_simulated_data.simulated_witness import (

    compute_face_encoding,

)





# ============================================================

# Configuration

# ============================================================



SEED = 2026



DEVICE = torch.device(

    "cuda" if torch.cuda.is_available() else "cpu"

)





# ------------------------------------------------------------

# Data

# ------------------------------------------------------------



DATASET_DIR = Path(

    "/kaggle/working/recallface/data/simulated_sessions/pilot_503"

)





# ------------------------------------------------------------

# Checkpoints

# ------------------------------------------------------------



ORIGINAL_CHECKPOINT_DIR = Path(

    "/kaggle/working/recallface/checkpoints/FINAL_TRAINING"

)



RANKING_CHECKPOINT_DIR = Path(

    "/kaggle/working/recallface/checkpoints/PHASE7_RANKING_OBJECTIVE"

)





ORIGINAL_PROJECTOR_CHECKPOINT = (

    ORIGINAL_CHECKPOINT_DIR

    / "projector_best.pt"

)



ORIGINAL_SAMPLER_CHECKPOINT = (

    ORIGINAL_CHECKPOINT_DIR

    / "sampler_best.pt"

)





RANKING_PROJECTOR_CHECKPOINT = (

    RANKING_CHECKPOINT_DIR

    / "projector_best.pt"

)



RANKING_SAMPLER_CHECKPOINT = (

    RANKING_CHECKPOINT_DIR

    / "sampler_best.pt"

)





# ------------------------------------------------------------

# Output

# ------------------------------------------------------------



OUTPUT_DIR = Path(

    "/kaggle/working/recallface/evaluation"

)





OUTPUT_PATH = (

    OUTPUT_DIR

    / "phase7_baseline_original_ranking_10_targets.pt"

)





# ------------------------------------------------------------

# Evaluation configuration

# ------------------------------------------------------------



NUM_TARGETS = 10

NUM_ROUNDS = 20

NUM_CANDIDATES = 12





# ------------------------------------------------------------

# DiffAE

# ------------------------------------------------------------



DIFFAE_CHECKPOINT = Path(

    "/kaggle/working/recallface/"

    "pretrained/diffae/checkpoints/last.ckpt"

)





LATENT_FILE = Path(

    "/kaggle/working/recallface/"

    "pretrained/diffae/checkpoints/latent.pkl"

)





# ------------------------------------------------------------

# Starter set

# ------------------------------------------------------------



STARTER_LATENTS_FILE = Path(

    "/kaggle/working/recallface/"

    "data/starter_set/starter_latents.pt"

)





STARTER_XT_FILE = Path(

    "/kaggle/working/recallface/"

    "data/starter_set/starter_xT.pt"

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

# Exact Phase 7 session split

# ============================================================



def build_session_split():



    files = sorted(

        DATASET_DIR.glob("session_*.pt")

    )



    if len(files) != 503:



        raise ValueError(

            f"Expected 503 session files, "

            f"found {len(files)}"

        )



    rng = random.Random(SEED)



    shuffled = files.copy()



    rng.shuffle(shuffled)



    split_index = int(

        len(shuffled) * 0.80

    )



    train_files = sorted(

        shuffled[:split_index],

        key=lambda p: p.name,

    )



    val_files = sorted(

        shuffled[split_index:],

        key=lambda p: p.name,

    )



    train_ids = {

        p.name for p in train_files

    }



    val_ids = {

        p.name for p in val_files

    }



    if train_ids & val_ids:



        raise RuntimeError(

            "Train/validation session overlap detected."

        )



    return train_files, val_files





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



        ORIGINAL_PROJECTOR_CHECKPOINT,



        ORIGINAL_SAMPLER_CHECKPOINT,



        RANKING_PROJECTOR_CHECKPOINT,



        RANKING_SAMPLER_CHECKPOINT,

    ]



    for path in required:



        if not path.exists():



            raise FileNotFoundError(

                f"Required path does not exist:\n{path}"

            )





# ============================================================

# Load latent statistics

# ============================================================



def load_latent_statistics():



    latent_data = torch.load(

        LATENT_FILE,

        map_location="cpu",

        weights_only=False,

    )



    if not isinstance(

        latent_data,

        dict,

    ):



        raise ValueError(

            "Expected latent.pkl to contain "

            "a dictionary."

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

            "latent_std must contain "

            "positive values."

        )



    return latent_mean, latent_std





# ============================================================

# Load a navigation system

# ============================================================



def load_navigation_system(

    system_name: str,

):



    projector = (

        DirectionProjector()

        .to(DEVICE)

    )



    sampler = (

        ExplorationSampler()

        .to(DEVICE)

    )





    # --------------------------------------------------------

    # BASELINE

    # --------------------------------------------------------



    if system_name == "baseline":



        print(

            "  Using clean bootstrap "

            "Projector + Sampler."

        )





    # --------------------------------------------------------

    # ORIGINAL

    # --------------------------------------------------------



    elif system_name == "original":



        projector_data = torch.load(

            ORIGINAL_PROJECTOR_CHECKPOINT,

            map_location=DEVICE,

            weights_only=False,

        )



        sampler_data = torch.load(

            ORIGINAL_SAMPLER_CHECKPOINT,

            map_location=DEVICE,

            weights_only=False,

        )



        projector.load_state_dict(

            projector_data[

                "model_state_dict"

            ]

        )



        sampler.load_state_dict(

            sampler_data[

                "model_state_dict"

            ]

        )



        print(

            "  Loaded ORIGINAL "

            f"Projector epoch "

            f"{projector_data['epoch']}."

        )



        print(

            "  Loaded ORIGINAL "

            f"Sampler epoch "

            f"{sampler_data['epoch']}."

        )





    # --------------------------------------------------------

    # RANKING-AWARE

    # --------------------------------------------------------



    elif system_name == "ranking":



        projector_data = torch.load(

            RANKING_PROJECTOR_CHECKPOINT,

            map_location=DEVICE,

            weights_only=False,

        )



        sampler_data = torch.load(

            RANKING_SAMPLER_CHECKPOINT,

            map_location=DEVICE,

            weights_only=False,

        )



        projector.load_state_dict(

            projector_data[

                "model_state_dict"

            ]

        )



        sampler.load_state_dict(

            sampler_data[

                "model_state_dict"

            ]

        )



        print(

            "  Loaded RANKING-AWARE "

            f"Projector epoch "

            f"{projector_data['epoch']}."

        )



        print(

            "  Loaded RANKING-AWARE "

            f"Sampler epoch "

            f"{sampler_data['epoch']}."

        )





    else:



        raise ValueError(

            f"Unknown system: {system_name}"

        )





    projector.eval()



    sampler.eval()



    return projector, sampler





# ============================================================

# Candidate generator

# ============================================================



def build_candidate_generator():



    generator = DiffAECandidateGenerator(



        checkpoint_path=str(

            DIFFAE_CHECKPOINT

        ),



        xT_path=str(

            STARTER_XT_FILE

        ),



        device=str(

            DEVICE

        ),

    )



    generator.load()



    return generator





# ============================================================

# Single system run

# ============================================================




def _capture_rng_state():
    """Capture all RNG states needed for exact checkpoint/resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    """Restore RNG state saved by _capture_rng_state()."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(obj, path: Path):
    """Write a checkpoint atomically so an interrupted write does not corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def run_single_system(
    system_name: str,
    target_latent: np.ndarray,
    target_encoding: np.ndarray,
    starter_latents: np.ndarray,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    candidate_generator,
    seed: int,
    session_id: int,
    checkpoint_path: Path,
) -> dict:
    """Run one navigation system with round-level checkpoint/resume."""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        expected = {
            "system": system_name,
            "session_id": session_id,
            "seed": seed,
            "num_rounds": NUM_ROUNDS,
            "num_candidates": NUM_CANDIDATES,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise RuntimeError(
                    f"Checkpoint mismatch in {checkpoint_path} "
                    f"for '{key}': expected {value!r}, "
                    f"found {checkpoint.get(key)!r}."
                )

        completed_rounds = int(checkpoint["completed_rounds"])
        if completed_rounds >= NUM_ROUNDS:
            print(
                f"    {system_name}: checkpoint already complete "
                f"({completed_rounds}/{NUM_ROUNDS} rounds)."
            )
            return checkpoint["result"]

        print(
            f"    {system_name}: resuming from round "
            f"{completed_rounds + 1}/{NUM_ROUNDS}."
        )

        session = checkpoint["session"]
        current_latent = checkpoint["current_latent"]
        round_best_similarity = list(checkpoint["round_best_similarity"])
        round_validity = list(checkpoint["round_validity"])
        round_confidence = list(checkpoint["round_confidence"])
        _restore_rng_state(checkpoint["rng_state"])

        projector, sampler = load_navigation_system(system_name)
    else:
        set_seed(seed)
        projector, sampler = load_navigation_system(system_name)
        session = SimulatedSession(
            session_id=session_id,
            target_latent=target_latent.tolist(),
        )
        current_latent = None
        round_best_similarity = []
        round_validity = []
        round_confidence = []

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

    start_round = len(round_best_similarity) + 1

    for round_number in range(start_round, NUM_ROUNDS + 1):
        current_latent_input = None if round_number == 1 else current_latent
        print(
            f"      {system_name} | round {round_number:02d}/{NUM_ROUNDS}",
            flush=True,
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

        scores = np.asarray(similarity_scores, dtype=np.float32)
        valid_mask = scores > 0
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            raise RuntimeError(
                f"No valid candidates in {system_name}, "
                f"session {session_id}, round {round_number}."
            )

        round_best_similarity.append(float(scores[valid_mask].max()))
        round_validity.append(float(valid_count / len(scores)))
        round_confidence.append(float(confidence))
        current_latent = next_latent

        partial_result = {
            "system": system_name,
            "session_id": session_id,
            "round_best_similarity": round_best_similarity,
            "round_validity": round_validity,
            "round_confidence": round_confidence,
        }

        _atomic_torch_save(
            {
                "checkpoint_version": 2,
                "system": system_name,
                "session_id": session_id,
                "seed": seed,
                "num_rounds": NUM_ROUNDS,
                "num_candidates": NUM_CANDIDATES,
                "completed_rounds": round_number,
                "session": session,
                "current_latent": current_latent,
                "round_best_similarity": round_best_similarity,
                "round_validity": round_validity,
                "round_confidence": round_confidence,
                "rng_state": _capture_rng_state(),
                "result": partial_result,
            },
            checkpoint_path,
        )
        print(
            f"        saved checkpoint: round {round_number:02d}",
            flush=True,
        )

    return {
        "system": system_name,
        "session_id": session_id,
        "round_best_similarity": round_best_similarity,
        "round_validity": round_validity,
        "round_confidence": round_confidence,
    }


# Main

# ============================================================



def main() -> None:



    set_seed(SEED)





    print(

        "=================================================="

    )



    print(

        "PHASE 7 THREE-WAY END-TO-END EVALUATION"

    )



    print(

        "=================================================="

    )





    print(

        "\nDevice:",

        DEVICE,

    )





    if DEVICE.type == "cuda":



        print(

            "GPU:",

            torch.cuda.get_device_name(0)

        )





    # --------------------------------------------------------

    # Required files

    # --------------------------------------------------------



    print(

        "\nChecking required files..."

    )



    check_required_files()



    print(

        "All required files found."

    )





    # --------------------------------------------------------

    # Load latent statistics

    # --------------------------------------------------------



    print(

        "\nLoading latent statistics..."

    )



    latent_mean, latent_std = (

        load_latent_statistics()

    )





    # --------------------------------------------------------

    # Starter latents

    # --------------------------------------------------------



    print(

        "Loading starter latents..."

    )



    starter_latents = torch.load(

        STARTER_LATENTS_FILE,

        map_location="cpu",

        weights_only=False,

    )





    starter_latents = (



        starter_latents



        .detach()



        .cpu()



        .numpy()



        .astype(

            np.float32

        )

    )





    if starter_latents.shape != (

        NUM_CANDIDATES,

        512,

    ):



        raise ValueError(



            "Unexpected starter_latents "

            "shape: "



            f"{starter_latents.shape}"

        )





    # --------------------------------------------------------

    # Exact Phase 7 validation split

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

        len(train_files),

    )





    print(

        "Validation sessions:",

        len(val_files),

    )





    if len(train_files) != 402:



        raise RuntimeError(

            "Expected 402 training sessions."

        )





    if len(val_files) != 101:



        raise RuntimeError(

            "Expected 101 validation sessions."

        )





    evaluation_files = (

        val_files[

            :NUM_TARGETS

        ]

    )





    print(

        "\nEvaluation targets:"

    )





    for path in evaluation_files:



        print(

            " ",

            path.name,

        )





    # --------------------------------------------------------

    # Candidate generator

    # --------------------------------------------------------



    print(

        "\nLoading DiffAE candidate generator..."

    )





    candidate_generator = (

        build_candidate_generator()

    )






    # --------------------------------------------------------
    # Resumable checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_ROOT = OUTPUT_DIR / "phase7_three_way_resumable"
    SESSION_CHECKPOINT_ROOT = CHECKPOINT_ROOT / "sessions"
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    SESSION_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    print(
        "\nResumable checkpoint directory:"
        f"\n{SESSION_CHECKPOINT_ROOT}"
    )
    print(
        "\nIMPORTANT: save/download this checkpoint directory "
        "before ending the Kaggle runtime if you want to resume "
        "in a future runtime."
    )

    results = []

    # ========================================================
    # Evaluate target sessions
    # ========================================================

    for target_number, session_path in enumerate(
        evaluation_files,
        start=1,
    ):
        session_id = int(session_path.stem.split("_")[1])
        session_dir = SESSION_CHECKPOINT_ROOT / f"session_{session_id:06d}"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_complete_path = session_dir / "session_complete.pt"
        session_meta_path = session_dir / "session_meta.pt"

        print("\n" + "=" * 60)
        print(f"[{target_number}/{NUM_TARGETS}] Session {session_id}")
        print("=" * 60)

        if session_complete_path.exists():
            completed_session = torch.load(
                session_complete_path,
                map_location="cpu",
                weights_only=False,
            )
            if completed_session.get("num_rounds") != NUM_ROUNDS:
                raise RuntimeError(
                    f"Completed checkpoint round-count mismatch for session {session_id}."
                )
            if completed_session.get("num_candidates") != NUM_CANDIDATES:
                raise RuntimeError(
                    f"Completed checkpoint candidate-count mismatch for session {session_id}."
                )
            completed_result = completed_session["result"]
            results.append(completed_result)
            print("  SESSION ALREADY COMPLETE — loaded checkpoint.")
            for system_name in ("baseline", "original", "ranking"):
                values = completed_result[system_name]["round_best_similarity"]
                print(
                    f"  {system_name:9s}: R1={values[0]:.4f}, "
                    f"R20={values[-1]:.4f}, "
                    f"Δ={values[-1] - values[0]:+.4f}"
                )
            continue

        session_data = torch.load(
            session_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(session_data, dict):
            raise ValueError(
                f"{session_path.name}: expected dictionary, got {type(session_data)}"
            )
        if "target_latent" not in session_data:
            raise KeyError(f"{session_path.name}: missing target_latent")

        target_latent = np.asarray(session_data["target_latent"], dtype=np.float32)
        if target_latent.shape != (512,):
            raise ValueError(
                f"{session_path.name}: target latent has shape {target_latent.shape}"
            )

        if session_meta_path.exists():
            meta = torch.load(
                session_meta_path,
                map_location="cpu",
                weights_only=False,
            )
            target_encoding = np.asarray(meta["target_encoding"], dtype=np.float32)
            saved_target_latent = np.asarray(meta["target_latent"], dtype=np.float32)
            if not np.array_equal(target_latent, saved_target_latent):
                raise RuntimeError(f"Target latent mismatch in {session_meta_path}.")
            print("  Loaded saved target metadata.")
        else:
            print("  Rendering target and computing face encoding...")
            target_image = candidate_generator.generate(target_latent[None, :])[0]
            target_encoding = compute_face_encoding(target_image)
            if target_encoding is None:
                raise RuntimeError(
                    f"Could not compute target face encoding for {session_path.name}"
                )
            _atomic_torch_save(
                {
                    "session_id": session_id,
                    "target_latent": target_latent,
                    "target_encoding": target_encoding,
                },
                session_meta_path,
            )
            print("  Saved target metadata checkpoint.")

        run_seed = SEED + session_id
        system_checkpoint_paths = {
            "baseline": session_dir / "baseline.pt",
            "original": session_dir / "original.pt",
            "ranking": session_dir / "ranking.pt",
        }

        print("\nBASELINE")
        baseline_result = run_single_system(
            system_name="baseline",
            target_latent=target_latent,
            target_encoding=target_encoding,
            starter_latents=starter_latents,
            latent_mean=latent_mean,
            latent_std=latent_std,
            candidate_generator=candidate_generator,
            seed=run_seed,
            session_id=session_id,
            checkpoint_path=system_checkpoint_paths["baseline"],
        )

        print("\nORIGINAL LEARNED")
        original_result = run_single_system(
            system_name="original",
            target_latent=target_latent,
            target_encoding=target_encoding,
            starter_latents=starter_latents,
            latent_mean=latent_mean,
            latent_std=latent_std,
            candidate_generator=candidate_generator,
            seed=run_seed,
            session_id=session_id,
            checkpoint_path=system_checkpoint_paths["original"],
        )

        print("\nRANKING-AWARE LEARNED")
        ranking_result = run_single_system(
            system_name="ranking",
            target_latent=target_latent,
            target_encoding=target_encoding,
            starter_latents=starter_latents,
            latent_mean=latent_mean,
            latent_std=latent_std,
            candidate_generator=candidate_generator,
            seed=run_seed,
            session_id=session_id,
            checkpoint_path=system_checkpoint_paths["ranking"],
        )

        session_result = {
            "session_id": session_id,
            "baseline": baseline_result,
            "original": original_result,
            "ranking": ranking_result,
        }
        results.append(session_result)

        _atomic_torch_save(
            {
                "checkpoint_version": 2,
                "session_id": session_id,
                "num_rounds": NUM_ROUNDS,
                "num_candidates": NUM_CANDIDATES,
                "result": session_result,
            },
            session_complete_path,
        )

        print("\n  SESSION CHECKPOINT COMPLETE.")
        print(f"  Saved: {session_complete_path}")
        for system_name in ("baseline", "original", "ranking"):
            values = session_result[system_name]["round_best_similarity"]
            print(
                f"  {system_name:9s}: R1={values[0]:.4f}, "
                f"R20={values[-1]:.4f}, "
                f"Δ={values[-1] - values[0]:+.4f}"
            )

    # ========================================================

    # Helper

    # ========================================================



    def collect(

        system: str,

        metric: str,

    ):



        return np.asarray(



            [



                result[

                    system

                ][metric]



                for result in results



            ],



            dtype=np.float32,

        )





    # --------------------------------------------------------

    # Similarity

    # --------------------------------------------------------



    baseline_similarity = collect(

        "baseline",

        "round_best_similarity",

    )





    original_similarity = collect(

        "original",

        "round_best_similarity",

    )





    ranking_similarity = collect(

        "ranking",

        "round_best_similarity",

    )





    # --------------------------------------------------------

    # Validity

    # --------------------------------------------------------



    baseline_validity = collect(

        "baseline",

        "round_validity",

    )





    original_validity = collect(

        "original",

        "round_validity",

    )





    ranking_validity = collect(

        "ranking",

        "round_validity",

    )





    # --------------------------------------------------------

    # Confidence

    # --------------------------------------------------------



    baseline_confidence = collect(

        "baseline",

        "round_confidence",

    )





    original_confidence = collect(

        "original",

        "round_confidence",

    )





    ranking_confidence = collect(

        "ranking",

        "round_confidence",

    )





    # ========================================================

    # R1 / R20

    # ========================================================



    baseline_r1 = (

        baseline_similarity[:, 0]

    )



    baseline_r20 = (

        baseline_similarity[:, -1]

    )





    original_r1 = (

        original_similarity[:, 0]

    )



    original_r20 = (

        original_similarity[:, -1]

    )





    ranking_r1 = (

        ranking_similarity[:, 0]

    )



    ranking_r20 = (

        ranking_similarity[:, -1]

    )





    # ========================================================

    # Improvements

    # ========================================================



    baseline_improvement = (

        baseline_r20

        - baseline_r1

    )





    original_improvement = (

        original_r20

        - original_r1

    )





    ranking_improvement = (

        ranking_r20

        - ranking_r1

    )





    # ========================================================

    # Summary dictionary

    # ========================================================



    summary = {



        "num_targets":

            NUM_TARGETS,



        "num_rounds":

            NUM_ROUNDS,



        "num_candidates":

            NUM_CANDIDATES,



        "seed":

            SEED,





        # ----------------------------------------------------

        # Baseline

        # ----------------------------------------------------



        "baseline_mean_r1":

            float(

                baseline_r1.mean()

            ),



        "baseline_mean_r20":

            float(

                baseline_r20.mean()

            ),



        "baseline_mean_improvement":

            float(

                baseline_improvement.mean()

            ),



        "baseline_candidate_validity":

            float(

                baseline_validity.mean()

            ),





        "baseline_mean_confidence_r1":

            float(

                baseline_confidence[

                    :, 0

                ].mean()

            ),



        "baseline_mean_confidence_r20":

            float(

                baseline_confidence[

                    :, -1

                ].mean()

            ),





        # ----------------------------------------------------

        # Original

        # ----------------------------------------------------



        "original_mean_r1":

            float(

                original_r1.mean()

            ),



        "original_mean_r20":

            float(

                original_r20.mean()

            ),



        "original_mean_improvement":

            float(

                original_improvement.mean()

            ),



        "original_candidate_validity":

            float(

                original_validity.mean()

            ),





        "original_mean_confidence_r1":

            float(

                original_confidence[

                    :, 0

                ].mean()

            ),



        "original_mean_confidence_r20":

            float(

                original_confidence[

                    :, -1

                ].mean()

            ),





        # ----------------------------------------------------

        # Ranking-aware

        # ----------------------------------------------------



        "ranking_mean_r1":

            float(

                ranking_r1.mean()

            ),



        "ranking_mean_r20":

            float(

                ranking_r20.mean()

            ),



        "ranking_mean_improvement":

            float(

                ranking_improvement.mean()

            ),



        "ranking_candidate_validity":

            float(

                ranking_validity.mean()

            ),





        "ranking_mean_confidence_r1":

            float(

                ranking_confidence[

                    :, 0

                ].mean()

            ),



        "ranking_mean_confidence_r20":

            float(

                ranking_confidence[

                    :, -1

                ].mean()

            ),





        # ----------------------------------------------------

        # Paired comparisons

        # ----------------------------------------------------



        "original_minus_baseline_r20":

            float(

                (

                    original_r20

                    - baseline_r20

                ).mean()

            ),



        "ranking_minus_baseline_r20":

            float(

                (

                    ranking_r20

                    - baseline_r20

                ).mean()

            ),



        "ranking_minus_original_r20":

            float(

                (

                    ranking_r20

                    - original_r20

                ).mean()

            ),





        "original_minus_baseline_improvement":

            float(

                (

                    original_improvement

                    - baseline_improvement

                ).mean()

            ),



        "ranking_minus_baseline_improvement":

            float(

                (

                    ranking_improvement

                    - baseline_improvement

                ).mean()

            ),



        "ranking_minus_original_improvement":

            float(

                (

                    ranking_improvement

                    - original_improvement

                ).mean()

            ),





        "ranking_r20_higher_than_original_count":

            int(

                np.sum(

                    ranking_r20

                    > original_r20

                )

            ),



        "ranking_r20_lower_than_original_count":

            int(

                np.sum(

                    ranking_r20

                    < original_r20

                )

            ),



        "ranking_r20_equal_original_count":

            int(

                np.sum(

                    ranking_r20

                    == original_r20

                )

            ),

    }





    # ========================================================

    # Print final results

    # ========================================================



    print(

        "\n" + "=================================================="

    )



    print(

        "THREE-WAY EVALUATION SUMMARY"

    )



    print(

        "=================================================="

    )





    print(

        "\nBASELINE"

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

        "\nORIGINAL LEARNED"

    )





    print(

        f"Mean R1 similarity  : "

        f"{summary['original_mean_r1']:.6f}"

    )



    print(

        f"Mean R20 similarity : "

        f"{summary['original_mean_r20']:.6f}"

    )



    print(

        f"Mean improvement    : "

        f"{summary['original_mean_improvement']:+.6f}"

    )



    print(

        f"Candidate validity  : "

        f"{summary['original_candidate_validity']:.6f}"

    )





    print(

        "\nRANKING-AWARE LEARNED"

    )





    print(

        f"Mean R1 similarity  : "

        f"{summary['ranking_mean_r1']:.6f}"

    )



    print(

        f"Mean R20 similarity : "

        f"{summary['ranking_mean_r20']:.6f}"

    )



    print(

        f"Mean improvement    : "

        f"{summary['ranking_mean_improvement']:+.6f}"

    )



    print(

        f"Candidate validity  : "

        f"{summary['ranking_candidate_validity']:.6f}"

    )





    # ========================================================

    # Paired comparisons

    # ========================================================



    print(

        "\n\nPAIRED COMPARISONS"

    )





    print(

        "\nOriginal - Baseline"

    )





    print(

        f"Mean R20 difference : "

        f"{summary['original_minus_baseline_r20']:+.6f}"

    )



    print(

        f"Improvement diff    : "

        f"{summary['original_minus_baseline_improvement']:+.6f}"

    )





    print(

        "\nRanking-aware - Baseline"

    )





    print(

        f"Mean R20 difference : "

        f"{summary['ranking_minus_baseline_r20']:+.6f}"

    )



    print(

        f"Improvement diff    : "

        f"{summary['ranking_minus_baseline_improvement']:+.6f}"

    )





    print(

        "\nRanking-aware - Original"

    )





    print(

        f"Mean R20 difference : "

        f"{summary['ranking_minus_original_r20']:+.6f}"

    )



    print(

        f"Improvement diff    : "

        f"{summary['ranking_minus_original_improvement']:+.6f}"

    )





    print(

        "\nRanking-aware R20:"

    )



    print(

        f"  Higher than original: "

        f"{summary['ranking_r20_higher_than_original_count']}/"

        f"{NUM_TARGETS}"

    )



    print(

        f"  Lower than original : "

        f"{summary['ranking_r20_lower_than_original_count']}/"

        f"{NUM_TARGETS}"

    )



    print(

        f"  Equal to original   : "

        f"{summary['ranking_r20_equal_original_count']}/"

        f"{NUM_TARGETS}"

    )





    # ========================================================

    # Confidence

    # ========================================================



    print(

        "\n\nCONFIDENCE"

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

        f"Original R1  : "

        f"{summary['original_mean_confidence_r1']:.6f}"

    )



    print(

        f"Original R20 : "

        f"{summary['original_mean_confidence_r20']:.6f}"

    )





    print(

        f"Ranking R1   : "

        f"{summary['ranking_mean_confidence_r1']:.6f}"

    )



    print(

        f"Ranking R20  : "

        f"{summary['ranking_mean_confidence_r20']:.6f}"

    )





    # ========================================================

    # Save artifact

    # ========================================================



    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )





    torch.save(



        {

            "summary":

                summary,



            "results":

                results,



            "evaluation_config": {



                "seed":

                    SEED,



                "num_targets":

                    NUM_TARGETS,



                "num_rounds":

                    NUM_ROUNDS,



                "num_candidates":

                    NUM_CANDIDATES,



                "original_projector":

                    str(

                        ORIGINAL_PROJECTOR_CHECKPOINT

                    ),



                "original_sampler":

                    str(

                        ORIGINAL_SAMPLER_CHECKPOINT

                    ),



                "ranking_projector":

                    str(

                        RANKING_PROJECTOR_CHECKPOINT

                    ),



                "ranking_sampler":

                    str(

                        RANKING_SAMPLER_CHECKPOINT

                    ),

            },

        },



        OUTPUT_PATH,

    )





    print(

        "\nSaved evaluation results:"

    )



    print(

        OUTPUT_PATH

    )





    print(

        "\n" + "=================================================="

    )



    print(

        "THREE-WAY EVALUATION COMPLETE"

    )



    print(

        "=================================================="

    )





# ============================================================

# Entry point

# ============================================================



if __name__ == "__main__":



    main()

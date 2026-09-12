from pathlib import Path


# ============================================================
# RecallFace - Central Configuration
# ============================================================

# ------------------------------------------------------------
# Project Root
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------
# Main Directories
# ------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


# ------------------------------------------------------------
# Raw / Processed Data
# ------------------------------------------------------------
RAW_DATA_DIR = DATA_DIR / "raw"
FFHQ_DATA_DIR = RAW_DATA_DIR / "ffhq"
CELEBA_DATA_DIR = RAW_DATA_DIR / "celeba"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


# ------------------------------------------------------------
# RecallFace Generated Data
# ------------------------------------------------------------
STARTER_SET_DIR = DATA_DIR / "starter_set"
LATENTS_DIR = DATA_DIR / "latents"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SIMULATED_SESSIONS_DIR = DATA_DIR / "simulated_sessions"


# ------------------------------------------------------------
# Pretrained Models
# ------------------------------------------------------------
DIFFAE_DIR = PRETRAINED_DIR / "diffae"


# ------------------------------------------------------------
# Trained Models
# ------------------------------------------------------------
PROJECTOR_CHECKPOINT_DIR = CHECKPOINT_DIR / "projector"
SAMPLER_CHECKPOINT_DIR = CHECKPOINT_DIR / "sampler"


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
TRAINING_RESULTS_DIR = RESULTS_DIR / "training"
EVALUATION_RESULTS_DIR = RESULTS_DIR / "evaluation"
USER_TESTING_RESULTS_DIR = RESULTS_DIR / "user_testing"


# ------------------------------------------------------------
# DiffAE / Latent Configuration
# ------------------------------------------------------------
LATENT_DIM = 512


# ------------------------------------------------------------
# Phase 2 - Starter Set
# ------------------------------------------------------------
STARTER_SET_SIZE = 12

STARTER_POOL_MIN = 500
STARTER_POOL_MAX = 1000


# ------------------------------------------------------------
# Phase 5 - Candidate Generation
# ------------------------------------------------------------
MIN_CANDIDATES = 12
MAX_CANDIDATES = 20


# ------------------------------------------------------------
# Phase 6 - Simulated Sessions
# ------------------------------------------------------------
MIN_ROUNDS = 15
MAX_ROUNDS = 20

TARGET_SESSIONS_MIN = 5000
TARGET_SESSIONS_MAX = 10000


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
RANDOM_SEED = 2026


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
# The actual device will be selected dynamically.
# Colab T4 will normally use CUDA.
DEFAULT_DEVICE = "cuda"
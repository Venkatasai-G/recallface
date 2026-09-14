import sys
from pathlib import Path

import torch

# ============================================================
# Official DiffAE repository
# ============================================================

# ============================================================
# Official DiffAE repository
# ============================================================

POSSIBLE_DIFFAE_REPOS = [
    Path("/content/diffae"),             # Google Colab
    Path("/kaggle/working/diffae"),      # Kaggle
]

DIFFAE_REPO = next(
    (
        path
        for path in POSSIBLE_DIFFAE_REPOS
        if path.exists()
    ),
    None,
)

if DIFFAE_REPO is None:
    raise FileNotFoundError(
        "Official DiffAE repository not found. "
        "Checked:\n"
        + "\n".join(
            str(path)
            for path in POSSIBLE_DIFFAE_REPOS
        )
    )

if str(DIFFAE_REPO) not in sys.path:
    sys.path.insert(0, str(DIFFAE_REPO))

# ============================================================
# Legacy PyTorch / PyTorch Lightning compatibility
# ============================================================

def _patch_legacy_checkpoint_loader():
    """
    DiffAE was created with an older PyTorch checkpoint format.

    PyTorch 2.6+ changed torch.load() to default to
    weights_only=True. The official DiffAE checkpoint contains
    legacy Lightning objects, so we explicitly use
    weights_only=False for this trusted checkpoint.
    """

    import pytorch_lightning.core.saving as pl_saving

    def _legacy_pl_load(path_or_url, map_location=None):
        with open(path_or_url, "rb") as f:
            return torch.load(
                f,
                map_location=map_location,
                weights_only=False,
            )

    pl_saving.pl_load = _legacy_pl_load


_patch_legacy_checkpoint_loader()


# ============================================================
# DiffAE imports
# ============================================================
import importlib
import sys

# ------------------------------------------------------------
# DiffAE uses generic module names such as "config".
# RecallFace also has config.py, so temporarily remove the
# RecallFace config module while importing official DiffAE.
# ------------------------------------------------------------

recallface_config = sys.modules.pop("config", None)

# Make sure the official DiffAE repository has priority.
if str(DIFFAE_REPO) in sys.path:
    sys.path.remove(str(DIFFAE_REPO))

sys.path.insert(0, str(DIFFAE_REPO))

# Import the official DiffAE modules.
diffae_config = importlib.import_module("config")
experiment_module = importlib.import_module("experiment")
choices_module = importlib.import_module("choices")

# Restore RecallFace's config module for the rest of our project.
if recallface_config is not None:
    sys.modules["config"] = recallface_config

# Expose the DiffAE classes/functions needed by this loader.
LitModel = experiment_module.LitModel
TrainConfig = diffae_config.TrainConfig
ModelName = choices_module.ModelName


# ============================================================
# Model loader
# ============================================================

def load_diffae(checkpoint_path):
    """
    Load the official pretrained FFHQ256 DiffAE model.

    This function performs inference-model loading only.
    It does not train DiffAE or modify its architecture.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DiffAE checkpoint not found: {checkpoint_path}"
        )

    print("Loading DiffAE checkpoint metadata...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    hparams = checkpoint["hyper_parameters"]

    # Current TrainConfig fields.
    valid_fields = set(
        TrainConfig.__dataclass_fields__.keys()
    )

    config_kwargs = {
        key: value
        for key, value in hparams.items()
        if key in valid_fields
    }

    conf = TrainConfig(**config_kwargs)

    # The old checkpoint does not contain model_name.
    # Its stored model configuration is the BeatGANs Autoencoder.
    conf.model_name = ModelName.beatgans_autoenc

    # Preserve the exact model configuration stored in the checkpoint.
    if "model_conf" in config_kwargs:
        conf.model_conf = config_kwargs["model_conf"]

    print("============================================")
    print("Building pretrained DiffAE")
    print("============================================")
    print("Model:", conf.model_name)
    print("Image size:", conf.img_size)
    print("Style dimension:", conf.style_ch)

    model = LitModel(conf)

    print("Loading pretrained weights...")

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["state_dict"],
        strict=False,
    )

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "DiffAE checkpoint/model mismatch.\n"
            f"Missing keys: {len(missing_keys)}\n"
            f"Unexpected keys: {len(unexpected_keys)}"
        )

    model.eval()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)

    print("============================================")
    print("DiffAE loaded successfully.")
    print("============================================")
    print("Device:", device)
    print("Image size:", conf.img_size)
    print("Style dimension:", conf.style_ch)
    print(
        "Model parameters:",
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M",
    )
    print("Missing keys:", len(missing_keys))
    print("Unexpected keys:", len(unexpected_keys))

    return model, conf, device
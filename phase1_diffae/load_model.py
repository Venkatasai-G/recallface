import sys
from pathlib import Path

import torch

# Add the official DiffAE repository to Python's import path.
DIFFAE_REPO = Path(__file__).resolve().parents[1] / "pretrained" / "diffae"

if str(DIFFAE_REPO) not in sys.path:
    sys.path.insert(0, str(DIFFAE_REPO))

from experiment import LitModel
from config import TrainConfig
from choices import ModelName


def load_diffae(checkpoint_path):
    """
    Load the official pretrained DiffAE checkpoint.

    This function performs inference-model loading only.
    It does not train or modify the DiffAE architecture.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DiffAE checkpoint not found: {checkpoint_path}"
        )

    # Load the legacy DiffAE checkpoint explicitly.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    hparams = checkpoint["hyper_parameters"]

    # Reconstruct only fields supported by the current TrainConfig.
    valid_fields = set(TrainConfig.__dataclass_fields__.keys())

    config_kwargs = {
        key: value
        for key, value in hparams.items()
        if key in valid_fields
    }

    conf = TrainConfig(**config_kwargs)

    # The old checkpoint does not contain model_name,
    # but its stored model configuration is the BeatGANs Autoencoder.
    conf.model_name = ModelName.beatgans_autoenc

    # Preserve the exact model configuration stored in the checkpoint.
    if "model_conf" in config_kwargs:
        conf.model_conf = config_kwargs["model_conf"]

    print("Building pretrained DiffAE...")
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
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    print("============================================")
    print("DiffAE loaded successfully.")
    print("Device:", device)
    print("Image size:", conf.img_size)
    print("Style dimension:", conf.style_ch)
    print("============================================")

    return model, conf, device
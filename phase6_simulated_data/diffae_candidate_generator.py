from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Official DiffAE source copied/available through our project setup.
DIFFAE_REPO = PROJECT_ROOT / "pretrained" / "diffae"

# Pretrained DiffAE checkpoint.
DIFFAE_CHECKPOINT = (
    PROJECT_ROOT
    / "pretrained"
    / "diffae"
    / "checkpoints"
    / "last.ckpt"
)

# Fixed stochastic latent used by Phase 2.
STARTER_XT = (
    PROJECT_ROOT
    / "data"
    / "starter_set"
    / "starter_xT.pt"
)


class DiffAECandidateGenerator:
    """
    Convert RecallFace latent vectors into face images
    using the pretrained DiffAE model.

    This class performs inference only.
    No DiffAE training occurs here.
    """

    def __init__(
        self,
        checkpoint_path: Path = DIFFAE_CHECKPOINT,
        xT_path: Path = STARTER_XT,
        device: str = "cuda",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.xT_path = Path(xT_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"DiffAE checkpoint not found:\n"
                f"{self.checkpoint_path}"
            )

        if not self.xT_path.exists():
            raise FileNotFoundError(
                f"Phase 2 x_T not found:\n"
                f"{self.xT_path}"
            )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        self.device = torch.device(device)

        self.model = None
        self.conf = None
        self.xT = None

    def load(self):
        """
        Load the pretrained DiffAE model and fixed Phase 2 x_T.
        """

        print("Loading DiffAE...")

        # Import our existing Phase 1 loader.
        from phase1_diffae.load_model import load_diffae_model

        self.model, self.conf = load_diffae_model(
            checkpoint_path=self.checkpoint_path,
            device=str(self.device),
        )

        # Load the fixed stochastic latent from Phase 2.
        self.xT = torch.load(
            self.xT_path,
            map_location=self.device,
            weights_only=False,
        )

        if self.xT.ndim != 4:
            raise ValueError(
                f"Expected x_T shape "
                f"(1, 3, H, W), got {tuple(self.xT.shape)}"
            )

        print("DiffAE loaded.")
        print(f"Device: {self.device}")
        print(f"Image size: {self.conf.img_size}")
        print(f"Latent dimension: {self.conf.style_ch}")
        print(f"Fixed x_T shape: {tuple(self.xT.shape)}")

        return self

    @staticmethod
    def _to_numpy_images(images: torch.Tensor) -> list[np.ndarray]:
        """
        Convert tensor images in [0,1] to RGB uint8 NumPy images.
        """

        images = images.detach().cpu().clamp(0, 1)

        result = []

        for image in images:
            image = (
                image.permute(1, 2, 0)
                .numpy()
                * 255.0
            )

            image = np.round(image).astype(np.uint8)

            result.append(image)

        return result

    def generate(
        self,
        candidate_latents: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Generate face images from candidate latent vectors.

        Args:
            candidate_latents:
                NumPy array with shape (N, 512).

        Returns:
            List of RGB uint8 NumPy images.
        """

        if self.model is None:
            raise RuntimeError(
                "DiffAE is not loaded. "
                "Call load() first."
            )

        candidate_latents = np.asarray(
            candidate_latents,
            dtype=np.float32,
        )

        if candidate_latents.ndim != 2:
            raise ValueError(
                "candidate_latents must have shape (N, 512)."
            )

        if candidate_latents.shape[1] != 512:
            raise ValueError(
                "candidate_latents must have "
                "512 latent dimensions."
            )

        z = torch.from_numpy(
            candidate_latents
        ).to(self.device)

        # Reuse the same fixed x_T for every candidate,
        # matching the Phase 2 fixed stochastic latent design.
        xT = self.xT.expand(
            z.shape[0],
            -1,
            -1,
            -1,
        ).contiguous()

        with torch.no_grad():
            generated = self.model.render(
                noise=xT,
                cond=z,
                T=self.conf.T_eval,
            )

        return self._to_numpy_images(generated)
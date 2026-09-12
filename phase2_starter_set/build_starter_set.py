from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from PIL import Image


def sample_latent_pool(
    conds: torch.Tensor,
    pool_size: int = 1000,
    seed: int = 2026,
) -> torch.Tensor:
    """
    Sample latent codes from the learned DiffAE conditioning pool.

    The pretrained DiffAE provides 70,000 learned 512-D conditioning
    vectors. We sample a fixed subset from that empirical distribution.
    """

    if conds.ndim != 2:
        raise ValueError(
            f"Expected conds with shape (N, latent_dim), "
            f"got {tuple(conds.shape)}"
        )

    if pool_size < 500 or pool_size > 1000:
        raise ValueError(
            "Phase 2 pool_size must be between 500 and 1000."
        )

    if pool_size > conds.shape[0]:
        raise ValueError(
            "pool_size cannot exceed the number of learned latent codes."
        )

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        conds.shape[0],
        size=pool_size,
        replace=False,
    )

    indices = torch.from_numpy(indices).long()

    return conds[indices].clone()


def select_representative_latents(
    latent_pool: torch.Tensor,
    starter_size: int = 12,
    seed: int = 2026,
) -> torch.Tensor:
    """
    Cluster the sampled latent pool and select the actual latent code
    nearest to each KMeans centroid.
    """

    if latent_pool.ndim != 2:
        raise ValueError(
            f"Expected latent_pool with shape (N, latent_dim), "
            f"got {tuple(latent_pool.shape)}"
        )

    if starter_size != 12:
        raise ValueError(
            "The RecallFace Phase 2 starter set must contain 12 faces."
        )

    if latent_pool.shape[0] < starter_size:
        raise ValueError(
            "Latent pool is smaller than the requested starter set."
        )

    latent_np = latent_pool.cpu().numpy()

    print("Running KMeans...")
    print("  Pool size:", latent_np.shape[0])
    print("  Latent dimension:", latent_np.shape[1])
    print("  Number of clusters:", starter_size)

    kmeans = KMeans(
        n_clusters=starter_size,
        random_state=seed,
        n_init=10,
    )

    labels = kmeans.fit_predict(latent_np)
    centers = kmeans.cluster_centers_

    selected_indices = []

    for cluster_id in range(starter_size):
        cluster_indices = np.where(labels == cluster_id)[0]

        cluster_vectors = latent_np[cluster_indices]

        distances = np.linalg.norm(
            cluster_vectors - centers[cluster_id],
            axis=1,
        )

        nearest_position = np.argmin(distances)

        selected_indices.append(
            cluster_indices[nearest_position]
        )

    # Sort indices to make the saved starter-set ordering deterministic.
    selected_indices = np.array(sorted(selected_indices))

    selected_latents = latent_pool[
        torch.from_numpy(selected_indices).long()
    ].clone()

    return selected_latents


def create_fixed_xT(
    starter_size: int,
    image_size: int,
    seed: int = 2026,
) -> torch.Tensor:
    """
    Create one stochastic xT tensor.

    The same xT is reused for all starter faces,
    as required by the Phase 2 starter-set procedure.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    return torch.randn(
        1,
        3,
        image_size,
        image_size,
        generator=generator,
    )
    
def save_latents(
    latents: torch.Tensor,
    output_path: Path,
) -> None:
    """Save the permanent starter latent codes."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        latents.cpu(),
        output_path,
    )


def save_xT(
    x_T: torch.Tensor,
    output_path: Path,
) -> None:
    """Save the fixed stochastic xT used by the starter set."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        x_T.cpu(),
        output_path,
    )


def save_image(
    image_tensor: torch.Tensor,
    output_path: Path,
) -> None:
    """Save a single DiffAE output tensor as PNG."""

    image = image_tensor.detach().cpu()

    image = image.clamp(0, 1)

    image = (
        image.permute(1, 2, 0)
        .numpy()
        * 255
    ).round().astype(np.uint8)

    Image.fromarray(image).save(output_path)


def save_starter_images(
    images: torch.Tensor,
    output_dir: Path,
) -> None:
    """Save all 12 starter faces as numbered PNG files."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, image in enumerate(images):
        output_path = output_dir / f"starter_{index + 1:02d}.png"
        save_image(image, output_path)
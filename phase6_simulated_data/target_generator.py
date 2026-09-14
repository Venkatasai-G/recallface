import torch


LATENT_DIM = 512


def validate_target_latent(
    target_latent: torch.Tensor,
) -> torch.Tensor:
    """
    Validate and return a target latent in the required
    512-dimensional format.

    Expected input:
        (512,) or (1, 512)

    Returns:
        (512,)
    """

    if not isinstance(target_latent, torch.Tensor):
        raise TypeError(
            "target_latent must be a torch.Tensor"
        )

    if target_latent.ndim == 2:
        if target_latent.shape != (1, LATENT_DIM):
            raise ValueError(
                f"Expected shape (1, {LATENT_DIM}), "
                f"got {tuple(target_latent.shape)}"
            )

        target_latent = target_latent.squeeze(0)

    elif target_latent.ndim == 1:
        if target_latent.shape[0] != LATENT_DIM:
            raise ValueError(
                f"Expected dimension {LATENT_DIM}, "
                f"got {target_latent.shape[0]}"
            )

    else:
        raise ValueError(
            "Target latent must have shape "
            "(512,) or (1, 512)"
        )

    return target_latent.float().contiguous()


def sample_target_latent(
    latent_pool: torch.Tensor,
    seed: int = 2026,
) -> torch.Tensor:
    """
    Select one target latent from the learned DiffAE
    latent distribution.

    This function is intended for simulated training
    sessions. It samples one existing latent from the
    provided learned latent pool.

    Args:
        latent_pool:
            Tensor of shape (N, 512).

        seed:
            Random seed for reproducibility.

    Returns:
        Target latent of shape (512,).
    """

    if not isinstance(latent_pool, torch.Tensor):
        raise TypeError(
            "latent_pool must be a torch.Tensor"
        )

    if latent_pool.ndim != 2:
        raise ValueError(
            "latent_pool must have shape (N, 512)"
        )

    if latent_pool.shape[1] != LATENT_DIM:
        raise ValueError(
            f"Expected latent dimension {LATENT_DIM}, "
            f"got {latent_pool.shape[1]}"
        )

    if latent_pool.shape[0] == 0:
        raise ValueError(
            "latent_pool cannot be empty"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    index = torch.randint(
        low=0,
        high=latent_pool.shape[0],
        size=(1,),
        generator=generator,
    ).item()

    return validate_target_latent(
        latent_pool[index].clone()
    )
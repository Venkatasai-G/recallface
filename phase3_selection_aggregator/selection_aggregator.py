import torch


def aggregate_selected_latents(
    selected_latents: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Aggregate selected latent codes using a weighted average.

    Args:
        selected_latents:
            Tensor of shape (N, 512).

        weights:
            Optional tensor of shape (N,).
            If None, equal weighting is used.

    Returns:
        Aggregated latent tensor of shape (512,).
    """

    if not isinstance(selected_latents, torch.Tensor):
        raise TypeError("selected_latents must be a torch.Tensor")

    if selected_latents.ndim != 2:
        raise ValueError(
            "selected_latents must have shape (N, 512)"
        )

    if selected_latents.shape[1] != 512:
        raise ValueError(
            f"Expected latent dimension 512, "
            f"got {selected_latents.shape[1]}"
        )

    if selected_latents.shape[0] == 0:
        raise ValueError(
            "At least one selected latent is required"
        )

    # Equal weighting when no confidence is provided.
    if weights is None:
        weights = torch.ones(
            selected_latents.shape[0],
            dtype=selected_latents.dtype,
            device=selected_latents.device,
        )
    else:
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor")

        if weights.ndim != 1:
            raise ValueError("weights must have shape (N,)")

        if weights.shape[0] != selected_latents.shape[0]:
            raise ValueError(
                "Number of weights must match "
                "number of selected latents"
            )

        weights = weights.to(
            device=selected_latents.device,
            dtype=selected_latents.dtype,
        )

    if torch.any(weights < 0):
        raise ValueError("weights cannot be negative")

    weight_sum = weights.sum()

    if weight_sum <= 0:
        raise ValueError(
            "Sum of weights must be greater than zero"
        )

    # Weighted average:
    #
    # z_agg = sum(w_i * z_i) / sum(w_i)
    aggregated_latent = (
        selected_latents * weights.unsqueeze(1)
    ).sum(dim=0) / weight_sum

    return aggregated_latent
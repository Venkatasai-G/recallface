import torch
import torch.nn as nn
import torch.nn.functional as F


class ExplorationSampler(nn.Module):
    """
    Exploration-Exploitation Sampler.

    Input:
        - Current latent z: (batch_size, 512)
        - Round number: (batch_size, 1)

    Output:
        - Variance for each latent/semantic dimension:
          (batch_size, 512)

    The network is initially randomly initialized.
    Actual training is performed later in Phase 7.
    """

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 1024,
    ):
        super().__init__()

        if latent_dim != 512:
            raise ValueError(
                f"Latent dimension must be 512, got {latent_dim}"
            )

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Current latent + round number.
        input_dim = latent_dim + 1

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        round_number: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict exploration variance for every latent dimension.

        Args:
            z:
                Current latent tensor of shape (batch_size, 512).

            round_number:
                Round numbers of shape (batch_size, 1).

        Returns:
            Positive variance tensor of shape (batch_size, 512).
        """

        if z.ndim != 2:
            raise ValueError(
                f"Expected z shape (batch_size, 512), "
                f"got {tuple(z.shape)}"
            )

        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected z dimension {self.latent_dim}, "
                f"got {z.shape[1]}"
            )

        if round_number.ndim != 2:
            raise ValueError(
                f"Expected round_number shape (batch_size, 1), "
                f"got {tuple(round_number.shape)}"
            )

        if round_number.shape[0] != z.shape[0]:
            raise ValueError(
                "Batch size of round_number must match z"
            )

        if round_number.shape[1] != 1:
            raise ValueError(
                "round_number must have shape (batch_size, 1)"
            )

        round_number = round_number.to(
            device=z.device,
            dtype=z.dtype,
        )

        # Normalize the round number so that its scale does not
        # dominate the latent representation.
        round_feature = round_number / 20.0

        x = torch.cat(
            [z, round_feature],
            dim=1,
        )

        raw_variance = self.network(x)

        # Softplus guarantees strictly positive variance.
        variance = F.softplus(raw_variance) + 1e-6

        return variance

    def sample_candidates(
        self,
        z: torch.Tensor,
        round_number: torch.Tensor,
        num_candidates: int = 12,
    ) -> torch.Tensor:
        """
        Generate candidate latent codes using predicted variance.

        Args:
            z:
                Current latent tensor of shape (batch_size, 512).

            round_number:
                Round numbers of shape (batch_size, 1).

            num_candidates:
                Number of candidates to generate.
                Must be between 12 and 20.

        Returns:
            Candidate latents with shape:
            (batch_size, num_candidates, 512)
        """

        if not 12 <= num_candidates <= 20:
            raise ValueError(
                "num_candidates must be between 12 and 20"
            )

        variance = self.forward(
            z,
            round_number,
        )

        standard_deviation = torch.sqrt(variance)

        batch_size = z.shape[0]

        noise = torch.randn(
            batch_size,
            num_candidates,
            self.latent_dim,
            device=z.device,
            dtype=z.dtype,
        )

        candidates = (
            z.unsqueeze(1)
            + noise * standard_deviation.unsqueeze(1)
        )

        return candidates
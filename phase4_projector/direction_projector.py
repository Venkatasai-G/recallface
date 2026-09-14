import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual fully connected block.

    Input and output dimensions are the same so that
    the input can be added back to the transformed output.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DirectionProjector(nn.Module):
    """
    Disentangled Direction Projector.

    Input:
        512-dimensional latent vector.

    Output:
        512-dimensional projected direction.

    Architecture:
        Input projection
        -> residual MLP blocks
        -> output projection
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        num_blocks: int = 3,
        output_dim: int = 512,
    ):
        super().__init__()

        if input_dim != 512:
            raise ValueError(
                f"Projector input dimension must be 512, got {input_dim}"
            )

        if output_dim != 512:
            raise ValueError(
                f"Projector output dimension must be 512, got {output_dim}"
            )

        if num_blocks < 1:
            raise ValueError(
                "num_blocks must be at least 1"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.output_dim = output_dim

        self.input_projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    dim=hidden_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(num_blocks)
            ]
        )

        self.output_projection = nn.Linear(
            hidden_dim,
            output_dim,
        )

        # Phase 6 bootstrap initialization:
        # start with an approximately zero direction so that the
        # untrained projector does not move the latent off-manifold.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project a 512-dimensional latent vector.

        Args:
            z:
                Tensor with shape (batch_size, 512).

        Returns:
            Tensor with shape (batch_size, 512).
        """

        if z.ndim != 2:
            raise ValueError(
                f"Expected input shape (batch_size, 512), got {tuple(z.shape)}"
            )

        if z.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {z.shape[1]}"
            )

        x = self.input_projection(z)

        x = self.residual_blocks(x)

        direction = self.output_projection(x)

        return direction
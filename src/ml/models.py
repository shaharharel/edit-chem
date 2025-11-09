"""
Neural network models for edit-based property prediction.
"""

import torch
import torch.nn as nn
from typing import List


class EditMLP(nn.Module):
    """
    Multi-layer perceptron for edit-based property prediction.

    Architecture:
        Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x N -> Linear -> Output

    Args:
        input_dim: Input dimension (embedding size)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability (default: 0.1)
        use_batchnorm: Whether to use batch normalization

    Example:
        >>> model = EditMLP(
        ...     input_dim=2048,  # Morgan fingerprint size
        ...     hidden_dims=[1024, 512, 256],
        ...     dropout=0.2
        ... )
        >>> x = torch.randn(32, 2048)  # batch_size=32
        >>> y_pred = model(x)  # shape: (32, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.1,
        use_batchnorm: bool = True
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Output tensor (batch_size, 1)
        """
        return self.network(x)


class EditResidualMLP(nn.Module):
    """
    MLP with residual connections for edit-based property prediction.

    More powerful than basic MLP, with skip connections to help gradient flow.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions (must be same size for residuals)
        dropout: Dropout probability
        use_batchnorm: Whether to use batch normalization

    Example:
        >>> model = EditResidualMLP(
        ...     input_dim=2048,
        ...     hidden_dims=[512, 512, 512],  # Same size for residuals
        ...     dropout=0.2
        ... )
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 512, 512],
        dropout: float = 0.1,
        use_batchnorm: bool = True
    ):
        super().__init__()

        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # First block doesn't need same input/output dim
                block = self._make_block(hidden_dim, hidden_dim, dropout, use_batchnorm)
            else:
                # Check if we can use residual
                if hidden_dim == hidden_dims[i-1]:
                    block = self._make_block(hidden_dim, hidden_dim, dropout, use_batchnorm)
                else:
                    # Different dimensions, use projection
                    block = self._make_block(hidden_dims[i-1], hidden_dim, dropout, use_batchnorm)

            self.blocks.append(block)

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)

    def _make_block(self, in_dim, out_dim, dropout, use_batchnorm):
        """Create a residual block."""
        layers = []
        layers.append(nn.Linear(in_dim, out_dim))

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass with residual connections."""
        x = self.input_proj(x)

        for block in self.blocks:
            # Residual connection if dimensions match
            identity = x
            x = block(x)
            if x.shape == identity.shape:
                x = x + identity  # Residual connection

        return self.output(x)


# Convenience function
def create_model(
    input_dim: int,
    model_type: str = 'mlp',
    hidden_dims: List[int] = [512, 256, 128],
    dropout: float = 0.1,
    device: str = 'cpu'
) -> nn.Module:
    """
    Create a model and move to device.

    Args:
        input_dim: Input embedding dimension
        model_type: 'mlp' or 'residual'
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        device: Device to use

    Returns:
        Model on specified device

    Example:
        >>> model = create_model(
        ...     input_dim=2048,
        ...     model_type='mlp',
        ...     hidden_dims=[1024, 512, 256],
        ...     device='cuda' if torch.cuda.is_available() else 'cpu'
        ... )
    """
    if model_type == 'mlp':
        model = EditMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
    elif model_type == 'residual':
        model = EditResidualMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'mlp' or 'residual'")

    return model.to(device)

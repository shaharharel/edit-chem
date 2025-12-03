"""
SE(3)-Transformer Structural Encoder.

SE(3)-Transformer is an equivariant attention network that operates on
3D point clouds while respecting SE(3) symmetry (rotation + translation).

Reference:
    Fuchs et al. "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks"
    NeurIPS 2020
    https://github.com/FabianFuchsML/se3-transformer-public

Installation:
    pip install se3-transformer-pytorch
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

from ..base import StructuralEncoder


class SE3TransformerEncoder(StructuralEncoder):
    """
    SE(3)-Transformer structural encoder.

    This encoder uses SE(3)-equivariant self-attention to process
    3D protein structures.

    Args:
        hidden_dim: Hidden dimension (default: 128)
        num_layers: Number of transformer layers (default: 3)
        num_heads: Number of attention heads (default: 4)
        num_degrees: Number of spherical harmonic degrees (default: 2)
        div: Division factor for head dimension (default: 4)
        n_neighbors: Number of neighbors for attention (default: 32)
        device: Device to use

    Example:
        >>> encoder = SE3TransformerEncoder()
        >>> coords = torch.randn(100, 3)  # CA coordinates
        >>> embeddings = encoder.encode(coords, "EVQL...")
        >>> print(embeddings.shape)
        torch.Size([100, 128])

    Note:
        Requires: pip install se3-transformer-pytorch
        For better performance, consider using Equiformer instead.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_degrees: int = 2,
        div: int = 4,
        n_neighbors: int = 32,
        device: str = 'auto',
    ):
        super().__init__(
            node_dim=hidden_dim,
            edge_dim=hidden_dim // 4,
            num_layers=num_layers,
            device=device,
        )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_degrees = num_degrees
        self.div = div
        self.n_neighbors = n_neighbors

        self._load_model()

    def _load_model(self):
        """Load SE(3)-Transformer model."""
        try:
            from se3_transformer_pytorch import SE3Transformer
            self._se3_available = True

            self.model = SE3Transformer(
                dim=self.hidden_dim,
                heads=self.num_heads,
                depth=self.num_layers,
                num_degrees=self.num_degrees,
                num_positions=None,  # Use coordinates directly
                reduce_dim_out=True,
                input_degrees=1,
                output_degrees=1,
            )

        except ImportError:
            import warnings
            warnings.warn(
                "se3-transformer-pytorch not found. Using fallback. "
                "Install with: pip install se3-transformer-pytorch"
            )
            self._se3_available = False
            self._build_fallback()

        self.to(self._device)

    def _build_fallback(self):
        """Build fallback model without SE3-Transformer."""
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(self.num_layers)
        ])

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim

    @property
    def name(self) -> str:
        return "SE3-Transformer"

    def _extract_ca_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Extract CA coordinates."""
        if coords.dim() == 3 and coords.shape[1] == 4:
            return coords[:, 1, :]  # CA is index 1
        elif coords.dim() == 2 and coords.shape[1] == 3:
            return coords
        else:
            raise ValueError(f"Unexpected coords shape: {coords.shape}")

    def encode(
        self,
        coords: torch.Tensor,
        sequence: str,
        chain_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Encode a protein structure.

        Args:
            coords: Backbone coordinates [L, 4, 3] or [L, 3]
            sequence: Amino acid sequence
            chain_ids: Optional chain IDs

        Returns:
            Per-residue structural embeddings [L, embedding_dim]
        """
        coords = coords.to(self._device)
        ca_coords = self._extract_ca_coords(coords)

        if self._se3_available:
            return self._encode_se3(ca_coords, sequence)
        else:
            return self._encode_fallback(ca_coords, sequence)

    def _encode_se3(
        self,
        coords: torch.Tensor,
        sequence: str,
    ) -> torch.Tensor:
        """Encode using SE(3)-Transformer."""
        L = coords.shape[0]

        # SE3-Transformer expects:
        # - feats: dict with 'type0' for scalar features, 'type1' for vector features
        # - coors: coordinates [B, L, 3]
        # - mask: attention mask

        # Create simple scalar features (learnable per-position embedding)
        feats = {
            '0': torch.zeros(1, L, self.hidden_dim, device=self._device),
        }

        # Add batch dimension to coordinates
        coors = coords.unsqueeze(0)  # [1, L, 3]

        # Forward pass
        with torch.set_grad_enabled(self.training):
            out = self.model(feats, coors)

        # Extract scalar output
        if isinstance(out, dict):
            embeddings = out.get('0', out.get('type0', None))
            if embeddings is None:
                embeddings = list(out.values())[0]
        else:
            embeddings = out

        # Remove batch dimension
        if embeddings.dim() == 3:
            embeddings = embeddings[0]

        return embeddings

    def _encode_fallback(
        self,
        coords: torch.Tensor,
        sequence: str,
    ) -> torch.Tensor:
        """Fallback encoding without SE3-Transformer."""
        L = coords.shape[0]

        # Simple coordinate encoding
        h = self.coord_encoder(coords)  # [L, hidden_dim]
        h = h.unsqueeze(0)  # [1, L, hidden_dim]

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)

        return h[0]  # Remove batch dim

    def encode_batch(
        self,
        coords_list: List[torch.Tensor],
        sequences: List[str],
        chain_ids_list: Optional[List[List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of structures.

        Args:
            coords_list: List of coordinate tensors
            sequences: List of sequences
            chain_ids_list: Optional list of chain ID lists

        Returns:
            embeddings: [B, L_max, embedding_dim]
            mask: [B, L_max]
        """
        embeddings = []
        lengths = []

        for i, (coords, seq) in enumerate(zip(coords_list, sequences)):
            chain_ids = chain_ids_list[i] if chain_ids_list else None
            emb = self.encode(coords, seq, chain_ids)
            embeddings.append(emb)
            lengths.append(emb.shape[0])

        # Pad and stack
        max_len = max(lengths)
        batch_size = len(embeddings)

        padded = torch.zeros(batch_size, max_len, self.embedding_dim, device=self._device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self._device)

        for i, (emb, length) in enumerate(zip(embeddings, lengths)):
            padded[i, :length] = emb
            mask[i, :length] = True

        return padded, mask

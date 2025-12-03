"""
Equiformer Structural Encoder.

Equiformer is a state-of-the-art SE(3)/E(3) equivariant graph transformer
that combines the benefits of transformers with geometric deep learning.
It has been adopted by EquiFold for protein folding.

Reference:
    Liao & Smidt "Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs"
    ICLR 2023
    https://github.com/lucidrains/equiformer-pytorch

Installation:
    pip install equiformer-pytorch
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

from ..base import StructuralEncoder


class EquiformerEncoder(StructuralEncoder):
    """
    Equiformer structural encoder.

    Equiformer combines equivariant graph attention with MLP attention
    from GATv2 and non-linear message passing. It achieves SOTA results
    on molecular property prediction tasks.

    Args:
        hidden_dim: Hidden dimension (default: 128)
        num_layers: Number of transformer layers (default: 4)
        num_heads: Number of attention heads (default: 8)
        max_degree: Maximum spherical harmonic degree (default: 2)
        dim_head: Dimension per head (default: 64)
        reduce_dim_out: Whether to reduce output to scalars (default: True)
        num_neighbors: Number of neighbors for attention (default: 32)
        device: Device to use

    Example:
        >>> encoder = EquiformerEncoder()
        >>> coords = torch.randn(100, 3)  # CA coordinates
        >>> embeddings = encoder.encode(coords, "EVQL...")
        >>> print(embeddings.shape)
        torch.Size([100, 128])

    Note:
        Requires: pip install equiformer-pytorch
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        max_degree: int = 2,
        dim_head: int = 64,
        reduce_dim_out: bool = True,
        num_neighbors: int = 32,
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
        self.max_degree = max_degree
        self.dim_head = dim_head
        self.reduce_dim_out = reduce_dim_out
        self.num_neighbors = num_neighbors

        self._load_model()

    def _load_model(self):
        """Load Equiformer model."""
        try:
            from equiformer_pytorch import Equiformer
            self._equiformer_available = True

            self.model = Equiformer(
                num_tokens=21,  # 20 amino acids + padding
                dim=self.hidden_dim,
                depth=self.num_layers,
                heads=self.num_heads,
                dim_head=self.dim_head,
                num_degrees=self.max_degree + 1,
                reduce_dim_out=self.reduce_dim_out,
            )

        except ImportError:
            import warnings
            warnings.warn(
                "equiformer-pytorch not found. Using fallback. "
                "Install with: pip install equiformer-pytorch"
            )
            self._equiformer_available = False
            self._build_fallback()

        self.to(self._device)

    def _build_fallback(self):
        """Build fallback model without Equiformer."""
        # Use a simple graph transformer as fallback
        self.node_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True,
                activation='gelu',
            )
            for _ in range(self.num_layers)
        ])

        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim

    @property
    def name(self) -> str:
        return "Equiformer"

    def _extract_ca_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Extract CA coordinates."""
        if coords.dim() == 3 and coords.shape[1] == 4:
            return coords[:, 1, :]
        elif coords.dim() == 2 and coords.shape[1] == 3:
            return coords
        else:
            raise ValueError(f"Unexpected coords shape: {coords.shape}")

    def _sequence_to_tokens(self, sequence: str) -> torch.Tensor:
        """Convert sequence to token indices."""
        AA_TO_IDX = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        }
        tokens = [AA_TO_IDX.get(aa.upper(), 0) for aa in sequence]
        return torch.tensor(tokens, dtype=torch.long, device=self._device)

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

        if self._equiformer_available:
            return self._encode_equiformer(ca_coords, sequence)
        else:
            return self._encode_fallback(ca_coords, sequence)

    def _encode_equiformer(
        self,
        coords: torch.Tensor,
        sequence: str,
    ) -> torch.Tensor:
        """Encode using Equiformer."""
        L = coords.shape[0]

        # Convert sequence to tokens
        tokens = self._sequence_to_tokens(sequence)
        tokens = tokens.unsqueeze(0)  # [1, L]

        # Add batch dimension to coordinates
        coors = coords.unsqueeze(0)  # [1, L, 3]

        # Create mask (all valid)
        mask = torch.ones(1, L, dtype=torch.bool, device=self._device)

        # Forward pass
        with torch.set_grad_enabled(self.training):
            out = self.model(tokens, coors, mask=mask)

        # Handle output format
        if isinstance(out, tuple):
            embeddings = out[0]
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
        """Fallback encoding without Equiformer."""
        L = coords.shape[0]

        # Encode coordinates
        h = self.node_encoder(coords)  # [L, hidden_dim]
        h = h.unsqueeze(0)  # [1, L, hidden_dim]

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)

        # Project output
        h = self.output_proj(h)

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
        # For Equiformer, we can do true batched processing
        if self._equiformer_available:
            return self._encode_batch_equiformer(coords_list, sequences)

        # Fallback: process individually
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

    def _encode_batch_equiformer(
        self,
        coords_list: List[torch.Tensor],
        sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch encode using Equiformer."""
        batch_size = len(coords_list)
        lengths = [coords.shape[0] if coords.dim() == 2 else coords.shape[0] for coords in coords_list]
        max_len = max(lengths)

        # Pad coordinates
        padded_coords = torch.zeros(batch_size, max_len, 3, device=self._device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self._device)

        for i, coords in enumerate(coords_list):
            ca_coords = self._extract_ca_coords(coords.to(self._device))
            L = ca_coords.shape[0]
            padded_coords[i, :L] = ca_coords
            mask[i, :L] = True

        # Pad tokens
        padded_tokens = torch.zeros(batch_size, max_len, dtype=torch.long, device=self._device)
        for i, seq in enumerate(sequences):
            tokens = self._sequence_to_tokens(seq)
            padded_tokens[i, :len(tokens)] = tokens

        # Forward pass
        with torch.set_grad_enabled(self.training):
            out = self.model(padded_tokens, padded_coords, mask=mask)

        if isinstance(out, tuple):
            embeddings = out[0]
        else:
            embeddings = out

        return embeddings, mask

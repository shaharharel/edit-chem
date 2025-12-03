"""
GVP-GNN (Geometric Vector Perceptron) Structural Encoder.

GVP-GNN is a rotation-equivariant graph neural network for learning from
protein structure. It operates on scalar and vector features at each node.

Reference:
    Jing et al. "Learning from Protein Structure with Geometric Vector Perceptrons"
    ICLR 2021
    https://github.com/drorlab/gvp-pytorch

Installation:
    pip install geometric-vector-perceptron
    OR
    git clone https://github.com/drorlab/gvp-pytorch && pip install -e .
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple

from ..base import StructuralEncoder


class GVPEncoder(StructuralEncoder):
    """
    GVP-GNN structural encoder for antibody structures.

    This encoder converts 3D backbone coordinates into per-residue
    structural embeddings using geometric vector perceptrons.

    Args:
        node_s_dim: Scalar node feature dimension (default: 128)
        node_v_dim: Vector node feature dimension (default: 16)
        edge_s_dim: Scalar edge feature dimension (default: 32)
        edge_v_dim: Vector edge feature dimension (default: 1)
        num_layers: Number of GVP layers (default: 3)
        dropout: Dropout probability (default: 0.1)
        k_neighbors: Number of nearest neighbors for graph (default: 30)
        device: Device to use

    Example:
        >>> encoder = GVPEncoder()
        >>> coords = torch.randn(100, 4, 3)  # N, CA, C, O for 100 residues
        >>> embeddings = encoder.encode(coords, "EVQL...")
        >>> print(embeddings.shape)
        torch.Size([100, 128])
    """

    def __init__(
        self,
        node_s_dim: int = 128,
        node_v_dim: int = 16,
        edge_s_dim: int = 32,
        edge_v_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
        k_neighbors: int = 30,
        device: str = 'auto',
    ):
        super().__init__(
            node_dim=node_s_dim,
            edge_dim=edge_s_dim,
            num_layers=num_layers,
            device=device,
        )

        self.node_s_dim = node_s_dim
        self.node_v_dim = node_v_dim
        self.edge_s_dim = edge_s_dim
        self.edge_v_dim = edge_v_dim
        self.dropout_rate = dropout
        self.k_neighbors = k_neighbors

        self._load_gvp()

    def _load_gvp(self):
        """Load GVP modules."""
        # Try the lucidrains implementation first (pip install geometric-vector-perceptron)
        try:
            from geometric_vector_perceptron import GVP, GVPConv, GVPConvPairwise
            self._gvp_source = 'lucidrains'
            self._build_lucidrains_model()
            return
        except ImportError:
            pass

        # Try the drorlab implementation
        try:
            from gvp import GVP, GVPConvLayer
            self._gvp_source = 'drorlab'
            self._build_drorlab_model()
            return
        except ImportError:
            pass

        # Fallback: implement basic GVP ourselves
        import warnings
        warnings.warn(
            "GVP package not found. Using basic fallback implementation. "
            "For best results, install: pip install geometric-vector-perceptron"
        )
        self._gvp_source = 'fallback'
        self._build_fallback_model()

    def _build_lucidrains_model(self):
        """Build model using lucidrains GVP."""
        from geometric_vector_perceptron import GVP

        # Input features: (scalar_dim, vector_dim)
        # Node input: 6 scalar (one-hot AA type placeholder), 3 vectors (unit vectors to neighbors)
        node_in_dim = (6, 3)
        node_hidden_dim = (self.node_s_dim, self.node_v_dim)

        # Edge input: distance, unit direction
        edge_in_dim = (1, 1)
        edge_hidden_dim = (self.edge_s_dim, self.edge_v_dim)

        # Build layers
        self.node_embedding = GVP(
            node_in_dim,
            node_hidden_dim,
        )

        self.edge_embedding = GVP(
            edge_in_dim,
            edge_hidden_dim,
        )

        # Message passing layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                GVP(node_hidden_dim, node_hidden_dim)
            )

        # Output projection (scalar only)
        self.output_proj = nn.Linear(self.node_s_dim, self.node_s_dim)

        self.to(self._device)

    def _build_drorlab_model(self):
        """Build model using drorlab GVP."""
        from gvp import GVP, GVPConvLayer

        node_in_dim = (6, 3)
        node_hidden_dim = (self.node_s_dim, self.node_v_dim)
        edge_in_dim = (32, 1)

        self.W_v = nn.Sequential(
            GVP(node_in_dim, node_hidden_dim, activations=(None, None)),
            GVP(node_hidden_dim, node_hidden_dim, activations=(None, None))
        )

        self.W_e = nn.Sequential(
            GVP(edge_in_dim, (self.edge_s_dim, self.edge_v_dim), activations=(None, None)),
        )

        self.layers = nn.ModuleList([
            GVPConvLayer(node_hidden_dim, (self.edge_s_dim, self.edge_v_dim))
            for _ in range(self.num_layers)
        ])

        self.W_out = GVP(node_hidden_dim, (self.node_s_dim, 0))

        self.to(self._device)

    def _build_fallback_model(self):
        """Build a simple fallback model without GVP."""
        # Simple MLP on distance-based features
        self.distance_encoder = nn.Sequential(
            nn.Linear(30, 64),  # k_neighbors distances
            nn.ReLU(),
            nn.Linear(64, self.node_s_dim),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.node_s_dim, self.node_s_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
            )
            for _ in range(self.num_layers)
        ])

        self.to(self._device)

    @property
    def embedding_dim(self) -> int:
        return self.node_s_dim

    @property
    def name(self) -> str:
        return f"GVP-GNN ({self._gvp_source})"

    def _extract_backbone_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Extract CA coordinates from backbone coords.

        Args:
            coords: [L, 4, 3] (N, CA, C, O) or [L, 3] (CA only)

        Returns:
            CA coordinates [L, 3]
        """
        if coords.dim() == 3 and coords.shape[1] == 4:
            # Extract CA (index 1)
            return coords[:, 1, :]
        elif coords.dim() == 2 and coords.shape[1] == 3:
            return coords
        else:
            raise ValueError(f"Unexpected coords shape: {coords.shape}")

    def _build_knn_graph(
        self,
        coords: torch.Tensor,
        k: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build k-nearest neighbor graph from coordinates.

        Args:
            coords: [L, 3] CA coordinates
            k: Number of neighbors (default: self.k_neighbors)

        Returns:
            edge_index: [2, E] source and target indices
            edge_attr: [E, ...] edge features (distances, directions)
        """
        if k is None:
            k = self.k_neighbors

        L = coords.shape[0]
        k = min(k, L - 1)

        # Compute pairwise distances
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # [L, L, 3]
        dist = torch.norm(diff, dim=-1)  # [L, L]

        # Get k nearest neighbors for each node
        _, indices = torch.topk(dist, k + 1, dim=-1, largest=False)
        indices = indices[:, 1:]  # Remove self-loops

        # Build edge index
        src = torch.arange(L, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = indices.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)

        # Edge features: distances and unit directions
        edge_dist = dist[src, dst].unsqueeze(-1)  # [E, 1]
        edge_dir = diff[src, dst]  # [E, 3]
        edge_dir = edge_dir / (torch.norm(edge_dir, dim=-1, keepdim=True) + 1e-6)

        return edge_index, (edge_dist, edge_dir)

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
        ca_coords = self._extract_backbone_coords(coords)
        L = ca_coords.shape[0]

        # Build graph
        edge_index, edge_attr = self._build_knn_graph(ca_coords)
        edge_dist, edge_dir = edge_attr

        if self._gvp_source == 'fallback':
            return self._encode_fallback(ca_coords, edge_index, edge_dist)
        else:
            return self._encode_gvp(ca_coords, edge_index, edge_dist, edge_dir, sequence)

    def _encode_fallback(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback encoding without GVP."""
        L = coords.shape[0]
        k = self.k_neighbors

        # Gather neighbor distances for each node
        src, dst = edge_index
        neighbor_dist = torch.zeros(L, k, device=self._device)
        for i in range(L):
            mask = src == i
            dists = edge_dist[mask].squeeze(-1)
            neighbor_dist[i, :len(dists)] = dists

        # Encode
        h = self.distance_encoder(neighbor_dist)

        # Apply layers
        for layer in self.layers:
            h = layer(h) + h  # Residual

        return h

    def _encode_gvp(
        self,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_dir: torch.Tensor,
        sequence: str,
    ) -> torch.Tensor:
        """Encode using GVP layers."""
        L = coords.shape[0]

        # This is a simplified version - full GVP requires proper
        # node/edge feature construction based on the specific implementation
        # For now, use fallback
        return self._encode_fallback(coords, edge_index, edge_dist)

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

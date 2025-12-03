"""
Base class for antibody embedders.

Defines the common interface for all antibody sequence and structure embedders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AntibodyEmbedderOutput:
    """
    Output from an antibody embedder.

    Attributes:
        heavy_residue_embeddings: Per-residue embeddings for heavy chain [L_H, d_model]
        light_residue_embeddings: Per-residue embeddings for light chain [L_L, d_model]
        global_embedding: Pooled embedding for the whole antibody [d_model]
        heavy_attention_weights: Optional attention weights for heavy chain
        light_attention_weights: Optional attention weights for light chain
        heavy_sequence: The input heavy chain sequence
        light_sequence: The input light chain sequence
    """
    heavy_residue_embeddings: torch.Tensor
    light_residue_embeddings: torch.Tensor
    global_embedding: torch.Tensor
    heavy_attention_weights: Optional[torch.Tensor] = None
    light_attention_weights: Optional[torch.Tensor] = None
    heavy_sequence: Optional[str] = None
    light_sequence: Optional[str] = None

    def to(self, device: torch.device) -> 'AntibodyEmbedderOutput':
        """Move all tensors to device."""
        return AntibodyEmbedderOutput(
            heavy_residue_embeddings=self.heavy_residue_embeddings.to(device),
            light_residue_embeddings=self.light_residue_embeddings.to(device),
            global_embedding=self.global_embedding.to(device),
            heavy_attention_weights=self.heavy_attention_weights.to(device) if self.heavy_attention_weights is not None else None,
            light_attention_weights=self.light_attention_weights.to(device) if self.light_attention_weights is not None else None,
            heavy_sequence=self.heavy_sequence,
            light_sequence=self.light_sequence,
        )


@dataclass
class BatchedAntibodyEmbedderOutput:
    """
    Batched output from an antibody embedder.

    Attributes:
        heavy_residue_embeddings: Per-residue embeddings for heavy chains [B, L_H, d_model]
        light_residue_embeddings: Per-residue embeddings for light chains [B, L_L, d_model]
        global_embeddings: Pooled embeddings for antibodies [B, d_model]
        heavy_attention_weights: Optional attention weights [B, L_H, L_H] or [B, heads, L_H, L_H]
        light_attention_weights: Optional attention weights [B, L_L, L_L] or [B, heads, L_L, L_L]
        heavy_sequences: List of heavy chain sequences
        light_sequences: List of light chain sequences
        heavy_mask: Padding mask for heavy chain [B, L_H]
        light_mask: Padding mask for light chain [B, L_L]
    """
    heavy_residue_embeddings: torch.Tensor
    light_residue_embeddings: torch.Tensor
    global_embeddings: torch.Tensor
    heavy_attention_weights: Optional[torch.Tensor] = None
    light_attention_weights: Optional[torch.Tensor] = None
    heavy_sequences: Optional[List[str]] = None
    light_sequences: Optional[List[str]] = None
    heavy_mask: Optional[torch.Tensor] = None
    light_mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'BatchedAntibodyEmbedderOutput':
        """Move all tensors to device."""
        return BatchedAntibodyEmbedderOutput(
            heavy_residue_embeddings=self.heavy_residue_embeddings.to(device),
            light_residue_embeddings=self.light_residue_embeddings.to(device),
            global_embeddings=self.global_embeddings.to(device),
            heavy_attention_weights=self.heavy_attention_weights.to(device) if self.heavy_attention_weights is not None else None,
            light_attention_weights=self.light_attention_weights.to(device) if self.light_attention_weights is not None else None,
            heavy_sequences=self.heavy_sequences,
            light_sequences=self.light_sequences,
            heavy_mask=self.heavy_mask.to(device) if self.heavy_mask is not None else None,
            light_mask=self.light_mask.to(device) if self.light_mask is not None else None,
        )

    def __getitem__(self, idx: int) -> AntibodyEmbedderOutput:
        """Get single sample from batch."""
        # Get valid length from mask if available
        if self.heavy_mask is not None:
            h_len = self.heavy_mask[idx].sum().int().item()
        else:
            h_len = self.heavy_residue_embeddings.shape[1]

        if self.light_mask is not None:
            l_len = self.light_mask[idx].sum().int().item()
        else:
            l_len = self.light_residue_embeddings.shape[1]

        return AntibodyEmbedderOutput(
            heavy_residue_embeddings=self.heavy_residue_embeddings[idx, :h_len],
            light_residue_embeddings=self.light_residue_embeddings[idx, :l_len],
            global_embedding=self.global_embeddings[idx],
            heavy_attention_weights=self.heavy_attention_weights[idx] if self.heavy_attention_weights is not None else None,
            light_attention_weights=self.light_attention_weights[idx] if self.light_attention_weights is not None else None,
            heavy_sequence=self.heavy_sequences[idx] if self.heavy_sequences is not None else None,
            light_sequence=self.light_sequences[idx] if self.light_sequences is not None else None,
        )

    def __len__(self) -> int:
        return self.global_embeddings.shape[0]


class AntibodyEmbedder(nn.Module, ABC):
    """
    Abstract base class for antibody embedders.

    All antibody embedders must implement:
    - encode(): Takes heavy and light chain sequences, returns embeddings
    - embedding_dim: The dimension of the output embeddings
    - name: A unique identifier for the embedder

    Embedders can optionally implement:
    - encode_batch(): Batch encoding for efficiency
    - supports_paired: Whether the model natively supports paired H/L input
    """

    def __init__(
        self,
        trainable: bool = False,
        device: str = 'auto',
        pooling: str = 'mean',
    ):
        """
        Initialize the antibody embedder.

        Args:
            trainable: Whether the embedder parameters should be trainable
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            pooling: Pooling strategy for global embedding ('mean', 'cls', 'max')
        """
        super().__init__()
        self.trainable = trainable
        self._device = self._resolve_device(device)
        self.pooling = pooling

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)

    @property
    def device(self) -> torch.device:
        """Get the device of the embedder."""
        return self._device

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the output embeddings."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique identifier for this embedder."""
        pass

    @property
    def supports_paired(self) -> bool:
        """Whether this embedder natively supports paired H/L input."""
        return True  # Default to True, override in subclasses

    @abstractmethod
    def encode(
        self,
        heavy_sequence: str,
        light_sequence: str,
        return_attention: bool = False,
    ) -> AntibodyEmbedderOutput:
        """
        Encode a single antibody (heavy + light chain).

        Args:
            heavy_sequence: Heavy chain amino acid sequence
            light_sequence: Light chain amino acid sequence
            return_attention: Whether to return attention weights

        Returns:
            AntibodyEmbedderOutput with per-residue and global embeddings
        """
        pass

    def encode_batch(
        self,
        heavy_sequences: List[str],
        light_sequences: List[str],
        return_attention: bool = False,
    ) -> BatchedAntibodyEmbedderOutput:
        """
        Encode a batch of antibodies.

        Default implementation calls encode() in a loop.
        Subclasses should override for efficiency.

        Args:
            heavy_sequences: List of heavy chain sequences
            light_sequences: List of light chain sequences
            return_attention: Whether to return attention weights

        Returns:
            BatchedAntibodyEmbedderOutput with batched embeddings
        """
        outputs = [
            self.encode(h, l, return_attention=return_attention)
            for h, l in zip(heavy_sequences, light_sequences)
        ]

        # Pad and stack
        max_h_len = max(o.heavy_residue_embeddings.shape[0] for o in outputs)
        max_l_len = max(o.light_residue_embeddings.shape[0] for o in outputs)
        batch_size = len(outputs)

        # Initialize padded tensors
        h_emb = torch.zeros(batch_size, max_h_len, self.embedding_dim, device=self.device)
        l_emb = torch.zeros(batch_size, max_l_len, self.embedding_dim, device=self.device)
        g_emb = torch.zeros(batch_size, self.embedding_dim, device=self.device)
        h_mask = torch.zeros(batch_size, max_h_len, dtype=torch.bool, device=self.device)
        l_mask = torch.zeros(batch_size, max_l_len, dtype=torch.bool, device=self.device)

        for i, o in enumerate(outputs):
            h_len = o.heavy_residue_embeddings.shape[0]
            l_len = o.light_residue_embeddings.shape[0]

            h_emb[i, :h_len] = o.heavy_residue_embeddings
            l_emb[i, :l_len] = o.light_residue_embeddings
            g_emb[i] = o.global_embedding
            h_mask[i, :h_len] = True
            l_mask[i, :l_len] = True

        return BatchedAntibodyEmbedderOutput(
            heavy_residue_embeddings=h_emb,
            light_residue_embeddings=l_emb,
            global_embeddings=g_emb,
            heavy_sequences=heavy_sequences,
            light_sequences=light_sequences,
            heavy_mask=h_mask,
            light_mask=l_mask,
        )

    def _pool_embeddings(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool per-residue embeddings to a single vector.

        Args:
            embeddings: [seq_len, d_model] or [batch, seq_len, d_model]
            mask: Optional mask for valid positions

        Returns:
            Pooled embedding [d_model] or [batch, d_model]
        """
        if self.pooling == 'mean':
            if mask is not None:
                # Masked mean
                if embeddings.dim() == 2:
                    return (embeddings * mask.unsqueeze(-1)).sum(0) / mask.sum()
                else:
                    return (embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            else:
                return embeddings.mean(dim=-2)

        elif self.pooling == 'cls':
            # Use first token (CLS)
            if embeddings.dim() == 2:
                return embeddings[0]
            else:
                return embeddings[:, 0]

        elif self.pooling == 'max':
            if mask is not None:
                # Masked max
                embeddings = embeddings.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            return embeddings.max(dim=-2)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.trainable = False

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self.trainable = True


class StructuralEncoder(nn.Module, ABC):
    """
    Abstract base class for structural encoders.

    Structural encoders take 3D coordinates and produce per-residue embeddings.
    They are used in addition to sequence embedders for structure-aware predictions.
    """

    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 32,
        num_layers: int = 3,
        device: str = 'auto',
    ):
        """
        Initialize the structural encoder.

        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            num_layers: Number of message passing layers
            device: Device to use
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self._device = self._resolve_device(device)

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of the output embeddings."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a unique identifier for this encoder."""
        pass

    @abstractmethod
    def encode(
        self,
        coords: torch.Tensor,
        sequence: str,
        chain_ids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Encode a protein structure.

        Args:
            coords: Backbone coordinates [L, 4, 3] (N, CA, C, O) or [L, 3] (CA only)
            sequence: Amino acid sequence of length L
            chain_ids: Optional chain identifiers for each residue

        Returns:
            Per-residue structural embeddings [L, embedding_dim]
        """
        pass

    @abstractmethod
    def encode_batch(
        self,
        coords_list: List[torch.Tensor],
        sequences: List[str],
        chain_ids_list: Optional[List[List[str]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of protein structures.

        Args:
            coords_list: List of coordinate tensors
            sequences: List of sequences
            chain_ids_list: Optional list of chain ID lists

        Returns:
            Tuple of (embeddings [B, L, d], mask [B, L])
        """
        pass

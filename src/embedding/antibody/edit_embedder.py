"""
Simple Antibody Edit Embedder.

Produces edit embeddings by computing the difference between
mutant and wild-type antibody embeddings.

This is analogous to the simple edit embedding approach used in
small molecules (emb_B - emb_A) and RNA.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass

from .base import AntibodyEmbedder, AntibodyEmbedderOutput


@dataclass
class AntibodyEditOutput:
    """
    Output from the antibody edit embedder.

    Attributes:
        edit_embedding: The edit embedding vector [d_edit]
        context_embedding: Global context embedding [d_context]
        wt_output: Wild-type antibody embedding output
        mut_output: Mutant antibody embedding output
    """
    edit_embedding: torch.Tensor
    context_embedding: torch.Tensor
    wt_output: Optional[AntibodyEmbedderOutput] = None
    mut_output: Optional[AntibodyEmbedderOutput] = None


@dataclass
class BatchedAntibodyEditOutput:
    """
    Batched output from the antibody edit embedder.

    Attributes:
        edit_embeddings: Edit embedding vectors [B, d_edit]
        context_embeddings: Global context embeddings [B, d_context]
    """
    edit_embeddings: torch.Tensor
    context_embeddings: torch.Tensor


class AntibodyEditEmbedder(nn.Module):
    """
    Simple antibody edit embedder using embedding differences.

    Computes:
        edit_embedding = f(emb_mut - emb_wt)

    where emb_mut and emb_wt are global embeddings from a base embedder.

    Args:
        base_embedder: AntibodyEmbedder for encoding sequences
        edit_mlp_dims: Hidden dimensions for edit MLP (default: [512, 256])
        output_dim: Output edit embedding dimension (default: 320)
        dropout: Dropout probability (default: 0.1)
        use_concat: If True, concatenate wt and mut instead of diff (default: False)

    Example:
        >>> from src.embedding.antibody import IgT5Embedder, AntibodyEditEmbedder
        >>> base_embedder = IgT5Embedder()
        >>> edit_embedder = AntibodyEditEmbedder(base_embedder)
        >>> output = edit_embedder.encode(
        ...     wt_heavy="EVQL...", wt_light="DIVMT...",
        ...     mut_heavy="EVKL...", mut_light="DIVMT..."  # E->K mutation
        ... )
    """

    def __init__(
        self,
        base_embedder: AntibodyEmbedder,
        edit_mlp_dims: List[int] = None,
        output_dim: int = 320,
        dropout: float = 0.1,
        use_concat: bool = False,
    ):
        super().__init__()

        self.base_embedder = base_embedder
        self.output_dim = output_dim
        self.use_concat = use_concat

        if edit_mlp_dims is None:
            edit_mlp_dims = [512, 256]

        # Input dimension depends on whether we concat or diff
        if use_concat:
            input_dim = base_embedder.embedding_dim * 2
        else:
            input_dim = base_embedder.embedding_dim

        # Build edit MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in edit_mlp_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.LayerNorm(output_dim),
        ])

        self.edit_mlp = nn.Sequential(*layers)

        # Move edit MLP to the same device as base embedder
        if hasattr(base_embedder, 'device'):
            self.edit_mlp = self.edit_mlp.to(base_embedder.device)

    @property
    def embedding_dim(self) -> int:
        """Output embedding dimension."""
        return self.output_dim

    @property
    def context_dim(self) -> int:
        """Context embedding dimension (from base embedder)."""
        return self.base_embedder.embedding_dim

    def encode(
        self,
        wt_heavy: str,
        wt_light: str,
        mut_heavy: str,
        mut_light: str,
        return_base_outputs: bool = False,
    ) -> AntibodyEditOutput:
        """
        Encode a single mutation.

        Args:
            wt_heavy: Wild-type heavy chain sequence
            wt_light: Wild-type light chain sequence
            mut_heavy: Mutant heavy chain sequence
            mut_light: Mutant light chain sequence
            return_base_outputs: Whether to return base embedder outputs

        Returns:
            AntibodyEditOutput with edit and context embeddings
        """
        # Encode wild-type and mutant
        wt_output = self.base_embedder.encode(wt_heavy, wt_light)
        mut_output = self.base_embedder.encode(mut_heavy, mut_light)

        # Compute edit representation
        if self.use_concat:
            edit_input = torch.cat([wt_output.global_embedding, mut_output.global_embedding], dim=-1)
        else:
            edit_input = mut_output.global_embedding - wt_output.global_embedding

        # Apply edit MLP
        edit_embedding = self.edit_mlp(edit_input.unsqueeze(0)).squeeze(0)

        # Context is the wild-type global embedding
        context_embedding = wt_output.global_embedding

        return AntibodyEditOutput(
            edit_embedding=edit_embedding,
            context_embedding=context_embedding,
            wt_output=wt_output if return_base_outputs else None,
            mut_output=mut_output if return_base_outputs else None,
        )

    def encode_batch(
        self,
        wt_heavy_sequences: List[str],
        wt_light_sequences: List[str],
        mut_heavy_sequences: List[str],
        mut_light_sequences: List[str],
    ) -> BatchedAntibodyEditOutput:
        """
        Encode a batch of mutations.

        Args:
            wt_heavy_sequences: Wild-type heavy chain sequences
            wt_light_sequences: Wild-type light chain sequences
            mut_heavy_sequences: Mutant heavy chain sequences
            mut_light_sequences: Mutant light chain sequences

        Returns:
            BatchedAntibodyEditOutput with edit and context embeddings
        """
        # Encode all sequences
        wt_output = self.base_embedder.encode_batch(wt_heavy_sequences, wt_light_sequences)
        mut_output = self.base_embedder.encode_batch(mut_heavy_sequences, mut_light_sequences)

        # Compute edit representations
        if self.use_concat:
            edit_input = torch.cat([
                wt_output.global_embeddings,
                mut_output.global_embeddings
            ], dim=-1)
        else:
            edit_input = mut_output.global_embeddings - wt_output.global_embeddings

        # Apply edit MLP
        edit_embeddings = self.edit_mlp(edit_input)

        return BatchedAntibodyEditOutput(
            edit_embeddings=edit_embeddings,
            context_embeddings=wt_output.global_embeddings,
        )

    def encode_from_mutations(
        self,
        wt_heavy: str,
        wt_light: str,
        mutations: List[Tuple[str, int, str, str]],
    ) -> AntibodyEditOutput:
        """
        Encode mutations specified as a list of (chain, position, from_aa, to_aa).

        Args:
            wt_heavy: Wild-type heavy chain sequence
            wt_light: Wild-type light chain sequence
            mutations: List of (chain, position, from_aa, to_aa) tuples
                      chain: 'H' for heavy, 'L' for light
                      position: 0-indexed position in sequence
                      from_aa: Original amino acid (for validation)
                      to_aa: Mutated amino acid

        Returns:
            AntibodyEditOutput
        """
        # Apply mutations to create mutant sequences
        mut_heavy = list(wt_heavy)
        mut_light = list(wt_light)

        for chain, pos, from_aa, to_aa in mutations:
            if chain.upper() == 'H':
                if pos < len(mut_heavy):
                    assert mut_heavy[pos].upper() == from_aa.upper(), \
                        f"Expected {from_aa} at H{pos}, got {mut_heavy[pos]}"
                    mut_heavy[pos] = to_aa
            else:
                if pos < len(mut_light):
                    assert mut_light[pos].upper() == from_aa.upper(), \
                        f"Expected {from_aa} at L{pos}, got {mut_light[pos]}"
                    mut_light[pos] = to_aa

        mut_heavy = ''.join(mut_heavy)
        mut_light = ''.join(mut_light)

        return self.encode(wt_heavy, wt_light, mut_heavy, mut_light)

    def forward(
        self,
        wt_heavy: Union[str, List[str]],
        wt_light: Union[str, List[str]],
        mut_heavy: Union[str, List[str]],
        mut_light: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            wt_heavy: Wild-type heavy chain(s)
            wt_light: Wild-type light chain(s)
            mut_heavy: Mutant heavy chain(s)
            mut_light: Mutant light chain(s)

        Returns:
            Tuple of (edit_embeddings, context_embeddings)
        """
        if isinstance(wt_heavy, str):
            output = self.encode(wt_heavy, wt_light, mut_heavy, mut_light)
            return output.edit_embedding.unsqueeze(0), output.context_embedding.unsqueeze(0)
        else:
            output = self.encode_batch(wt_heavy, wt_light, mut_heavy, mut_light)
            return output.edit_embeddings, output.context_embeddings

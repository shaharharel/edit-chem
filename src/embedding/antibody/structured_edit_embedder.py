"""
Structured Antibody Edit Embedder.

A sophisticated edit embedding module that captures:
1. Identity features (BLOSUM, hydropathy, charge, volume)
2. Location features (chain, IMGT position, CDR/FR region)
3. Sequence LM context (Δ at mutation site, window context)
4. Structure context (optional, from structural encoders)
5. Multi-mutation aggregation (mean or self-attention)

This produces a rich edit embedding z_edit of dimension >= 300 that describes
the mutation(s) in the context of the antibody sequence and structure.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
from dataclasses import dataclass

from .base import AntibodyEmbedder, AntibodyEmbedderOutput, StructuralEncoder
from .utils.amino_acid_features import (
    get_mutation_feature_vector,
    AMINO_ACIDS,
    AA_TO_IDX,
)
from .utils.numbering import (
    number_antibody,
    get_region_one_hot,
    IMGTPosition,
    CDRRegion,
)


@dataclass
class Mutation:
    """Represents a single mutation."""
    chain: str  # 'H' or 'L'
    position: int  # 0-indexed position in sequence
    from_aa: str
    to_aa: str
    imgt_position: Optional[IMGTPosition] = None


@dataclass
class StructuredEditOutput:
    """
    Output from the structured antibody edit embedder.

    Attributes:
        z_edit: Final edit embedding [d_edit]
        h_context: Context embedding (Ab global) [d_context]
        per_mutation_embeddings: Individual mutation embeddings [n_mutations, d_edit]
        mutations: List of Mutation objects
    """
    z_edit: torch.Tensor
    h_context: torch.Tensor
    per_mutation_embeddings: Optional[torch.Tensor] = None
    mutations: Optional[List[Mutation]] = None


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for IMGT positions."""

    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions.clamp(0, self.pe.shape[0] - 1)]


class MutationAggregator(nn.Module):
    """
    Aggregates multiple per-mutation embeddings into a single edit embedding.

    Supports:
    - mean: Simple mean pooling
    - attention: Self-attention over mutations (captures epistasis)
    """

    def __init__(
        self,
        embed_dim: int,
        aggregation: str = 'mean',
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.embed_dim = embed_dim

        if aggregation == 'attention':
            self.self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(embed_dim)
            self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        mutation_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate mutation embeddings.

        Args:
            mutation_embeddings: [batch, n_mutations, embed_dim] or [n_mutations, embed_dim]
            mask: Optional mask [batch, n_mutations] or [n_mutations]

        Returns:
            Aggregated embedding [batch, embed_dim] or [embed_dim]
        """
        squeeze = mutation_embeddings.dim() == 2
        if squeeze:
            mutation_embeddings = mutation_embeddings.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        if self.aggregation == 'mean':
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(-1).float()
                z = (mutation_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            else:
                z = mutation_embeddings.mean(dim=1)

        elif self.aggregation == 'attention':
            # Self-attention over mutations
            attn_mask = ~mask if mask is not None else None
            attended, _ = self.self_attn(
                mutation_embeddings,
                mutation_embeddings,
                mutation_embeddings,
                key_padding_mask=attn_mask,
            )
            attended = self.norm(mutation_embeddings + attended)

            # Pool attended representations
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                z = (attended * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            else:
                z = attended.mean(dim=1)

            z = self.output_proj(z)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        if squeeze:
            z = z.squeeze(0)

        return z


class StructuredAntibodyEditEmbedder(nn.Module):
    """
    Structured Antibody Edit Embedder.

    Generates rich edit embeddings from:
    - Wild-type sequences (heavy + light)
    - List of mutations (chain, position, from_aa, to_aa)
    - Optional: 3D structure

    Architecture:
    1. Identity features: BLOSUM, hydropathy, charge, volume differences
    2. Location features: chain embedding, IMGT position, CDR/FR region
    3. Sequence LM context:
       - Δ_h: embedding difference at mutation site
       - Δ_w: local window context difference
    4. Structure features (optional): from structural encoder
    5. Global context: antibody global embedding
    6. Per-mutation fusion MLP: combines all features per mutation
    7. Multi-mutation aggregation: combines per-mutation embeddings

    Args:
        base_embedder: AntibodyEmbedder for encoding sequences
        structural_encoder: Optional StructuralEncoder for 3D features
        identity_dim: Dimension for identity feature projection (default: 32)
        location_dim: Dimension for location features (default: 64)
        context_dim: Dimension for sequence context features (default: 128)
        structure_dim: Dimension for structure features (default: 64)
        fusion_hidden_dims: Hidden dims for fusion MLP (default: [512, 384])
        output_dim: Final edit embedding dimension (default: 320)
        window_size: Window size for local context (default: 5)
        dropout: Dropout probability (default: 0.1)
        aggregation: Multi-mutation aggregation ('mean' or 'attention')

    Example:
        >>> embedder = StructuredAntibodyEditEmbedder(IgT5Embedder())
        >>> mutations = [Mutation('H', 50, 'E', 'K')]
        >>> output = embedder.encode(wt_heavy, wt_light, mutations)
    """

    def __init__(
        self,
        base_embedder: AntibodyEmbedder,
        structural_encoder: Optional[StructuralEncoder] = None,
        identity_dim: int = 32,
        location_dim: int = 64,
        context_dim: int = 128,
        structure_dim: int = 64,
        fusion_hidden_dims: List[int] = None,
        output_dim: int = 320,
        window_size: int = 5,
        dropout: float = 0.1,
        aggregation: str = 'mean',
    ):
        super().__init__()

        self.base_embedder = base_embedder
        self.structural_encoder = structural_encoder
        self.window_size = window_size
        self.output_dim = output_dim
        self.use_structure = structural_encoder is not None

        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 384]

        base_dim = base_embedder.embedding_dim

        # ========================================
        # 1. Identity Features
        # ========================================
        # From AA embeddings (learnable)
        self.aa_embedding = nn.Embedding(20, 32)

        # From physicochemical features (5 features: BLOSUM, hydro, charge, vol, mass)
        self.identity_proj = nn.Sequential(
            nn.Linear(5 + 64, identity_dim),  # 5 features + 2*32 AA embeddings
            nn.LayerNorm(identity_dim),
            nn.GELU(),
        )

        # ========================================
        # 2. Location Features
        # ========================================
        # Chain embedding (H or L)
        self.chain_embedding = nn.Embedding(2, 16)

        # IMGT position encoding
        self.imgt_pos_encoding = SinusoidalPositionalEncoding(32, max_len=150)
        self.learned_pos_embedding = nn.Embedding(150, 16)

        # Region embedding (7 regions: FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4)
        self.region_embedding = nn.Embedding(7, 16)

        # Relative position projection
        self.relative_pos_proj = nn.Linear(1, 8)

        # Location fusion
        location_raw_dim = 16 + 32 + 16 + 16 + 8  # chain + sinusoidal + learned + region + relative
        self.location_proj = nn.Sequential(
            nn.Linear(location_raw_dim, location_dim),
            nn.LayerNorm(location_dim),
            nn.GELU(),
        )

        # ========================================
        # 3. Sequence LM Context
        # ========================================
        # Δ at mutation site
        self.delta_site_proj = nn.Sequential(
            nn.Linear(base_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )

        # Δ window context
        self.delta_window_proj = nn.Sequential(
            nn.Linear(base_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )

        # ========================================
        # 4. Structure Features (Optional)
        # ========================================
        if self.use_structure:
            struct_dim = structural_encoder.embedding_dim
            self.structure_site_proj = nn.Sequential(
                nn.Linear(struct_dim, structure_dim),
                nn.LayerNorm(structure_dim),
                nn.GELU(),
            )
            self.structure_window_proj = nn.Sequential(
                nn.Linear(struct_dim, structure_dim),
                nn.LayerNorm(structure_dim),
                nn.GELU(),
            )
            self.structure_dim = structure_dim
        else:
            self.structure_dim = 0

        # ========================================
        # 5. Global Context
        # ========================================
        self.global_proj = nn.Sequential(
            nn.Linear(base_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )

        # ========================================
        # 6. Per-Mutation Fusion MLP
        # ========================================
        per_mutation_dim = (
            identity_dim +
            location_dim +
            context_dim * 2 +  # delta_site + delta_window
            self.structure_dim * 2 +  # structure_site + structure_window (if used)
            context_dim  # global context
        )

        fusion_layers = []
        prev_dim = per_mutation_dim

        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        fusion_layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.LayerNorm(output_dim),
        ])

        self.fusion_mlp = nn.Sequential(*fusion_layers)

        # ========================================
        # 7. Multi-Mutation Aggregation
        # ========================================
        self.aggregator = MutationAggregator(
            embed_dim=output_dim,
            aggregation=aggregation,
        )

        # Store component dimensions for debugging
        self._component_dims = {
            'identity': identity_dim,
            'location': location_dim,
            'context_site': context_dim,
            'context_window': context_dim,
            'structure_site': self.structure_dim,
            'structure_window': self.structure_dim,
            'global': context_dim,
            'per_mutation': per_mutation_dim,
            'output': output_dim,
        }

        # Move all modules to the same device as base embedder
        if hasattr(base_embedder, 'device'):
            device = base_embedder.device
            self.aa_embedding = self.aa_embedding.to(device)
            self.identity_proj = self.identity_proj.to(device)
            self.chain_embedding = self.chain_embedding.to(device)
            self.region_embedding = self.region_embedding.to(device)
            self.imgt_pos_encoding = self.imgt_pos_encoding.to(device)
            self.learned_pos_embedding = self.learned_pos_embedding.to(device)
            self.relative_pos_proj = self.relative_pos_proj.to(device)
            self.location_proj = self.location_proj.to(device)
            self.delta_site_proj = self.delta_site_proj.to(device)
            self.delta_window_proj = self.delta_window_proj.to(device)
            self.global_proj = self.global_proj.to(device)
            self.fusion_mlp = self.fusion_mlp.to(device)
            self.aggregator = self.aggregator.to(device)
            if self.use_structure:
                self.structure_site_proj = self.structure_site_proj.to(device)
                self.structure_window_proj = self.structure_window_proj.to(device)

    @property
    def embedding_dim(self) -> int:
        return self.output_dim

    @property
    def context_dim(self) -> int:
        return self.base_embedder.embedding_dim

    def _get_wt_embeddings(
        self,
        wt_heavy: str,
        wt_light: str,
    ) -> Tuple[AntibodyEmbedderOutput, List[IMGTPosition], List[IMGTPosition]]:
        """Get wild-type embeddings and numbering."""
        wt_output = self.base_embedder.encode(wt_heavy, wt_light, return_attention=False)

        # Number sequences
        heavy_numbering = number_antibody(wt_heavy, chain_type='H')
        light_numbering = number_antibody(wt_light, chain_type='L')

        return wt_output, heavy_numbering, light_numbering

    def _compute_identity_features(
        self,
        mutations: List[Mutation],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute identity features for each mutation."""
        features = []

        for mut in mutations:
            # Physicochemical features
            phys_feat = get_mutation_feature_vector(
                mut.from_aa, mut.to_aa, normalize=True, device=device
            )

            # AA embeddings
            from_idx = AA_TO_IDX.get(mut.from_aa.upper(), 0)
            to_idx = AA_TO_IDX.get(mut.to_aa.upper(), 0)
            from_emb = self.aa_embedding(torch.tensor(from_idx, device=device))
            to_emb = self.aa_embedding(torch.tensor(to_idx, device=device))

            # Combine
            combined = torch.cat([phys_feat, from_emb, to_emb])
            features.append(combined)

        features = torch.stack(features)
        return self.identity_proj(features)

    def _compute_location_features(
        self,
        mutations: List[Mutation],
        heavy_numbering: List[IMGTPosition],
        light_numbering: List[IMGTPosition],
        heavy_len: int,
        light_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute location features for each mutation."""
        features = []

        for mut in mutations:
            # Chain embedding
            chain_idx = 0 if mut.chain.upper() == 'H' else 1
            chain_emb = self.chain_embedding(torch.tensor(chain_idx, device=device))

            # Get IMGT position
            numbering = heavy_numbering if mut.chain.upper() == 'H' else light_numbering
            seq_len = heavy_len if mut.chain.upper() == 'H' else light_len

            imgt_pos = None
            for pos in numbering:
                if pos.sequence_position == mut.position:
                    imgt_pos = pos
                    break

            if imgt_pos is not None:
                imgt_num = imgt_pos.position
                region_idx = [
                    CDRRegion.FR1, CDRRegion.CDR1, CDRRegion.FR2, CDRRegion.CDR2,
                    CDRRegion.FR3, CDRRegion.CDR3, CDRRegion.FR4
                ].index(imgt_pos.region) if imgt_pos.region else 0
            else:
                imgt_num = mut.position + 1
                region_idx = 0

            # Sinusoidal position encoding
            sin_pos = self.imgt_pos_encoding(torch.tensor(imgt_num, device=device))

            # Learned position embedding
            learned_pos = self.learned_pos_embedding(
                torch.tensor(min(imgt_num, 149), device=device)
            )

            # Region embedding
            region_emb = self.region_embedding(torch.tensor(region_idx, device=device))

            # Relative position (position / sequence_length)
            rel_pos = torch.tensor([mut.position / seq_len], device=device)
            rel_pos_emb = self.relative_pos_proj(rel_pos)

            # Combine
            combined = torch.cat([chain_emb, sin_pos, learned_pos, region_emb, rel_pos_emb])
            features.append(combined)

        features = torch.stack(features)
        return self.location_proj(features)

    def _compute_context_features(
        self,
        mutations: List[Mutation],
        wt_output: AntibodyEmbedderOutput,
        mut_output: AntibodyEmbedderOutput,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute sequence LM context features.

        Returns:
            delta_site: Embedding difference at mutation sites
            delta_window: Local window context differences
            global_context: Global antibody context
        """
        delta_sites = []
        delta_windows = []

        for mut in mutations:
            # Get residue embeddings for this chain
            if mut.chain.upper() == 'H':
                wt_res = wt_output.heavy_residue_embeddings
                mut_res = mut_output.heavy_residue_embeddings
            else:
                wt_res = wt_output.light_residue_embeddings
                mut_res = mut_output.light_residue_embeddings

            pos = mut.position
            seq_len = wt_res.shape[0]

            # Δ at mutation site
            if pos < seq_len:
                delta_site = mut_res[pos] - wt_res[pos]
            else:
                delta_site = torch.zeros(wt_res.shape[-1], device=device)

            # Δ for local window
            start = max(0, pos - self.window_size)
            end = min(seq_len, pos + self.window_size + 1)

            if start < end:
                wt_window = wt_res[start:end].mean(dim=0)
                mut_window = mut_res[start:end].mean(dim=0)
                delta_window = mut_window - wt_window
            else:
                delta_window = torch.zeros(wt_res.shape[-1], device=device)

            delta_sites.append(delta_site)
            delta_windows.append(delta_window)

        delta_sites = torch.stack(delta_sites)
        delta_windows = torch.stack(delta_windows)

        # Project
        delta_site_proj = self.delta_site_proj(delta_sites)
        delta_window_proj = self.delta_window_proj(delta_windows)

        # Global context (same for all mutations)
        global_context = self.global_proj(wt_output.global_embedding)
        global_context = global_context.unsqueeze(0).expand(len(mutations), -1)

        return delta_site_proj, delta_window_proj, global_context

    def encode(
        self,
        wt_heavy: str,
        wt_light: str,
        mutations: List[Mutation],
        wt_coords: Optional[torch.Tensor] = None,
    ) -> StructuredEditOutput:
        """
        Encode mutations into a structured edit embedding.

        Args:
            wt_heavy: Wild-type heavy chain sequence
            wt_light: Wild-type light chain sequence
            mutations: List of Mutation objects
            wt_coords: Optional wild-type structure coordinates

        Returns:
            StructuredEditOutput with z_edit and h_context
        """
        if len(mutations) == 0:
            raise ValueError("At least one mutation required")

        device = next(self.parameters()).device

        # Get wild-type embeddings
        wt_output, heavy_numbering, light_numbering = self._get_wt_embeddings(
            wt_heavy, wt_light
        )

        # Create mutant sequences
        mut_heavy = list(wt_heavy)
        mut_light = list(wt_light)

        for mut in mutations:
            if mut.chain.upper() == 'H' and mut.position < len(mut_heavy):
                mut_heavy[mut.position] = mut.to_aa
            elif mut.chain.upper() == 'L' and mut.position < len(mut_light):
                mut_light[mut.position] = mut.to_aa

        mut_heavy = ''.join(mut_heavy)
        mut_light = ''.join(mut_light)

        # Get mutant embeddings
        mut_output = self.base_embedder.encode(mut_heavy, mut_light)

        # Compute features for each mutation
        # 1. Identity features
        identity_feat = self._compute_identity_features(mutations, device)

        # 2. Location features
        location_feat = self._compute_location_features(
            mutations, heavy_numbering, light_numbering,
            len(wt_heavy), len(wt_light), device
        )

        # 3. Sequence context features
        delta_site, delta_window, global_context = self._compute_context_features(
            mutations, wt_output, mut_output, device
        )

        # 4. Structure features (if available)
        if self.use_structure and wt_coords is not None:
            struct_site, struct_window = self._compute_structure_features(
                mutations, wt_coords, wt_heavy + wt_light, device
            )
        elif self.use_structure:
            n_mut = len(mutations)
            struct_site = torch.zeros(n_mut, self.structure_dim, device=device)
            struct_window = torch.zeros(n_mut, self.structure_dim, device=device)
        else:
            struct_site = None
            struct_window = None

        # 5. Concatenate all features per mutation
        per_mutation_features = [identity_feat, location_feat, delta_site, delta_window]
        if self.use_structure:
            per_mutation_features.extend([struct_site, struct_window])
        per_mutation_features.append(global_context)

        per_mutation = torch.cat(per_mutation_features, dim=-1)

        # 6. Apply fusion MLP
        per_mutation_emb = self.fusion_mlp(per_mutation)

        # 7. Aggregate mutations
        z_edit = self.aggregator(per_mutation_emb)

        # Context embedding
        h_context = wt_output.global_embedding

        return StructuredEditOutput(
            z_edit=z_edit,
            h_context=h_context,
            per_mutation_embeddings=per_mutation_emb,
            mutations=mutations,
        )

    def _compute_structure_features(
        self,
        mutations: List[Mutation],
        coords: torch.Tensor,
        sequence: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute structure features using structural encoder."""
        # Encode structure
        struct_emb = self.structural_encoder.encode(coords, sequence)

        struct_sites = []
        struct_windows = []

        for mut in mutations:
            pos = mut.position
            seq_len = struct_emb.shape[0]

            # Site feature
            if pos < seq_len:
                site = struct_emb[pos]
            else:
                site = torch.zeros(struct_emb.shape[-1], device=device)

            # Window feature
            start = max(0, pos - self.window_size)
            end = min(seq_len, pos + self.window_size + 1)
            if start < end:
                window = struct_emb[start:end].mean(dim=0)
            else:
                window = torch.zeros(struct_emb.shape[-1], device=device)

            struct_sites.append(site)
            struct_windows.append(window)

        struct_sites = torch.stack(struct_sites)
        struct_windows = torch.stack(struct_windows)

        return self.structure_site_proj(struct_sites), self.structure_window_proj(struct_windows)

    def _detect_mutations(
        self,
        wt_heavy: str,
        wt_light: str,
        mut_heavy: str,
        mut_light: str,
    ) -> List[Mutation]:
        """
        Detect mutations by comparing wild-type and mutant sequences.

        Args:
            wt_heavy: Wild-type heavy chain sequence
            wt_light: Wild-type light chain sequence
            mut_heavy: Mutant heavy chain sequence
            mut_light: Mutant light chain sequence

        Returns:
            List of detected Mutation objects
        """
        mutations = []

        # Detect heavy chain mutations
        min_len = min(len(wt_heavy), len(mut_heavy))
        for i in range(min_len):
            if wt_heavy[i] != mut_heavy[i]:
                mutations.append(Mutation(
                    chain='H',
                    position=i,
                    from_aa=wt_heavy[i],
                    to_aa=mut_heavy[i],
                ))

        # Detect light chain mutations
        min_len = min(len(wt_light), len(mut_light))
        for i in range(min_len):
            if wt_light[i] != mut_light[i]:
                mutations.append(Mutation(
                    chain='L',
                    position=i,
                    from_aa=wt_light[i],
                    to_aa=mut_light[i],
                ))

        return mutations

    def encode_from_sequences(
        self,
        wt_heavy: str,
        wt_light: str,
        mut_heavy: str,
        mut_light: str,
        wt_coords: Optional[torch.Tensor] = None,
    ) -> StructuredEditOutput:
        """
        Encode mutations by comparing wild-type and mutant sequences.

        This is a convenience method that detects mutations automatically
        by comparing the sequences, then calls encode().

        Args:
            wt_heavy: Wild-type heavy chain sequence
            wt_light: Wild-type light chain sequence
            mut_heavy: Mutant heavy chain sequence
            mut_light: Mutant light chain sequence
            wt_coords: Optional wild-type structure coordinates

        Returns:
            StructuredEditOutput with edit and context embeddings
        """
        # Detect mutations
        mutations = self._detect_mutations(wt_heavy, wt_light, mut_heavy, mut_light)

        # If no mutations detected, return zero edit embedding
        if not mutations:
            device = next(self.parameters()).device
            return StructuredEditOutput(
                z_edit=torch.zeros(self.output_dim, device=device),
                h_context=self.base_embedder.encode(wt_heavy, wt_light).global_embedding,
                per_mutation_embeddings=None,
                mutations=[],
            )

        return self.encode(wt_heavy, wt_light, mutations, wt_coords)

    def forward(
        self,
        wt_heavy: str,
        wt_light: str,
        mutations: List[Mutation],
        wt_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Returns:
            Tuple of (z_edit, h_context)
        """
        output = self.encode(wt_heavy, wt_light, mutations, wt_coords)
        return output.z_edit, output.h_context


def create_structured_embedder(
    embedder_type: str = 'igt5',
    structural_encoder_type: Optional[str] = None,
    output_dim: int = 320,
    aggregation: str = 'mean',
    device: str = 'auto',
    **kwargs,
) -> StructuredAntibodyEditEmbedder:
    """
    Factory function to create a StructuredAntibodyEditEmbedder.

    Args:
        embedder_type: Base embedder ('igt5', 'igbert', 'antiberta2', 'ablang2', 'balm', 'balm_paired')
        structural_encoder_type: Optional structural encoder ('gvp', 'se3', 'equiformer')
        output_dim: Output edit embedding dimension
        aggregation: Multi-mutation aggregation ('mean' or 'attention')
        device: Device to use
        **kwargs: Additional arguments for StructuredAntibodyEditEmbedder

    Returns:
        Configured StructuredAntibodyEditEmbedder
    """
    # Import embedders
    from . import (
        IgT5Embedder, IgBertEmbedder, AntiBERTa2Embedder,
        AbLang2Embedder, BALMEmbedder, BALMPairedEmbedder,
    )

    # Create base embedder
    embedder_map = {
        'igt5': IgT5Embedder,
        'igbert': IgBertEmbedder,
        'antiberta2': AntiBERTa2Embedder,
        'ablang2': AbLang2Embedder,
        'balm': BALMEmbedder,
        'balm_paired': BALMPairedEmbedder,
    }

    if embedder_type.lower() not in embedder_map:
        raise ValueError(f"Unknown embedder: {embedder_type}")

    EmbedderClass = embedder_map[embedder_type.lower()]
    if EmbedderClass is None:
        raise ImportError(f"Embedder {embedder_type} not available")

    base_embedder = EmbedderClass(device=device)

    # Create structural encoder if specified
    structural_encoder = None
    if structural_encoder_type:
        from .structural import GVPEncoder, SE3TransformerEncoder, EquiformerEncoder

        encoder_map = {
            'gvp': GVPEncoder,
            'se3': SE3TransformerEncoder,
            'equiformer': EquiformerEncoder,
        }

        if structural_encoder_type.lower() not in encoder_map:
            raise ValueError(f"Unknown structural encoder: {structural_encoder_type}")

        EncoderClass = encoder_map[structural_encoder_type.lower()]
        if EncoderClass is None:
            raise ImportError(f"Structural encoder {structural_encoder_type} not available")

        structural_encoder = EncoderClass(device=device)

    return StructuredAntibodyEditEmbedder(
        base_embedder=base_embedder,
        structural_encoder=structural_encoder,
        output_dim=output_dim,
        aggregation=aggregation,
        **kwargs,
    )

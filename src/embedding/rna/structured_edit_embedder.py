"""
Structured RNA Edit Embedder.

A sophisticated edit embedding module that captures:
1. Mutation type (12 SNV types)
2. Mutation effect (Δ token embedding from pretrained LM)
3. Position encoding (sinusoidal + learned)
4. Local context (window ±10nt around edit)
5. Attention context (attention-weighted token sum)

This embedder takes sequence A and edit information (position, from, to)
and produces a rich edit representation WITHOUT needing sequence B.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional, Tuple, Dict


# Mutation type mapping: 12 possible SNV transitions
MUTATION_TYPES = {
    ('A', 'C'): 0, ('A', 'G'): 1, ('A', 'U'): 2,
    ('C', 'A'): 3, ('C', 'G'): 4, ('C', 'U'): 5,
    ('G', 'A'): 6, ('G', 'C'): 7, ('G', 'U'): 8,
    ('U', 'A'): 9, ('U', 'C'): 10, ('U', 'G'): 11,
}

# Nucleotide to index for token embeddings
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as in Transformer.
    Encodes both absolute position and relative position (pos/seq_len).
    """

    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        self.dim = dim

        # Create sinusoidal encoding matrix
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [batch] tensor of integer positions

        Returns:
            [batch, dim] positional encodings
        """
        return self.pe[positions]


class StructuredRNAEditEmbedder(nn.Module):
    """
    Structured RNA Edit Embedder.

    Generates rich edit embeddings from:
    - Sequence A (the original sequence)
    - Edit position (0-indexed)
    - Edit from/to nucleotides

    Does NOT require sequence B, making it suitable for pure edit modeling.

    Architecture:
    1. Mutation type embedding: Learned embedding for 12 SNV types
    2. Mutation effect: Δ of pretrained nucleotide token embeddings
    3. Position encoding: Sinusoidal + learned
    4. Local context: Mean-pooled window around edit site
    5. Attention context: Attention-weighted token sum
    6. Fusion MLP: Combines all components

    Args:
        rnafm_embedder: RNAFMEmbedder instance (pretrained, typically frozen)
        mutation_type_dim: Dimension of mutation type embedding (default: 64)
        mutation_effect_dim: Dimension of projected mutation effect (default: 256)
        position_dim: Total position encoding dimension (default: 64)
        local_context_dim: Dimension of local context (default: 256)
        attention_context_dim: Dimension of attention context (default: 128)
        fusion_hidden_dims: Hidden dims for fusion MLP (default: [512, 384])
        output_dim: Final edit embedding dimension (default: 256)
        window_size: Number of nucleotides on each side of edit (default: 10)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 512)
    """

    def __init__(
        self,
        rnafm_embedder: nn.Module,
        mutation_type_dim: int = 64,
        mutation_effect_dim: int = 256,
        position_dim: int = 64,
        local_context_dim: int = 256,
        attention_context_dim: int = 128,
        fusion_hidden_dims: List[int] = None,
        output_dim: int = 256,
        window_size: int = 10,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()

        self.rnafm = rnafm_embedder
        self.rnafm_dim = rnafm_embedder.embedding_dim  # 640 for RNA-FM
        self.window_size = window_size
        self.output_dim = output_dim

        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 384]

        # ========================================
        # 1. Mutation Type Embedding
        # ========================================
        self.mutation_type_embed = nn.Embedding(12, mutation_type_dim)

        # ========================================
        # 2. Mutation Effect (Δ token embedding)
        # ========================================
        # Learned nucleotide embeddings that mimic pretrained token semantics
        # We'll extract these from RNA-FM or learn them
        self.nucleotide_embed = nn.Embedding(4, self.rnafm_dim)
        self.mutation_effect_proj = nn.Sequential(
            nn.Linear(self.rnafm_dim, mutation_effect_dim),
            nn.LayerNorm(mutation_effect_dim),
            nn.ReLU()
        )

        # ========================================
        # 3. Position Encoding
        # ========================================
        # Sinusoidal (half of position_dim)
        sin_dim = position_dim // 2
        self.sinusoidal_pos = SinusoidalPositionalEncoding(sin_dim, max_seq_len)

        # Learned (other half)
        learned_dim = position_dim - sin_dim
        self.learned_pos_embed = nn.Embedding(max_seq_len, learned_dim)

        # Also encode relative position (pos / seq_len)
        self.relative_pos_proj = nn.Linear(1, position_dim // 4)

        # Final position dimension
        self.position_dim = sin_dim + learned_dim + position_dim // 4

        # ========================================
        # 4. Local Context (window ±N around edit)
        # ========================================
        self.local_context_proj = nn.Sequential(
            nn.Linear(self.rnafm_dim, local_context_dim),
            nn.LayerNorm(local_context_dim),
            nn.ReLU()
        )

        # ========================================
        # 5. Attention Context
        # ========================================
        self.attention_context_proj = nn.Sequential(
            nn.Linear(self.rnafm_dim, attention_context_dim),
            nn.LayerNorm(attention_context_dim),
            nn.ReLU()
        )

        # ========================================
        # 6. Fusion MLP
        # ========================================
        # Calculate total input dimension
        raw_dim = (
            mutation_type_dim +      # mutation type
            mutation_effect_dim +    # Δ token
            self.position_dim +      # position (sin + learned + relative)
            local_context_dim +      # local window
            attention_context_dim    # attention context
        )

        # Build fusion MLP
        fusion_layers = []
        prev_dim = raw_dim

        for hidden_dim in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final output layer (no dropout)
        fusion_layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        ])

        self.fusion_mlp = nn.Sequential(*fusion_layers)

        # Store dimensions for reference
        self._component_dims = {
            'mutation_type': mutation_type_dim,
            'mutation_effect': mutation_effect_dim,
            'position': self.position_dim,
            'local_context': local_context_dim,
            'attention_context': attention_context_dim,
            'raw_total': raw_dim,
            'output': output_dim
        }

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with reasonable defaults."""
        # Initialize nucleotide embeddings to capture basic differences
        # A, C, G, U have different chemical properties
        with torch.no_grad():
            # Simple initialization - will be refined during training
            nn.init.xavier_uniform_(self.nucleotide_embed.weight)
            nn.init.xavier_uniform_(self.mutation_type_embed.weight)

    def _get_rnafm_token_embeddings(
        self,
        sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get RNA-FM token-level embeddings and attention weights.

        Returns:
            token_embeddings: [batch, seq_len, 640]
            attention_weights: [batch, seq_len, seq_len] (averaged over heads/layers)
        """
        # Prepare data for RNA-FM
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.rnafm.batch_converter(data)
        batch_tokens = batch_tokens.to(next(self.rnafm.model.parameters()).device)

        # Get embeddings with attention
        with torch.no_grad():
            results = self.rnafm.model(batch_tokens, repr_layers=[12], need_head_weights=True)

            # Token embeddings [batch, seq_len, 640]
            token_embeddings = results["representations"][12]

            # Attention weights handling
            # RNA-FM returns attentions as tensor [batch, layers, heads, seq, seq]
            attentions = results["attentions"]

            if isinstance(attentions, torch.Tensor):
                # Shape: [batch, layers, heads, seq, seq]
                # Select last layer and average over heads
                attention_weights = attentions[:, -1, :, :, :].mean(dim=1)  # [batch, seq, seq]
            else:
                # If it's a tuple/list of per-layer tensors
                # Each element: [batch, heads, seq, seq]
                attention_weights = attentions[-1].mean(dim=1)  # [batch, seq, seq]

        return token_embeddings, attention_weights

    def forward(
        self,
        sequences: Union[str, List[str]],
        edit_positions: Union[int, List[int], torch.Tensor],
        edit_from: Union[str, List[str]],
        edit_to: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Generate edit embeddings.

        Args:
            sequences: RNA sequence(s) - the original sequence A
            edit_positions: 0-indexed position(s) of the edit
            edit_from: Original nucleotide(s) at edit position
            edit_to: New nucleotide(s) after edit

        Returns:
            edit_embeddings: [batch, output_dim] tensor
        """
        # Handle single inputs
        if isinstance(sequences, str):
            sequences = [sequences]
            edit_positions = [edit_positions]
            edit_from = [edit_from]
            edit_to = [edit_to]

        batch_size = len(sequences)
        device = next(self.parameters()).device

        # Convert edit_positions to tensor
        if isinstance(edit_positions, list):
            edit_positions = torch.tensor(edit_positions, device=device)
        elif isinstance(edit_positions, int):
            edit_positions = torch.tensor([edit_positions], device=device)

        # Normalize sequences (T -> U)
        sequences = [seq.upper().replace('T', 'U') for seq in sequences]

        # Get sequence lengths
        seq_lengths = torch.tensor([len(s) for s in sequences], device=device, dtype=torch.float)

        # ========================================
        # Get RNA-FM embeddings
        # ========================================
        token_embeddings, attention_weights = self._get_rnafm_token_embeddings(sequences)
        token_embeddings = token_embeddings.to(device)
        attention_weights = attention_weights.to(device)

        # ========================================
        # 1. Mutation Type Embedding
        # ========================================
        mutation_type_ids = torch.tensor([
            MUTATION_TYPES.get((f.upper(), t.upper()), 0)
            for f, t in zip(edit_from, edit_to)
        ], device=device)

        mutation_type_emb = self.mutation_type_embed(mutation_type_ids)  # [batch, 64]

        # ========================================
        # 2. Mutation Effect (Δ token)
        # ========================================
        from_ids = torch.tensor([NUC_TO_IDX[f.upper()] for f in edit_from], device=device)
        to_ids = torch.tensor([NUC_TO_IDX[t.upper()] for t in edit_to], device=device)

        from_emb = self.nucleotide_embed(from_ids)  # [batch, 640]
        to_emb = self.nucleotide_embed(to_ids)      # [batch, 640]
        delta_token = to_emb - from_emb             # [batch, 640]

        mutation_effect_emb = self.mutation_effect_proj(delta_token)  # [batch, 256]

        # ========================================
        # 3. Position Encoding
        # ========================================
        # Sinusoidal encoding
        sin_pos = self.sinusoidal_pos(edit_positions)  # [batch, sin_dim]

        # Learned position embedding
        learned_pos = self.learned_pos_embed(edit_positions)  # [batch, learned_dim]

        # Relative position (pos / seq_len)
        relative_pos = (edit_positions.float() / seq_lengths).unsqueeze(-1)  # [batch, 1]
        relative_pos_emb = self.relative_pos_proj(relative_pos)  # [batch, pos_dim//4]

        position_emb = torch.cat([sin_pos, learned_pos, relative_pos_emb], dim=-1)  # [batch, position_dim]

        # ========================================
        # 4. Local Context (window around edit)
        # ========================================
        # Note: RNA-FM adds special tokens, so actual sequence starts at index 1
        local_contexts = []

        for i in range(batch_size):
            pos = edit_positions[i].item()
            seq_len = int(seq_lengths[i].item())

            # Calculate window bounds (accounting for RNA-FM's +1 offset for special token)
            start = max(1, pos + 1 - self.window_size)
            end = min(seq_len + 1, pos + 1 + self.window_size + 1)

            # Extract window tokens
            window = token_embeddings[i, start:end, :]  # [window_len, 640]

            # Mean pool
            local_ctx = window.mean(dim=0)  # [640]
            local_contexts.append(local_ctx)

        local_context = torch.stack(local_contexts)  # [batch, 640]
        local_context_emb = self.local_context_proj(local_context)  # [batch, 256]

        # ========================================
        # 5. Attention Context
        # ========================================
        # Get attention weights TO the edit position from all other positions
        attention_contexts = []

        for i in range(batch_size):
            pos = edit_positions[i].item()
            seq_len = int(seq_lengths[i].item())

            # Attention from all positions to edit position (+1 for special token)
            attn_to_edit = attention_weights[i, :seq_len+2, pos+1]  # [seq_len+2]

            # Attention-weighted sum of tokens
            weighted_tokens = token_embeddings[i, :seq_len+2, :] * attn_to_edit.unsqueeze(-1)
            attn_ctx = weighted_tokens.sum(dim=0)  # [640]
            attention_contexts.append(attn_ctx)

        attention_context = torch.stack(attention_contexts)  # [batch, 640]
        attention_context_emb = self.attention_context_proj(attention_context)  # [batch, 128]

        # ========================================
        # 6. Concatenate and Fuse
        # ========================================
        raw_embedding = torch.cat([
            mutation_type_emb,      # [batch, 64]
            mutation_effect_emb,    # [batch, 256]
            position_emb,           # [batch, ~80]
            local_context_emb,      # [batch, 256]
            attention_context_emb   # [batch, 128]
        ], dim=-1)

        # Fusion MLP
        edit_embedding = self.fusion_mlp(raw_embedding)  # [batch, output_dim]

        return edit_embedding

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim

    @property
    def component_dims(self) -> Dict[str, int]:
        """Return dimensions of each component."""
        return self._component_dims

    def get_component_embeddings(
        self,
        sequences: Union[str, List[str]],
        edit_positions: Union[int, List[int], torch.Tensor],
        edit_from: Union[str, List[str]],
        edit_to: Union[str, List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Get individual component embeddings (for analysis/debugging).

        Returns dict with:
            - mutation_type: [batch, 64]
            - mutation_effect: [batch, 256]
            - position: [batch, ~80]
            - local_context: [batch, 256]
            - attention_context: [batch, 128]
            - raw: [batch, raw_dim] (concatenated)
            - fused: [batch, output_dim] (final)
        """
        # This is a debugging method - run forward pass but capture intermediates
        # For production, use forward() directly

        # Handle single inputs
        if isinstance(sequences, str):
            sequences = [sequences]
            edit_positions = [edit_positions]
            edit_from = [edit_from]
            edit_to = [edit_to]

        batch_size = len(sequences)
        device = next(self.parameters()).device

        if isinstance(edit_positions, list):
            edit_positions = torch.tensor(edit_positions, device=device)

        sequences = [seq.upper().replace('T', 'U') for seq in sequences]
        seq_lengths = torch.tensor([len(s) for s in sequences], device=device, dtype=torch.float)

        token_embeddings, attention_weights = self._get_rnafm_token_embeddings(sequences)
        token_embeddings = token_embeddings.to(device)
        attention_weights = attention_weights.to(device)

        # Component 1: Mutation type
        mutation_type_ids = torch.tensor([
            MUTATION_TYPES.get((f.upper(), t.upper()), 0)
            for f, t in zip(edit_from, edit_to)
        ], device=device)
        mutation_type_emb = self.mutation_type_embed(mutation_type_ids)

        # Component 2: Mutation effect
        from_ids = torch.tensor([NUC_TO_IDX[f.upper()] for f in edit_from], device=device)
        to_ids = torch.tensor([NUC_TO_IDX[t.upper()] for t in edit_to], device=device)
        delta_token = self.nucleotide_embed(to_ids) - self.nucleotide_embed(from_ids)
        mutation_effect_emb = self.mutation_effect_proj(delta_token)

        # Component 3: Position
        sin_pos = self.sinusoidal_pos(edit_positions)
        learned_pos = self.learned_pos_embed(edit_positions)
        relative_pos = (edit_positions.float() / seq_lengths).unsqueeze(-1)
        relative_pos_emb = self.relative_pos_proj(relative_pos)
        position_emb = torch.cat([sin_pos, learned_pos, relative_pos_emb], dim=-1)

        # Component 4: Local context
        local_contexts = []
        for i in range(batch_size):
            pos = edit_positions[i].item()
            seq_len = int(seq_lengths[i].item())
            start = max(1, pos + 1 - self.window_size)
            end = min(seq_len + 1, pos + 1 + self.window_size + 1)
            window = token_embeddings[i, start:end, :]
            local_contexts.append(window.mean(dim=0))
        local_context = torch.stack(local_contexts)
        local_context_emb = self.local_context_proj(local_context)

        # Component 5: Attention context
        attention_contexts = []
        for i in range(batch_size):
            pos = edit_positions[i].item()
            seq_len = int(seq_lengths[i].item())
            attn_to_edit = attention_weights[i, :seq_len+2, pos+1]
            weighted_tokens = token_embeddings[i, :seq_len+2, :] * attn_to_edit.unsqueeze(-1)
            attention_contexts.append(weighted_tokens.sum(dim=0))
        attention_context = torch.stack(attention_contexts)
        attention_context_emb = self.attention_context_proj(attention_context)

        # Raw concatenation
        raw_embedding = torch.cat([
            mutation_type_emb,
            mutation_effect_emb,
            position_emb,
            local_context_emb,
            attention_context_emb
        ], dim=-1)

        # Fused
        fused_embedding = self.fusion_mlp(raw_embedding)

        return {
            'mutation_type': mutation_type_emb,
            'mutation_effect': mutation_effect_emb,
            'position': position_emb,
            'local_context': local_context_emb,
            'attention_context': attention_context_emb,
            'raw': raw_embedding,
            'fused': fused_embedding
        }

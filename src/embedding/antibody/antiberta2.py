"""
AntiBERTa2 Antibody Embedder.

AntiBERTa2 is a RoFormer-based antibody language model trained with masked
language modeling on antibody sequences.

Reference:
    Leem et al. "Deciphering the language of antibodies using self-supervised learning"
    https://huggingface.co/alchemab/antiberta2

NOTE: AntiBERTa2 is single-chain only. This wrapper implements a cross-attention
fusion layer to handle paired heavy-light chain input.

LICENSING: AntiBERTa2 is only available for non-commercial use.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .base import AntibodyEmbedder, AntibodyEmbedderOutput, BatchedAntibodyEmbedderOutput


class CrossChainFusion(nn.Module):
    """
    Cross-attention fusion layer for combining heavy and light chain embeddings.

    This module allows information flow between the two chains using
    bidirectional cross-attention, followed by concatenation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Heavy attends to Light
        self.h_to_l_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Light attends to Heavy
        self.l_to_h_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.h_ln = nn.LayerNorm(embed_dim)
        self.l_ln = nn.LayerNorm(embed_dim)

        # FFN for fused representation
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        heavy_emb: torch.Tensor,
        light_emb: torch.Tensor,
        heavy_mask: Optional[torch.Tensor] = None,
        light_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Fuse heavy and light chain embeddings.

        Args:
            heavy_emb: Heavy chain embeddings [B, L_H, D] or [L_H, D]
            light_emb: Light chain embeddings [B, L_L, D] or [L_L, D]
            heavy_mask: Optional mask for heavy chain [B, L_H] or [L_H]
            light_mask: Optional mask for light chain [B, L_L] or [L_L]

        Returns:
            Tuple of (fused_heavy, fused_light, global_embedding)
        """
        # Handle unbatched input
        squeeze = False
        if heavy_emb.dim() == 2:
            heavy_emb = heavy_emb.unsqueeze(0)
            light_emb = light_emb.unsqueeze(0)
            if heavy_mask is not None:
                heavy_mask = heavy_mask.unsqueeze(0)
            if light_mask is not None:
                light_mask = light_mask.unsqueeze(0)
            squeeze = True

        # Convert masks to key_padding_mask format (True = ignore)
        h_key_mask = ~heavy_mask if heavy_mask is not None else None
        l_key_mask = ~light_mask if light_mask is not None else None

        # Cross-attention: Heavy attends to Light
        h_cross, _ = self.h_to_l_attn(
            query=heavy_emb,
            key=light_emb,
            value=light_emb,
            key_padding_mask=l_key_mask,
        )
        heavy_fused = self.h_ln(heavy_emb + h_cross)

        # Cross-attention: Light attends to Heavy
        l_cross, _ = self.l_to_h_attn(
            query=light_emb,
            key=heavy_emb,
            value=heavy_emb,
            key_padding_mask=h_key_mask,
        )
        light_fused = self.l_ln(light_emb + l_cross)

        # Global embedding: pool both chains and combine
        if heavy_mask is not None:
            h_pooled = (heavy_fused * heavy_mask.unsqueeze(-1)).sum(1) / heavy_mask.sum(1, keepdim=True)
        else:
            h_pooled = heavy_fused.mean(1)

        if light_mask is not None:
            l_pooled = (light_fused * light_mask.unsqueeze(-1)).sum(1) / light_mask.sum(1, keepdim=True)
        else:
            l_pooled = light_fused.mean(1)

        # Combine pooled representations
        combined = torch.cat([h_pooled, l_pooled], dim=-1)
        global_emb = self.ffn(combined)

        if squeeze:
            heavy_fused = heavy_fused.squeeze(0)
            light_fused = light_fused.squeeze(0)
            global_emb = global_emb.squeeze(0)

        return heavy_fused, light_fused, global_emb


class AntiBERTa2Embedder(AntibodyEmbedder):
    """
    AntiBERTa2 antibody embedder with cross-chain fusion.

    Since AntiBERTa2 is single-chain, this wrapper:
    1. Encodes heavy and light chains separately
    2. Uses a cross-attention fusion layer to combine them

    Args:
        model_name: HuggingFace model ID (default: "alchemab/antiberta2")
        trainable: Whether to allow fine-tuning (default: False)
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        pooling: Pooling strategy for global embedding ('mean', 'cls', 'max')
        max_length: Maximum sequence length per chain (default: 256)
        fusion_heads: Number of attention heads in fusion layer (default: 8)
        fusion_dropout: Dropout in fusion layer (default: 0.1)

    Example:
        >>> embedder = AntiBERTa2Embedder()
        >>> output = embedder.encode(
        ...     heavy_sequence="EVQLVQSGAEVKKPGAS...",
        ...     light_sequence="DIVMTQSPDSLAVSLGER..."
        ... )
        >>> print(output.global_embedding.shape)
        torch.Size([768])

    Note:
        AntiBERTa2 is only available for non-commercial use.
    """

    def __init__(
        self,
        model_name: str = "alchemab/antiberta2",
        trainable: bool = False,
        device: str = 'auto',
        pooling: str = 'mean',
        max_length: int = 256,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
    ):
        super().__init__(trainable=trainable, device=device, pooling=pooling)

        self.model_name = model_name
        self.max_length = max_length

        # Import here to avoid hard dependency
        try:
            from transformers import RoFormerModel, RoFormerTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for AntiBERTa2Embedder. "
                "Install with: pip install transformers"
            )

        # Load tokenizer and model
        self.tokenizer = RoFormerTokenizer.from_pretrained(model_name)
        self.model = RoFormerModel.from_pretrained(model_name)
        self.model.to(self._device)

        self._embedding_dim = self.model.config.hidden_size

        # Cross-chain fusion layer
        self.fusion = CrossChainFusion(
            embed_dim=self._embedding_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
        )
        self.fusion.to(self._device)

        # Set trainability
        if not trainable:
            self.freeze()
            # Keep fusion layer trainable by default
            for param in self.fusion.parameters():
                param.requires_grad = True

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def name(self) -> str:
        return "AntiBERTa2"

    @property
    def supports_paired(self) -> bool:
        return False  # Native model is single-chain

    def _encode_single_chain(
        self,
        sequence: str,
        return_attention: bool = False,
    ) -> tuple:
        """Encode a single chain."""
        # CRITICAL: AntiBERTa2 (RoFormer) requires SPACE-SEPARATED amino acids!
        # Without spaces, the entire sequence becomes a single [UNK] token.
        # E.g., "EVQL" -> [CLS, UNK, SEP] (wrong)
        #       "E V Q L" -> [CLS, E, V, Q, L, SEP] (correct)
        spaced_sequence = ' '.join(sequence)

        # Tokenize
        tokens = self.tokenizer(
            spaced_sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        # Forward pass
        with torch.set_grad_enabled(self.trainable):
            outputs = self.model(
                **tokens,
                output_attentions=return_attention,
            )

        # Get hidden states [1, seq_len, d_model]
        hidden_states = outputs.last_hidden_state[0]

        # Remove special tokens (typically [CLS] at start, [SEP] at end)
        residue_emb = hidden_states[1:-1][:len(sequence)]

        attention = None
        if return_attention and outputs.attentions is not None:
            attention = torch.stack(outputs.attentions).mean(dim=(0, 2))

        return residue_emb, attention

    def encode(
        self,
        heavy_sequence: str,
        light_sequence: str,
        return_attention: bool = False,
    ) -> AntibodyEmbedderOutput:
        """
        Encode a single antibody.

        Args:
            heavy_sequence: Heavy chain amino acid sequence
            light_sequence: Light chain amino acid sequence
            return_attention: Whether to return attention weights

        Returns:
            AntibodyEmbedderOutput with per-residue and global embeddings
        """
        # Ensure uppercase
        heavy_sequence = heavy_sequence.upper().strip()
        light_sequence = light_sequence.upper().strip()

        has_heavy = len(heavy_sequence) > 0
        has_light = len(light_sequence) > 0

        # Handle missing chains - cross-attention requires both chains
        if not has_heavy and not has_light:
            # Return zeros if both chains are empty
            zero_emb = torch.zeros(self._embedding_dim, device=self._device)
            return AntibodyEmbedderOutput(
                heavy_residue_embeddings=torch.zeros(0, self._embedding_dim, device=self._device),
                light_residue_embeddings=torch.zeros(0, self._embedding_dim, device=self._device),
                global_embedding=zero_emb,
                heavy_attention_weights=None,
                light_attention_weights=None,
                heavy_sequence=heavy_sequence,
                light_sequence=light_sequence,
            )

        if not has_heavy or not has_light:
            # If only one chain, use self-attention by passing it as both
            single_seq = heavy_sequence if has_heavy else light_sequence
            single_emb, single_attn = self._encode_single_chain(single_seq, return_attention)

            # Use the single chain as both heavy and light for fusion
            # This gives a self-attention enriched representation
            heavy_fused, light_fused, global_emb = self.fusion(single_emb, single_emb)

            if has_heavy:
                return AntibodyEmbedderOutput(
                    heavy_residue_embeddings=heavy_fused,
                    light_residue_embeddings=torch.zeros(0, self._embedding_dim, device=self._device),
                    global_embedding=global_emb,
                    heavy_attention_weights=single_attn,
                    light_attention_weights=None,
                    heavy_sequence=heavy_sequence,
                    light_sequence=light_sequence,
                )
            else:
                return AntibodyEmbedderOutput(
                    heavy_residue_embeddings=torch.zeros(0, self._embedding_dim, device=self._device),
                    light_residue_embeddings=light_fused,
                    global_embedding=global_emb,
                    heavy_attention_weights=None,
                    light_attention_weights=single_attn,
                    heavy_sequence=heavy_sequence,
                    light_sequence=light_sequence,
                )

        # Both chains present - normal path
        # Encode each chain separately
        heavy_emb, heavy_attn = self._encode_single_chain(heavy_sequence, return_attention)
        light_emb, light_attn = self._encode_single_chain(light_sequence, return_attention)

        # Fuse chains using cross-attention
        heavy_fused, light_fused, global_emb = self.fusion(heavy_emb, light_emb)

        return AntibodyEmbedderOutput(
            heavy_residue_embeddings=heavy_fused,
            light_residue_embeddings=light_fused,
            global_embedding=global_emb,
            heavy_attention_weights=heavy_attn,
            light_attention_weights=light_attn,
            heavy_sequence=heavy_sequence,
            light_sequence=light_sequence,
        )

    def encode_batch(
        self,
        heavy_sequences: List[str],
        light_sequences: List[str],
        return_attention: bool = False,
    ) -> BatchedAntibodyEmbedderOutput:
        """
        Encode a batch of antibodies.

        Args:
            heavy_sequences: List of heavy chain sequences
            light_sequences: List of light chain sequences
            return_attention: Whether to return attention weights

        Returns:
            BatchedAntibodyEmbedderOutput with batched embeddings
        """
        batch_size = len(heavy_sequences)

        # Normalize sequences
        heavy_sequences = [h.upper().strip() for h in heavy_sequences]
        light_sequences = [l.upper().strip() for l in light_sequences]

        # CRITICAL: Space-separate for RoFormer tokenizer
        heavy_spaced = [' '.join(h) for h in heavy_sequences]
        light_spaced = [' '.join(l) for l in light_sequences]

        # Tokenize heavy chains
        heavy_tokens = self.tokenizer(
            heavy_spaced,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        heavy_tokens = {k: v.to(self._device) for k, v in heavy_tokens.items()}

        # Tokenize light chains
        light_tokens = self.tokenizer(
            light_spaced,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        light_tokens = {k: v.to(self._device) for k, v in light_tokens.items()}

        # Encode both chains
        with torch.set_grad_enabled(self.trainable):
            heavy_outputs = self.model(**heavy_tokens, output_attentions=return_attention)
            light_outputs = self.model(**light_tokens, output_attentions=return_attention)

        heavy_hidden = heavy_outputs.last_hidden_state  # [B, seq_len, d]
        light_hidden = light_outputs.last_hidden_state  # [B, seq_len, d]

        # Find max lengths for output
        max_h_len = max(len(h) for h in heavy_sequences)
        max_l_len = max(len(l) for l in light_sequences)

        # Extract residue embeddings (remove special tokens)
        h_emb = torch.zeros(batch_size, max_h_len, self.embedding_dim, device=self._device)
        l_emb = torch.zeros(batch_size, max_l_len, self.embedding_dim, device=self._device)
        h_mask = torch.zeros(batch_size, max_h_len, dtype=torch.bool, device=self._device)
        l_mask = torch.zeros(batch_size, max_l_len, dtype=torch.bool, device=self._device)

        for i in range(batch_size):
            h_len = len(heavy_sequences[i])
            l_len = len(light_sequences[i])

            # Remove [CLS] and [SEP] tokens
            h_emb[i, :h_len] = heavy_hidden[i, 1:h_len + 1]
            l_emb[i, :l_len] = light_hidden[i, 1:l_len + 1]
            h_mask[i, :h_len] = True
            l_mask[i, :l_len] = True

        # Fuse chains
        h_fused, l_fused, g_emb = self.fusion(h_emb, l_emb, h_mask, l_mask)

        return BatchedAntibodyEmbedderOutput(
            heavy_residue_embeddings=h_fused,
            light_residue_embeddings=l_fused,
            global_embeddings=g_emb,
            heavy_sequences=heavy_sequences,
            light_sequences=light_sequences,
            heavy_mask=h_mask,
            light_mask=l_mask,
        )

    def freeze_base_model(self):
        """Freeze only the base RoFormer model, keep fusion trainable."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        """Unfreeze the base RoFormer model."""
        for param in self.model.parameters():
            param.requires_grad = True

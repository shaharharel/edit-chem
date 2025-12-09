"""
BALM (Bio-inspired Antibody Language Model) Embedder.

BALM is an ESM-based antibody language model trained on 336 million antibody
sequences. It's designed for single-chain encoding with strong performance
on antibody-specific tasks.

Reference:
    "Accurate prediction of antibody function and structure using
    bio-inspired antibody language model" (2024)
    https://github.com/BEAM-Labs/BALM

Note: BALM is single-chain. This wrapper handles heavy and light chains
separately and provides a fusion layer for paired input.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import os

from .base import AntibodyEmbedder, AntibodyEmbedderOutput, BatchedAntibodyEmbedderOutput
from .antiberta2 import CrossChainFusion  # Reuse fusion layer


class BALMEmbedder(AntibodyEmbedder):
    """
    BALM (Bio-inspired Antibody Language Model) embedder.

    BALM is built on the ESM architecture and trained specifically on antibody
    sequences. Since it's single-chain, we use a cross-chain fusion layer
    for paired H/L input.

    Args:
        model_path: Path to BALM checkpoint or 'auto' to download
        trainable: Whether to allow fine-tuning (default: False)
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        pooling: Pooling strategy for global embedding ('mean', 'cls', 'max')
        max_length: Maximum sequence length per chain (default: 256)
        fusion_heads: Number of attention heads in fusion layer (default: 8)

    Example:
        >>> embedder = BALMEmbedder()
        >>> output = embedder.encode(
        ...     heavy_sequence="EVQLVQSGAEVKKPGAS...",
        ...     light_sequence="DIVMTQSPDSLAVSLGER..."
        ... )

    Note:
        Requires downloading BALM weights from the official repository.
        See: https://github.com/BEAM-Labs/BALM
    """

    def __init__(
        self,
        model_path: str = 'auto',
        trainable: bool = False,
        device: str = 'auto',
        pooling: str = 'mean',
        max_length: int = 256,
        fusion_heads: int = 8,
    ):
        super().__init__(trainable=trainable, device=device, pooling=pooling)

        self.model_path = model_path
        self.max_length = max_length

        # Load model
        self._load_model(model_path)

        # Fusion layer for paired chains
        self.fusion = CrossChainFusion(
            embed_dim=self._embedding_dim,
            num_heads=fusion_heads,
        )
        self.fusion.to(self._device)

        if not trainable:
            self.freeze()
            # Keep fusion trainable
            for param in self.fusion.parameters():
                param.requires_grad = True

    def _load_model(self, model_path: str):
        """Load BALM model."""
        try:
            from transformers import EsmTokenizer, EsmModel
        except ImportError:
            raise ImportError(
                "transformers is required for BALM. "
                "Install with: pip install transformers"
            )

        # BALM uses ESM architecture with custom weights
        # Try to load from local path or use ESM as fallback
        if model_path == 'auto' or not os.path.exists(model_path):
            # Try to use ESM-2 as a fallback (similar architecture)
            # In production, you would download BALM weights from their repo
            import warnings
            warnings.warn(
                "BALM weights not found. Using ESM-2 (650M) as a fallback. "
                "For best results, download BALM weights from "
                "https://github.com/BEAM-Labs/BALM"
            )
            model_name = "facebook/esm2_t33_650M_UR50D"
            self.tokenizer = EsmTokenizer.from_pretrained(model_name)
            self.model = EsmModel.from_pretrained(model_name)
        else:
            # Load from local checkpoint
            # BALM uses EsmTokenizer
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

            # Load custom weights
            self.model = EsmModel.from_pretrained(
                "facebook/esm2_t33_650M_UR50D"
            )
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self._device)
        self._embedding_dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def name(self) -> str:
        return "BALM"

    @property
    def supports_paired(self) -> bool:
        return False  # Native model is single-chain

    def _encode_single_chain(
        self,
        sequence: str,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode a single chain."""
        # Tokenize
        tokens = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        # Forward
        with torch.set_grad_enabled(self.trainable):
            outputs = self.model(
                **tokens,
                output_attentions=return_attention,
            )

        hidden = outputs.last_hidden_state[0]  # [seq_len, d]

        # Remove special tokens (ESM uses <cls> and <eos>)
        residue_emb = hidden[1:-1][:len(sequence)]

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
        # Encode each chain
        heavy_emb, heavy_attn = self._encode_single_chain(heavy_sequence, return_attention)
        light_emb, light_attn = self._encode_single_chain(light_sequence, return_attention)

        # Fuse
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
        """
        batch_size = len(heavy_sequences)

        heavy_sequences = [h.upper().strip() for h in heavy_sequences]
        light_sequences = [l.upper().strip() for l in light_sequences]

        # Tokenize heavy chains
        heavy_tokens = self.tokenizer(
            heavy_sequences,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        heavy_tokens = {k: v.to(self._device) for k, v in heavy_tokens.items()}

        # Tokenize light chains
        light_tokens = self.tokenizer(
            light_sequences,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        light_tokens = {k: v.to(self._device) for k, v in light_tokens.items()}

        # Encode
        with torch.set_grad_enabled(self.trainable):
            heavy_outputs = self.model(**heavy_tokens)
            light_outputs = self.model(**light_tokens)

        heavy_hidden = heavy_outputs.last_hidden_state
        light_hidden = light_outputs.last_hidden_state

        # Extract and pad
        max_h_len = max(len(h) for h in heavy_sequences)
        max_l_len = max(len(l) for l in light_sequences)

        h_emb = torch.zeros(batch_size, max_h_len, self.embedding_dim, device=self._device)
        l_emb = torch.zeros(batch_size, max_l_len, self.embedding_dim, device=self._device)
        h_mask = torch.zeros(batch_size, max_h_len, dtype=torch.bool, device=self._device)
        l_mask = torch.zeros(batch_size, max_l_len, dtype=torch.bool, device=self._device)

        for i in range(batch_size):
            h_len = len(heavy_sequences[i])
            l_len = len(light_sequences[i])

            h_emb[i, :h_len] = heavy_hidden[i, 1:h_len + 1]
            l_emb[i, :l_len] = light_hidden[i, 1:l_len + 1]
            h_mask[i, :h_len] = True
            l_mask[i, :l_len] = True

        # Fuse
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

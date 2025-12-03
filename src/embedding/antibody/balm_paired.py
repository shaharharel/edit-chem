"""
BALM-Paired Antibody Embedder.

BALM-Paired is a RoBERTa-based antibody language model trained on natively
paired heavy-light chain sequences from the Briney Lab.

Reference:
    "Improving antibody language models with native pairing" (2023)
    https://github.com/brineylab/BALM-paper
    Model weights available on Zenodo

Input format:
    BALM-Paired expects concatenated H/L sequences with a separator.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import os

from .base import AntibodyEmbedder, AntibodyEmbedderOutput, BatchedAntibodyEmbedderOutput


class BALMPairedEmbedder(AntibodyEmbedder):
    """
    BALM-Paired antibody embedder (Briney Lab).

    This model is trained on natively paired antibody sequences and supports
    both heavy and light chains as paired input.

    Args:
        model_path: Path to BALM-Paired checkpoint or HuggingFace ID
        trainable: Whether to allow fine-tuning (default: False)
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        pooling: Pooling strategy for global embedding ('mean', 'cls', 'max')
        max_length: Maximum sequence length (default: 512)

    Example:
        >>> embedder = BALMPairedEmbedder()
        >>> output = embedder.encode(
        ...     heavy_sequence="EVQLVQSGAEVKKPGAS...",
        ...     light_sequence="DIVMTQSPDSLAVSLGER..."
        ... )

    Note:
        Model weights available from Zenodo (CC BY-SA 4.0 license).
        See: https://github.com/brineylab/BALM-paper
    """

    # Separator token between H and L chains
    SEPARATOR = "[SEP]"

    def __init__(
        self,
        model_path: str = 'auto',
        trainable: bool = False,
        device: str = 'auto',
        pooling: str = 'mean',
        max_length: int = 512,
    ):
        super().__init__(trainable=trainable, device=device, pooling=pooling)

        self.model_path = model_path
        self.max_length = max_length

        self._load_model(model_path)

        if not trainable:
            self.freeze()

    def _load_model(self, model_path: str):
        """Load BALM-Paired model."""
        try:
            from transformers import RobertaModel, RobertaTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for BALM-Paired. "
                "Install with: pip install transformers"
            )

        if model_path == 'auto' or not os.path.exists(model_path):
            # BALM-Paired uses RoBERTa-large architecture
            # Use a fallback for testing
            import warnings
            warnings.warn(
                "BALM-Paired weights not found. Using roberta-base as fallback. "
                "Download BALM-Paired weights from Zenodo for best results. "
                "See: https://github.com/brineylab/BALM-paper"
            )
            model_name = "roberta-base"
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaModel.from_pretrained(model_name)
        else:
            # Load from local checkpoint
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.model = RobertaModel.from_pretrained(model_path)

        self.model.to(self._device)
        self._embedding_dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def name(self) -> str:
        return "BALM-Paired"

    @property
    def supports_paired(self) -> bool:
        return True

    def _prepare_sequence(self, heavy: str, light: str) -> str:
        """
        Prepare paired sequence for BALM-Paired.

        Format: "<s> HEAVY </s></s> LIGHT </s>"
        The tokenizer handles <s> and final </s> automatically when using
        two sequences.
        """
        heavy = heavy.upper().strip()
        light = light.upper().strip()

        # Space-separate amino acids for RoBERTa tokenizer
        heavy_spaced = ' '.join(heavy)
        light_spaced = ' '.join(light)

        return heavy_spaced, light_spaced

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
        heavy_spaced, light_spaced = self._prepare_sequence(heavy_sequence, light_sequence)

        # Tokenize as sequence pair
        tokens = self.tokenizer(
            heavy_spaced,
            light_spaced,
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

        # Find separator to split H and L
        # RoBERTa sequence pair format: <s> A </s></s> B </s>
        sep_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id
        input_ids = tokens['input_ids'][0]

        # Find </s> positions
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

        h_len = len(heavy_sequence)
        l_len = len(light_sequence)

        if len(sep_positions) >= 2:
            # First </s> marks end of heavy chain
            first_sep = sep_positions[0].item()
            # Second </s></s> or </s> marks start of light chain
            second_sep = sep_positions[1].item()

            # Extract embeddings
            # Skip <s> at position 0
            heavy_emb = hidden[1:first_sep][:h_len]

            # Light chain starts after the double </s></s>
            # Position varies by tokenizer version
            light_start = second_sep + 1
            light_emb = hidden[light_start:-1][:l_len]  # Exclude final </s>
        else:
            # Fallback: split based on sequence lengths
            heavy_emb = hidden[1:h_len + 1]
            light_emb = hidden[h_len + 3:h_len + 3 + l_len]  # +3 for </s></s>

        # Ensure correct dimensions
        if heavy_emb.shape[0] < h_len:
            pad = torch.zeros(h_len - heavy_emb.shape[0], self.embedding_dim, device=self._device)
            heavy_emb = torch.cat([heavy_emb, pad], dim=0)
        heavy_emb = heavy_emb[:h_len]

        if light_emb.shape[0] < l_len:
            pad = torch.zeros(l_len - light_emb.shape[0], self.embedding_dim, device=self._device)
            light_emb = torch.cat([light_emb, pad], dim=0)
        light_emb = light_emb[:l_len]

        # Global embedding
        all_emb = torch.cat([heavy_emb, light_emb], dim=0)
        global_emb = self._pool_embeddings(all_emb)

        # Attention
        heavy_attn = None
        light_attn = None
        if return_attention and outputs.attentions is not None:
            attn = torch.stack(outputs.attentions).mean(dim=(0, 2))
            heavy_attn = attn[1:h_len + 1, 1:h_len + 1]
            # Approximate light attention position
            l_start = h_len + 3
            light_attn = attn[l_start:l_start + l_len, l_start:l_start + l_len]

        return AntibodyEmbedderOutput(
            heavy_residue_embeddings=heavy_emb,
            light_residue_embeddings=light_emb,
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

        # Prepare sequences
        heavy_spaced = []
        light_spaced = []
        for h, l in zip(heavy_sequences, light_sequences):
            h_sp, l_sp = self._prepare_sequence(h, l)
            heavy_spaced.append(h_sp)
            light_spaced.append(l_sp)

        # Tokenize batch
        tokens = self.tokenizer(
            heavy_spaced,
            light_spaced,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        # Forward
        with torch.set_grad_enabled(self.trainable):
            outputs = self.model(**tokens, output_attentions=return_attention)

        hidden = outputs.last_hidden_state  # [B, seq_len, d]

        # Find max lengths
        max_h_len = max(len(h) for h in heavy_sequences)
        max_l_len = max(len(l) for l in light_sequences)

        # Initialize outputs
        h_emb = torch.zeros(batch_size, max_h_len, self.embedding_dim, device=self._device)
        l_emb = torch.zeros(batch_size, max_l_len, self.embedding_dim, device=self._device)
        g_emb = torch.zeros(batch_size, self.embedding_dim, device=self._device)
        h_mask = torch.zeros(batch_size, max_h_len, dtype=torch.bool, device=self._device)
        l_mask = torch.zeros(batch_size, max_l_len, dtype=torch.bool, device=self._device)

        sep_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        for i in range(batch_size):
            h_len = len(heavy_sequences[i])
            l_len = len(light_sequences[i])

            input_ids = tokens['input_ids'][i]
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

            if len(sep_positions) >= 2:
                first_sep = sep_positions[0].item()
                second_sep = sep_positions[1].item()
                heavy_emb_i = hidden[i, 1:first_sep][:h_len]
                light_start = second_sep + 1
                light_emb_i = hidden[i, light_start:-1][:l_len]
            else:
                heavy_emb_i = hidden[i, 1:h_len + 1]
                light_emb_i = hidden[i, h_len + 3:h_len + 3 + l_len]

            actual_h_len = min(heavy_emb_i.shape[0], h_len)
            actual_l_len = min(light_emb_i.shape[0], l_len)

            h_emb[i, :actual_h_len] = heavy_emb_i[:actual_h_len]
            l_emb[i, :actual_l_len] = light_emb_i[:actual_l_len]
            h_mask[i, :actual_h_len] = True
            l_mask[i, :actual_l_len] = True

            # Global
            all_emb = torch.cat([heavy_emb_i[:actual_h_len], light_emb_i[:actual_l_len]], dim=0)
            g_emb[i] = self._pool_embeddings(all_emb)

        return BatchedAntibodyEmbedderOutput(
            heavy_residue_embeddings=h_emb,
            light_residue_embeddings=l_emb,
            global_embeddings=g_emb,
            heavy_sequences=heavy_sequences,
            light_sequences=light_sequences,
            heavy_mask=h_mask,
            light_mask=l_mask,
        )

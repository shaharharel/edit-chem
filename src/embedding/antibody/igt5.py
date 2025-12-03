"""
IgT5 Antibody Embedder.

IgT5 is a T5-based antibody language model trained on paired heavy-light chain
sequences from the Observed Antibody Space (OAS) dataset.

Reference:
    Kenlay et al. "Large scale paired antibody language models" (2024)
    https://huggingface.co/Exscientia/IgT5

Input format:
    Heavy and light chains are concatenated with </s> separator,
    with amino acids space-separated: "E V Q L ... </s> D I V M ..."
"""

import torch
import torch.nn as nn
from typing import List, Optional
import warnings

from .base import AntibodyEmbedder, AntibodyEmbedderOutput, BatchedAntibodyEmbedderOutput


class IgT5Embedder(AntibodyEmbedder):
    """
    IgT5 antibody embedder using the Exscientia/IgT5 model.

    This model natively supports paired heavy-light chain input using
    a </s> separator token between the two chains.

    Args:
        model_name: HuggingFace model ID (default: "Exscientia/IgT5")
        trainable: Whether to allow fine-tuning (default: False)
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        pooling: Pooling strategy for global embedding ('mean', 'cls', 'max')
        max_length: Maximum sequence length (default: 512)

    Example:
        >>> embedder = IgT5Embedder()
        >>> output = embedder.encode(
        ...     heavy_sequence="EVQLVQSGAEVKKPGAS...",
        ...     light_sequence="DIVMTQSPDSLAVSLGER..."
        ... )
        >>> print(output.global_embedding.shape)
        torch.Size([1024])
    """

    def __init__(
        self,
        model_name: str = "Exscientia/IgT5",
        trainable: bool = False,
        device: str = 'auto',
        pooling: str = 'mean',
        max_length: int = 512,
    ):
        super().__init__(trainable=trainable, device=device, pooling=pooling)

        self.model_name = model_name
        self.max_length = max_length

        # Import here to avoid hard dependency
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for IgT5Embedder. "
                "Install with: pip install transformers"
            )

        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.to(self._device)

        # Set trainability
        if not trainable:
            self.freeze()

        self._embedding_dim = self.model.config.d_model

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def name(self) -> str:
        return "IgT5"

    @property
    def supports_paired(self) -> bool:
        return True

    def _prepare_sequence(self, heavy: str, light: str) -> str:
        """
        Prepare paired sequence in IgT5 format.

        IgT5 expects space-separated amino acids with </s> separator:
        "E V Q L ... </s> D I V M ..."
        """
        # Ensure uppercase
        heavy = heavy.upper().strip()
        light = light.upper().strip()

        # Space-separate amino acids
        heavy_spaced = ' '.join(heavy)
        light_spaced = ' '.join(light)

        # Combine with separator
        return f"{heavy_spaced} </s> {light_spaced}"

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
        # Prepare input
        paired_seq = self._prepare_sequence(heavy_sequence, light_sequence)

        # Tokenize
        tokens = self.tokenizer(
            paired_seq,
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

        # Get hidden states: [1, seq_len, d_model]
        hidden_states = outputs.last_hidden_state[0]  # [seq_len, d_model]

        # Find separator position to split H and L
        # The separator </s> is represented as token_id for '</s>'
        input_ids = tokens['input_ids'][0]
        sep_token_id = self.tokenizer.convert_tokens_to_ids('</s>')

        # Find separator position (should be after heavy chain)
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

        if len(sep_positions) > 0:
            # First separator marks end of heavy chain
            sep_pos = sep_positions[0].item()

            # Split embeddings (excluding special tokens)
            # Token 0 is usually <pad> or start, then heavy, then </s>, then light, then </s>
            heavy_emb = hidden_states[1:sep_pos]  # Skip first token, up to separator
            light_emb = hidden_states[sep_pos + 1:-1]  # After separator to before final </s>
        else:
            # Fallback: split roughly in half
            mid = len(heavy_sequence)
            heavy_emb = hidden_states[1:mid + 1]
            light_emb = hidden_states[mid + 1:-1]

        # Ensure dimensions match sequence lengths (may need padding adjustment)
        # Truncate if needed due to tokenization differences
        heavy_emb = heavy_emb[:len(heavy_sequence)]
        light_emb = light_emb[:len(light_sequence)]

        # Compute global embedding
        # Pool over both chains
        all_emb = torch.cat([heavy_emb, light_emb], dim=0)
        global_emb = self._pool_embeddings(all_emb)

        # Handle attention weights
        heavy_attn = None
        light_attn = None
        if return_attention and outputs.attentions is not None:
            # Average attention across heads and layers
            # attentions: tuple of [1, heads, seq, seq] for each layer
            attn = torch.stack(outputs.attentions).mean(dim=(0, 2))  # [seq, seq]
            if len(sep_positions) > 0:
                sep_pos = sep_positions[0].item()
                heavy_attn = attn[1:sep_pos, 1:sep_pos]
                light_attn = attn[sep_pos + 1:-1, sep_pos + 1:-1]

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
        Encode a batch of antibodies efficiently.

        Args:
            heavy_sequences: List of heavy chain sequences
            light_sequences: List of light chain sequences
            return_attention: Whether to return attention weights

        Returns:
            BatchedAntibodyEmbedderOutput with batched embeddings
        """
        batch_size = len(heavy_sequences)

        # Prepare all sequences
        paired_seqs = [
            self._prepare_sequence(h, l)
            for h, l in zip(heavy_sequences, light_sequences)
        ]

        # Tokenize batch
        tokens = self.tokenizer(
            paired_seqs,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        # Forward pass
        with torch.set_grad_enabled(self.trainable):
            outputs = self.model(
                **tokens,
                output_attentions=return_attention,
            )

        hidden_states = outputs.last_hidden_state  # [B, seq_len, d_model]

        # Find max lengths for padding
        max_h_len = max(len(h) for h in heavy_sequences)
        max_l_len = max(len(l) for l in light_sequences)

        # Initialize output tensors
        h_emb = torch.zeros(batch_size, max_h_len, self.embedding_dim, device=self._device)
        l_emb = torch.zeros(batch_size, max_l_len, self.embedding_dim, device=self._device)
        g_emb = torch.zeros(batch_size, self.embedding_dim, device=self._device)
        h_mask = torch.zeros(batch_size, max_h_len, dtype=torch.bool, device=self._device)
        l_mask = torch.zeros(batch_size, max_l_len, dtype=torch.bool, device=self._device)

        sep_token_id = self.tokenizer.convert_tokens_to_ids('</s>')

        for i in range(batch_size):
            input_ids = tokens['input_ids'][i]
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

            h_len = len(heavy_sequences[i])
            l_len = len(light_sequences[i])

            if len(sep_positions) > 0:
                sep_pos = sep_positions[0].item()
                heavy_emb_i = hidden_states[i, 1:sep_pos][:h_len]
                light_emb_i = hidden_states[i, sep_pos + 1:-1][:l_len]
            else:
                heavy_emb_i = hidden_states[i, 1:h_len + 1]
                light_emb_i = hidden_states[i, h_len + 1:h_len + 1 + l_len]

            # Ensure correct lengths
            actual_h_len = min(heavy_emb_i.shape[0], h_len)
            actual_l_len = min(light_emb_i.shape[0], l_len)

            h_emb[i, :actual_h_len] = heavy_emb_i[:actual_h_len]
            l_emb[i, :actual_l_len] = light_emb_i[:actual_l_len]
            h_mask[i, :actual_h_len] = True
            l_mask[i, :actual_l_len] = True

            # Global embedding
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

"""
IgBert Antibody Embedder.

IgBert is a BERT-based antibody language model trained on paired heavy-light chain
sequences from the Observed Antibody Space (OAS) dataset.

Reference:
    Kenlay et al. "Large scale paired antibody language models" (2024)
    https://huggingface.co/Exscientia/IgBert

Input format:
    Heavy and light chains are concatenated with [SEP] separator,
    with amino acids space-separated: "[CLS] E V Q L ... [SEP] D I V M ... [SEP]"
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .base import AntibodyEmbedder, AntibodyEmbedderOutput, BatchedAntibodyEmbedderOutput


class IgBertEmbedder(AntibodyEmbedder):
    """
    IgBert antibody embedder using the Exscientia/IgBert model.

    This model natively supports paired heavy-light chain input using
    a [SEP] separator token between the two chains.

    Args:
        model_name: HuggingFace model ID (default: "Exscientia/IgBert")
        trainable: Whether to allow fine-tuning (default: False)
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        pooling: Pooling strategy for global embedding ('mean', 'cls', 'max')
        max_length: Maximum sequence length (default: 512)

    Example:
        >>> embedder = IgBertEmbedder()
        >>> output = embedder.encode(
        ...     heavy_sequence="EVQLVQSGAEVKKPGAS...",
        ...     light_sequence="DIVMTQSPDSLAVSLGER..."
        ... )
        >>> print(output.global_embedding.shape)
        torch.Size([768])
    """

    def __init__(
        self,
        model_name: str = "Exscientia/IgBert",
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
            from transformers import BertModel, BertTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for IgBertEmbedder. "
                "Install with: pip install transformers"
            )

        # Load tokenizer and model
        # Note: add_pooling_layer=False to get raw hidden states
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        self.model.to(self._device)

        # Set trainability
        if not trainable:
            self.freeze()

        self._embedding_dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def name(self) -> str:
        return "IgBert"

    @property
    def supports_paired(self) -> bool:
        return True

    def _prepare_sequence(self, heavy: str, light: str) -> str:
        """
        Prepare paired sequence in IgBert format.

        IgBert expects space-separated amino acids with [SEP] between chains:
        "[CLS] E V Q L ... [SEP] D I V M ... [SEP]"

        The tokenizer handles [CLS] and final [SEP] automatically.
        """
        # Ensure uppercase
        heavy = heavy.upper().strip()
        light = light.upper().strip()

        # Space-separate amino acids
        heavy_spaced = ' '.join(heavy)
        light_spaced = ' '.join(light)

        # Return as two separate sequences for BERT's sequence pair handling
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
        # Prepare input as sequence pair
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

        # Forward pass
        with torch.set_grad_enabled(self.trainable):
            outputs = self.model(
                **tokens,
                output_attentions=return_attention,
            )

        # Get hidden states: [1, seq_len, d_model]
        hidden_states = outputs.last_hidden_state[0]  # [seq_len, d_model]

        # Find separator positions using token_type_ids
        # token_type_ids: 0 for first sequence (heavy), 1 for second sequence (light)
        token_type_ids = tokens.get('token_type_ids')

        if token_type_ids is not None:
            token_type_ids = token_type_ids[0]
            # Find where token_type changes from 0 to 1
            # This marks the boundary between heavy and light
            type_changes = (token_type_ids[:-1] != token_type_ids[1:]).nonzero(as_tuple=True)[0]

            if len(type_changes) > 0:
                sep_pos = type_changes[0].item() + 1
                # [CLS] heavy_tokens [SEP] light_tokens [SEP]
                heavy_emb = hidden_states[1:sep_pos]  # Skip [CLS], up to first [SEP]
                light_emb = hidden_states[sep_pos + 1:-1]  # After [SEP] to before final [SEP]
            else:
                # Fallback
                h_len = len(heavy_sequence)
                heavy_emb = hidden_states[1:h_len + 1]
                light_emb = hidden_states[h_len + 2:-1]
        else:
            # Alternative: find [SEP] tokens directly
            sep_token_id = self.tokenizer.sep_token_id
            input_ids = tokens['input_ids'][0]
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

            if len(sep_positions) >= 1:
                sep_pos = sep_positions[0].item()
                heavy_emb = hidden_states[1:sep_pos]
                if len(sep_positions) >= 2:
                    light_emb = hidden_states[sep_pos + 1:sep_positions[1].item()]
                else:
                    light_emb = hidden_states[sep_pos + 1:-1]
            else:
                h_len = len(heavy_sequence)
                heavy_emb = hidden_states[1:h_len + 1]
                light_emb = hidden_states[h_len + 2:-1]

        # Truncate to actual sequence lengths
        heavy_emb = heavy_emb[:len(heavy_sequence)]
        light_emb = light_emb[:len(light_sequence)]

        # Compute global embedding
        all_emb = torch.cat([heavy_emb, light_emb], dim=0)
        global_emb = self._pool_embeddings(all_emb)

        # Handle attention weights
        heavy_attn = None
        light_attn = None
        if return_attention and outputs.attentions is not None:
            # Average attention across heads and layers
            attn = torch.stack(outputs.attentions).mean(dim=(0, 2))  # [seq, seq]
            h_len = heavy_emb.shape[0]
            l_len = light_emb.shape[0]
            # Extract relevant attention blocks
            heavy_attn = attn[1:h_len + 1, 1:h_len + 1]
            # Light attention (offset by heavy + 2 for [CLS] and [SEP])
            l_start = h_len + 2
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
        heavy_spaced_list = []
        light_spaced_list = []
        for h, l in zip(heavy_sequences, light_sequences):
            h_sp, l_sp = self._prepare_sequence(h, l)
            heavy_spaced_list.append(h_sp)
            light_spaced_list.append(l_sp)

        # Tokenize batch as sequence pairs
        tokens = self.tokenizer(
            heavy_spaced_list,
            light_spaced_list,
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

        # Find max lengths for output padding
        max_h_len = max(len(h) for h in heavy_sequences)
        max_l_len = max(len(l) for l in light_sequences)

        # Initialize output tensors
        h_emb = torch.zeros(batch_size, max_h_len, self.embedding_dim, device=self._device)
        l_emb = torch.zeros(batch_size, max_l_len, self.embedding_dim, device=self._device)
        g_emb = torch.zeros(batch_size, self.embedding_dim, device=self._device)
        h_mask = torch.zeros(batch_size, max_h_len, dtype=torch.bool, device=self._device)
        l_mask = torch.zeros(batch_size, max_l_len, dtype=torch.bool, device=self._device)

        sep_token_id = self.tokenizer.sep_token_id

        for i in range(batch_size):
            h_len = len(heavy_sequences[i])
            l_len = len(light_sequences[i])

            # Find separator positions
            input_ids = tokens['input_ids'][i]
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]

            if len(sep_positions) >= 1:
                sep_pos = sep_positions[0].item()
                heavy_emb_i = hidden_states[i, 1:sep_pos][:h_len]
                if len(sep_positions) >= 2:
                    light_emb_i = hidden_states[i, sep_pos + 1:sep_positions[1].item()][:l_len]
                else:
                    light_emb_i = hidden_states[i, sep_pos + 1:-1][:l_len]
            else:
                heavy_emb_i = hidden_states[i, 1:h_len + 1]
                light_emb_i = hidden_states[i, h_len + 2:h_len + 2 + l_len]

            # Store with proper lengths
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

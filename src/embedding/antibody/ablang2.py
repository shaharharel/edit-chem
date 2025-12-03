"""
AbLang2 Antibody Embedder.

AbLang2 is an antibody-specific language model developed for predicting
non-germline residues, with support for paired heavy-light chain sequences.

Reference:
    Olsen et al. "Addressing the antibody germline bias and its effect on
    language models for improved antibody design" (2024)
    https://github.com/oxpig/AbLang2
    https://huggingface.co/hemantn/ablang2

Input format:
    AbLang2 expects paired antibody sequences through its adapter interface.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import warnings

from .base import AntibodyEmbedder, AntibodyEmbedderOutput, BatchedAntibodyEmbedderOutput


class AbLang2Embedder(AntibodyEmbedder):
    """
    AbLang2 antibody embedder.

    AbLang2 natively supports paired heavy-light chain input and provides
    both sequence-level and residue-level embeddings.

    Args:
        model_name: HuggingFace model ID (default: "hemantn/ablang2")
        trainable: Whether to allow fine-tuning (default: False)
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        pooling: Pooling strategy for global embedding ('mean', 'cls', 'max')
        max_length: Maximum sequence length (default: 512)

    Example:
        >>> embedder = AbLang2Embedder()
        >>> output = embedder.encode(
        ...     heavy_sequence="EVQLVQSGAEVKKPGAS...",
        ...     light_sequence="DIVMTQSPDSLAVSLGER..."
        ... )
        >>> print(output.global_embedding.shape)

    Note:
        Requires: pip install transformers rotary-embedding-torch
        Optional: conda install -c bioconda anarci (for numbering features)
    """

    def __init__(
        self,
        model_name: str = "hemantn/ablang2",
        trainable: bool = False,
        device: str = 'auto',
        pooling: str = 'mean',
        max_length: int = 512,
    ):
        super().__init__(trainable=trainable, device=device, pooling=pooling)

        self.model_name = model_name
        self.max_length = max_length

        # Try to load AbLang2
        self._load_model()

        # Set trainability
        if not trainable:
            self.freeze()

    def _load_model(self):
        """Load AbLang2 model - try multiple approaches."""
        # First, try the ablang2 package directly
        try:
            import ablang2
            self.ablang = ablang2.pretrained("ablang2-paired")
            # ablang2 has issues with MPS, so keep it on CPU and move outputs to target device
            # The ablang2 package manages its own device internally
            self._use_package = True
            self._embedding_dim = 480  # AbLang2 hidden size (not 768)
            self._ablang_device = 'cpu'  # ablang2 works best on CPU
            return
        except ImportError:
            pass
        except Exception as e:
            # If loading fails for any reason
            pass

        # Try HuggingFace with custom adapter
        try:
            from transformers import AutoModel, AutoTokenizer
            from huggingface_hub import hf_hub_download
            import sys
            import os

            # Load model and tokenizer
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Try to load adapter
            try:
                adapter_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename="adapter.py"
                )
                cached_model_dir = os.path.dirname(adapter_path)
                if cached_model_dir not in sys.path:
                    sys.path.insert(0, cached_model_dir)
                from adapter import AbLang2PairedHuggingFaceAdapter
                self.ablang = AbLang2PairedHuggingFaceAdapter(
                    model=self.model,
                    tokenizer=self.tokenizer
                )
                self._use_package = True
            except Exception:
                # Fall back to direct model usage
                self._use_package = False

            self.model.to(self._device)
            self._embedding_dim = self.model.config.hidden_size
            return

        except Exception as e:
            raise ImportError(
                f"Could not load AbLang2. Error: {e}\n"
                "Install with: pip install ablang2\n"
                "Or: pip install transformers rotary-embedding-torch"
            )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def name(self) -> str:
        return "AbLang2"

    @property
    def supports_paired(self) -> bool:
        return True

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

        if self._use_package and hasattr(self, 'ablang'):
            return self._encode_with_ablang(
                heavy_sequence, light_sequence, return_attention
            )
        else:
            return self._encode_with_transformers(
                heavy_sequence, light_sequence, return_attention
            )

    def _encode_with_ablang(
        self,
        heavy_sequence: str,
        light_sequence: str,
        return_attention: bool = False,
    ) -> AntibodyEmbedderOutput:
        """Encode using ablang2 package."""
        import numpy as np

        # AbLang2 expects paired sequences separated by asterisk
        paired_seq = f"{heavy_sequence}*{light_sequence}"

        # ablang2 runs on CPU internally, we'll move outputs to target device
        with torch.set_grad_enabled(self.trainable):
            # Get residue-level embeddings
            res_emb = self.ablang.rescoding([paired_seq], align=False)

        # res_emb is a list of numpy arrays, convert to tensor and move to target device
        res_emb = torch.from_numpy(np.array(res_emb[0])).float().to(self._device)

        # Split into heavy and light (asterisk is included in tokenization)
        h_len = len(heavy_sequence)
        l_len = len(light_sequence)
        # Total length = h_len + 1 (asterisk) + l_len

        heavy_emb = res_emb[:h_len]
        # Skip the asterisk token at position h_len
        light_emb = res_emb[h_len + 1:h_len + 1 + l_len]

        # Global embedding
        all_emb = torch.cat([heavy_emb, light_emb], dim=0)
        global_emb = self._pool_embeddings(all_emb)

        return AntibodyEmbedderOutput(
            heavy_residue_embeddings=heavy_emb,
            light_residue_embeddings=light_emb,
            global_embedding=global_emb,
            heavy_attention_weights=None,  # AbLang2 doesn't easily expose attention
            light_attention_weights=None,
            heavy_sequence=heavy_sequence,
            light_sequence=light_sequence,
        )

    def _encode_with_transformers(
        self,
        heavy_sequence: str,
        light_sequence: str,
        return_attention: bool = False,
    ) -> AntibodyEmbedderOutput:
        """Encode using direct transformers model."""
        # Combine sequences with separator
        combined = f"{heavy_sequence}|{light_sequence}"

        tokens = self.tokenizer(
            combined,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self._device) for k, v in tokens.items()}

        with torch.set_grad_enabled(self.trainable):
            outputs = self.model(
                **tokens,
                output_attentions=return_attention,
            )

        hidden_states = outputs.last_hidden_state[0]

        # Find separator and split
        h_len = len(heavy_sequence)
        l_len = len(light_sequence)

        # Assuming format: [CLS] H... | L... [SEP]
        # Adjust based on actual tokenization
        heavy_emb = hidden_states[1:h_len + 1]
        light_emb = hidden_states[h_len + 2:h_len + 2 + l_len]  # +2 for | separator

        # Handle potential length mismatches
        heavy_emb = heavy_emb[:h_len]
        light_emb = light_emb[:l_len]

        # Pad if needed
        if heavy_emb.shape[0] < h_len:
            pad = torch.zeros(h_len - heavy_emb.shape[0], self.embedding_dim, device=self._device)
            heavy_emb = torch.cat([heavy_emb, pad], dim=0)
        if light_emb.shape[0] < l_len:
            pad = torch.zeros(l_len - light_emb.shape[0], self.embedding_dim, device=self._device)
            light_emb = torch.cat([light_emb, pad], dim=0)

        # Global embedding
        all_emb = torch.cat([heavy_emb[:h_len], light_emb[:l_len]], dim=0)
        global_emb = self._pool_embeddings(all_emb)

        return AntibodyEmbedderOutput(
            heavy_residue_embeddings=heavy_emb,
            light_residue_embeddings=light_emb,
            global_embedding=global_emb,
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

        # Normalize
        heavy_sequences = [h.upper().strip() for h in heavy_sequences]
        light_sequences = [l.upper().strip() for l in light_sequences]

        if self._use_package and hasattr(self, 'ablang'):
            return self._encode_batch_with_ablang(
                heavy_sequences, light_sequences, return_attention
            )

        # Default: call encode in loop
        return super().encode_batch(heavy_sequences, light_sequences, return_attention)

    def _encode_batch_with_ablang(
        self,
        heavy_sequences: List[str],
        light_sequences: List[str],
        return_attention: bool = False,
    ) -> BatchedAntibodyEmbedderOutput:
        """Batch encode using ablang2 package."""
        import numpy as np

        batch_size = len(heavy_sequences)

        # Prepare input for ablang2 - use asterisk separator
        paired_sequences = [f"{h}*{l}" for h, l in zip(heavy_sequences, light_sequences)]

        # ablang2 runs on CPU internally
        with torch.set_grad_enabled(self.trainable):
            res_emb = self.ablang.rescoding(paired_sequences, align=False)

        # Find max lengths
        max_h_len = max(len(h) for h in heavy_sequences)
        max_l_len = max(len(l) for l in light_sequences)

        # Initialize output tensors on target device
        h_emb = torch.zeros(batch_size, max_h_len, self.embedding_dim, device=self._device)
        l_emb = torch.zeros(batch_size, max_l_len, self.embedding_dim, device=self._device)
        g_emb = torch.zeros(batch_size, self.embedding_dim, device=self._device)
        h_mask = torch.zeros(batch_size, max_h_len, dtype=torch.bool, device=self._device)
        l_mask = torch.zeros(batch_size, max_l_len, dtype=torch.bool, device=self._device)

        for i in range(batch_size):
            h_len = len(heavy_sequences[i])
            l_len = len(light_sequences[i])

            # Convert numpy array to tensor and move to target device
            emb_i = torch.from_numpy(np.array(res_emb[i])).float().to(self._device)

            # Split at asterisk position
            heavy_emb_i = emb_i[:h_len]
            light_emb_i = emb_i[h_len + 1:h_len + 1 + l_len]  # +1 to skip asterisk

            h_emb[i, :h_len] = heavy_emb_i
            l_emb[i, :l_len] = light_emb_i
            h_mask[i, :h_len] = True
            l_mask[i, :l_len] = True

            # Global embedding
            all_emb = torch.cat([heavy_emb_i, light_emb_i], dim=0)
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

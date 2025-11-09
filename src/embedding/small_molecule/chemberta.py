"""
ChemBERTa-based molecule embeddings.

Uses transformer models pre-trained on SMILES:
- ChemBERTa (seyonec/ChemBERTa-zinc-base-v1)
- MolBERT
- SMILES-BERT
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Optional
from .base import MoleculeEmbedder


class ChemBERTaEmbedder(MoleculeEmbedder):
    """
    ChemBERTa-based molecule embedder.

    Uses transformer models trained on SMILES strings.

    Args:
        model_name: HuggingFace model name or path
        pooling: Pooling strategy ('mean', 'cls', 'max')
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for encoding multiple molecules
    """

    DEFAULT_MODELS = {
        'chemberta': 'seyonec/ChemBERTa-zinc-base-v1',
        'chemberta-large': 'seyonec/ChemBERTa-zinc-large-v1',
        'molbert': 'Danhup/MolBERT',
    }

    def __init__(
        self,
        model_name: str = 'chemberta',
        pooling: str = 'mean',
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        # Resolve model name
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]

        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model and tokenizer
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Cache embedding dimension
        self._embedding_dim = self.model.config.hidden_size

    def encode(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Encode molecule(s) to embedding vector(s).

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Embedding vector(s) as numpy array
        """
        if isinstance(smiles, str):
            smiles = [smiles]
            return_single = True
        else:
            return_single = False

        # Batch encode
        all_embeddings = []
        for i in range(0, len(smiles), self.batch_size):
            batch = smiles[i:i + self.batch_size]
            batch_emb = self._encode_batch(batch)
            all_embeddings.append(batch_emb)

        embeddings = np.vstack(all_embeddings)

        if return_single:
            return embeddings[0]
        else:
            return embeddings

    def _encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Encode a batch of SMILES."""
        # Tokenize
        inputs = self.tokenizer(
            smiles_list,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Forward pass (no gradients)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Pool hidden states
        if self.pooling == 'mean':
            # Mean pool over sequence length (ignoring padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        elif self.pooling == 'cls':
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'max':
            # Max pool over sequence length
            embeddings = outputs.last_hidden_state.max(dim=1)[0]
        else:
            raise ValueError(f"Invalid pooling: {self.pooling}")

        return embeddings.cpu().numpy()

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this embedding method."""
        model_short = self.model_name.split('/')[-1]
        return f"chemberta_{model_short}_{self.pooling}"


# Convenience constructors
def chemberta_embedder(pooling: str = 'mean', device: Optional[str] = None) -> ChemBERTaEmbedder:
    """
    Create ChemBERTa base embedder.

    Args:
        pooling: Pooling strategy ('mean', 'cls', 'max')
        device: Device to run on
    """
    return ChemBERTaEmbedder(model_name='chemberta', pooling=pooling, device=device)


def chemberta_large_embedder(pooling: str = 'mean', device: Optional[str] = None) -> ChemBERTaEmbedder:
    """Create ChemBERTa large embedder (better quality, slower)."""
    return ChemBERTaEmbedder(model_name='chemberta-large', pooling=pooling, device=device)

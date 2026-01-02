"""
PyTorch Dataset for ADAR editing site prediction.

Provides datasets for binary classification: edited vs non-edited adenosines.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ADAREditingDataset(Dataset):
    """
    PyTorch Dataset for ADAR editing site classification.

    Each sample consists of:
    - RNA sequence window centered on an adenosine
    - Binary label: 1 = edited, 0 = non-edited

    This dataset is designed for:
    1. Sequence-only models (provide sequence as string)
    2. Embedding-based models (provide pre-computed embeddings)

    Args:
        data_df: DataFrame with columns:
            - sequence: RNA sequence window
            - is_edited: Binary label (0 or 1)
            - Optional: coverage, editing_rate, strand, etc.
        embeddings: Optional pre-computed sequence embeddings
        embedder: Optional embedder for on-the-fly embedding
        include_metadata: Include additional features (coverage, strand)

    Example:
        >>> dataset = ADAREditingDataset(data_df)
        >>> seq, label = dataset[0]  # Returns (sequence_str, label_tensor)

        >>> # With embeddings
        >>> dataset = ADAREditingDataset(data_df, embeddings=precomputed)
        >>> emb, label = dataset[0]  # Returns (embedding_tensor, label_tensor)
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None,
        embedder=None,
        include_metadata: bool = False,
        return_sequences: bool = False,
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.embeddings = embeddings
        self.embedder = embedder
        self.include_metadata = include_metadata
        self.return_sequences = return_sequences

        self.precomputed = embeddings is not None

        # Validate
        if 'sequence' not in self.data_df.columns:
            raise ValueError("DataFrame must have 'sequence' column")
        if 'is_edited' not in self.data_df.columns:
            raise ValueError("DataFrame must have 'is_edited' column")

    def __len__(self) -> int:
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        row = self.data_df.iloc[idx]
        sequence = row['sequence']
        label = torch.FloatTensor([row['is_edited']])

        # Get embedding or sequence
        if self.precomputed:
            emb = torch.FloatTensor(self.embeddings[idx])
        elif self.embedder is not None:
            emb = torch.FloatTensor(self.embedder.encode(sequence))
        else:
            # Return sequence directly (for models that embed internally)
            if self.return_sequences:
                return sequence, label
            else:
                raise ValueError("No embeddings or embedder provided")

        if self.include_metadata:
            # Additional features
            meta = torch.FloatTensor([
                row.get('coverage', 0) / 1000,  # Normalize coverage
                1.0 if row.get('strand', '+') == '+' else 0.0,
            ])
            return emb, meta, label

        return emb, label

    def get_sequences(self) -> List[str]:
        """Get all sequences for batch embedding."""
        return self.data_df['sequence'].tolist()

    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return self.data_df['is_edited'].values


class ADARPairDataset(Dataset):
    """
    Dataset for contrastive/pair-based learning.

    Creates pairs of (edited, non-edited) positions for contrastive learning
    or edit-focused representations.

    Args:
        data_df: DataFrame with edited and non-edited sites
        embedder: Embedder for sequence encoding
        pair_strategy: 'random' or 'matched' (by coverage, strand, etc.)
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        embedder=None,
        embeddings: Optional[np.ndarray] = None,
        pair_strategy: str = 'random',
    ):
        self.embedder = embedder
        self.embeddings = embeddings
        self.precomputed = embeddings is not None

        # Split into edited and non-edited
        self.edited = data_df[data_df['is_edited'] == 1].reset_index(drop=True)
        self.non_edited = data_df[data_df['is_edited'] == 0].reset_index(drop=True)

        if len(self.edited) == 0 or len(self.non_edited) == 0:
            raise ValueError("Need both edited and non-edited samples")

        self.pair_strategy = pair_strategy

        if pair_strategy == 'matched':
            self._create_matched_pairs()
        else:
            # Random pairing: use all edited, randomly sample non-edited
            self.n_pairs = len(self.edited)

    def _create_matched_pairs(self):
        """Create matched pairs based on coverage and strand."""
        # TODO: Implement coverage/strand matching
        self.n_pairs = len(self.edited)

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # Get edited sample
        edited_row = self.edited.iloc[idx]
        edited_seq = edited_row['sequence']

        # Get non-edited sample
        if self.pair_strategy == 'random':
            non_edited_idx = np.random.randint(len(self.non_edited))
        else:
            non_edited_idx = idx % len(self.non_edited)

        non_edited_row = self.non_edited.iloc[non_edited_idx]
        non_edited_seq = non_edited_row['sequence']

        # Get embeddings
        if self.precomputed:
            edited_emb = torch.FloatTensor(self.embeddings[idx])
            non_edited_emb = torch.FloatTensor(
                self.embeddings[len(self.edited) + non_edited_idx]
            )
        elif self.embedder is not None:
            edited_emb = torch.FloatTensor(self.embedder.encode(edited_seq))
            non_edited_emb = torch.FloatTensor(self.embedder.encode(non_edited_seq))
        else:
            raise ValueError("No embeddings or embedder provided")

        return edited_emb, non_edited_emb


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    embedder=None,
    embeddings_train: Optional[np.ndarray] = None,
    embeddings_val: Optional[np.ndarray] = None,
    embeddings_test: Optional[np.ndarray] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for ADAR editing prediction.

    Args:
        train_df: Training data DataFrame
        val_df: Validation data DataFrame
        test_df: Optional test data DataFrame
        embedder: RNA embedder (if not using pre-computed)
        embeddings_*: Pre-computed embeddings
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        Dict with 'train', 'val', and optionally 'test' DataLoaders
    """
    from torch.utils.data import DataLoader

    train_dataset = ADAREditingDataset(
        train_df,
        embeddings=embeddings_train,
        embedder=embedder,
    )

    val_dataset = ADAREditingDataset(
        val_df,
        embeddings=embeddings_val,
        embedder=embedder,
    )

    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    if test_df is not None:
        test_dataset = ADAREditingDataset(
            test_df,
            embeddings=embeddings_test,
            embedder=embedder,
        )
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders

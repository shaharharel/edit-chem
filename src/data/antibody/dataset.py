"""
PyTorch Dataset classes for antibody mutation data.

Provides Dataset implementations for training antibody effect predictors.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Dict, Any, Tuple, Callable
import numpy as np

from .schema import AbEditPair, AbEditPairsDataset, AbMutation


class AbPairDataset(Dataset):
    """
    PyTorch Dataset for antibody mutation pairs.

    This dataset provides samples for training edit effect predictors.
    Each sample contains wild-type and mutant sequences along with
    the measured delta value.

    Args:
        pairs: AbEditPairsDataset or list of AbEditPair
        transform: Optional transform to apply to each sample
        include_metadata: Whether to include metadata in samples
        cache_embeddings: Whether to cache embeddings (for pre-computed)

    Example:
        >>> dataset = AbPairDataset.from_loader('abbibench', data_dir='data/')
        >>> sample = dataset[0]
        >>> print(sample['heavy_wt'], sample['delta_value'])
    """

    def __init__(
        self,
        pairs: AbEditPairsDataset,
        transform: Optional[Callable] = None,
        include_metadata: bool = False,
        cache_embeddings: bool = False,
    ):
        if isinstance(pairs, AbEditPairsDataset):
            self.pairs = pairs.pairs
        else:
            self.pairs = list(pairs)

        self.transform = transform
        self.include_metadata = include_metadata
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: Dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]

        sample = {
            'antibody_id': pair.antibody_id,
            'heavy_wt': pair.heavy_wt,
            'light_wt': pair.light_wt,
            'heavy_mut': pair.heavy_mut,
            'light_mut': pair.light_mut,
            'mutations': self._encode_mutations(pair.mutations),
            'num_mutations': pair.num_mutations,
            'delta_value': pair.delta_value,
            'assay_type': pair.assay_type.value,
        }

        # Add optional fields
        if pair.antigen_id:
            sample['antigen_id'] = pair.antigen_id
        if pair.structure_id:
            sample['structure_id'] = pair.structure_id

        if self.include_metadata:
            sample['metadata'] = pair.metadata
            sample['source_dataset'] = pair.source_dataset

        # Check for cached embeddings
        if self.cache_embeddings and idx in self._embedding_cache:
            sample['cached_embeddings'] = self._embedding_cache[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _encode_mutations(self, mutations: List[AbMutation]) -> List[Dict[str, Any]]:
        """Encode mutations as list of dicts."""
        return [
            {
                'chain': m.chain,
                'position': m.position,
                'from_aa': m.from_aa,
                'to_aa': m.to_aa,
                'imgt_position': m.imgt_position,
                'region': m.region,
            }
            for m in mutations
        ]

    def cache_embedding(self, idx: int, embedding: Any):
        """Cache a pre-computed embedding for a sample."""
        self._embedding_cache[idx] = embedding

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    @classmethod
    def from_loader(
        cls,
        loader_name: str,
        data_dir: str,
        **loader_kwargs,
    ) -> 'AbPairDataset':
        """
        Create dataset using a named loader.

        Args:
            loader_name: Name of loader ('abbibench', 'abagym', 'ab_bind', 'skempi2')
            data_dir: Directory containing data
            **loader_kwargs: Additional arguments for the loader

        Returns:
            AbPairDataset instance
        """
        from .loaders import (
            load_abbibench, load_abagym, load_ab_bind, load_skempi2_antibodies
        )

        loaders = {
            'abbibench': load_abbibench,
            'abagym': load_abagym,
            'ab_bind': load_ab_bind,
            'skempi2': load_skempi2_antibodies,
        }

        if loader_name not in loaders:
            raise ValueError(f"Unknown loader: {loader_name}. Available: {list(loaders.keys())}")

        pairs = loaders[loader_name](data_dir, **loader_kwargs)
        return cls(pairs)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        by_antibody: bool = True,
    ) -> Tuple['AbPairDataset', 'AbPairDataset', 'AbPairDataset']:
        """
        Split dataset into train/val/test.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for test
            seed: Random seed
            by_antibody: If True, split by antibody ID to prevent leakage

        Returns:
            Tuple of (train, val, test) AbPairDataset
        """
        dataset = AbEditPairsDataset(self.pairs)
        train, val, test = dataset.split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            by_antibody=by_antibody,
        )

        return (
            AbPairDataset(train, transform=self.transform, include_metadata=self.include_metadata),
            AbPairDataset(val, transform=self.transform, include_metadata=self.include_metadata),
            AbPairDataset(test, transform=self.transform, include_metadata=self.include_metadata),
        )

    def filter(self, **filter_kwargs) -> 'AbPairDataset':
        """Filter dataset and return new AbPairDataset."""
        dataset = AbEditPairsDataset(self.pairs)
        filtered = dataset.filter(**filter_kwargs)
        return AbPairDataset(
            filtered,
            transform=self.transform,
            include_metadata=self.include_metadata,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return AbEditPairsDataset(self.pairs).get_statistics()


class AbPairCollator:
    """
    Collator for batching AbPairDataset samples.

    Handles variable-length sequences and mutation lists.

    Args:
        embedder: Optional antibody embedder for on-the-fly embedding
        max_mutations: Maximum number of mutations to include per sample
        pad_token: Padding token for sequences
    """

    def __init__(
        self,
        embedder=None,
        max_mutations: int = 10,
        pad_token: str = 'X',
    ):
        self.embedder = embedder
        self.max_mutations = max_mutations
        self.pad_token = pad_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a batch of samples."""
        batch_size = len(batch)

        # Collect sequences
        heavy_wt = [s['heavy_wt'] for s in batch]
        light_wt = [s['light_wt'] for s in batch]
        heavy_mut = [s['heavy_mut'] for s in batch]
        light_mut = [s['light_mut'] for s in batch]

        # Collect delta values
        delta_values = torch.tensor(
            [s['delta_value'] for s in batch],
            dtype=torch.float32
        )

        # Collect mutations
        mutations = []
        mutation_mask = []

        for s in batch:
            sample_muts = s['mutations'][:self.max_mutations]
            mutations.append(sample_muts)
            mutation_mask.append([True] * len(sample_muts) + [False] * (self.max_mutations - len(sample_muts)))

        # Pad mutations
        padded_mutations = self._pad_mutations(mutations)
        mutation_mask = torch.tensor(mutation_mask, dtype=torch.bool)

        collated = {
            'heavy_wt': heavy_wt,
            'light_wt': light_wt,
            'heavy_mut': heavy_mut,
            'light_mut': light_mut,
            'delta_value': delta_values,
            'mutations': padded_mutations,
            'mutation_mask': mutation_mask,
            'num_mutations': torch.tensor([s['num_mutations'] for s in batch]),
            'antibody_id': [s['antibody_id'] for s in batch],
            'assay_type': [s['assay_type'] for s in batch],
        }

        # Embed if embedder provided
        if self.embedder is not None:
            collated['embeddings'] = self._embed_batch(heavy_wt, light_wt, heavy_mut, light_mut)

        return collated

    def _pad_mutations(
        self,
        mutations: List[List[Dict[str, Any]]],
    ) -> Dict[str, torch.Tensor]:
        """Pad mutation lists to same length."""
        batch_size = len(mutations)

        # Initialize tensors
        chains = torch.zeros(batch_size, self.max_mutations, dtype=torch.long)
        positions = torch.zeros(batch_size, self.max_mutations, dtype=torch.long)
        from_aas = torch.zeros(batch_size, self.max_mutations, dtype=torch.long)
        to_aas = torch.zeros(batch_size, self.max_mutations, dtype=torch.long)

        aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        chain_to_idx = {'H': 0, 'L': 1, 'K': 1}

        for i, sample_muts in enumerate(mutations):
            for j, mut in enumerate(sample_muts):
                if j >= self.max_mutations:
                    break
                chains[i, j] = chain_to_idx.get(mut['chain'], 0)
                positions[i, j] = mut['position']
                from_aas[i, j] = aa_to_idx.get(mut['from_aa'], 0)
                to_aas[i, j] = aa_to_idx.get(mut['to_aa'], 0)

        return {
            'chain': chains,
            'position': positions,
            'from_aa': from_aas,
            'to_aa': to_aas,
        }

    def _embed_batch(
        self,
        heavy_wt: List[str],
        light_wt: List[str],
        heavy_mut: List[str],
        light_mut: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Embed sequences using the embedder."""
        # Embed wild-type
        wt_output = self.embedder.encode_batch(heavy_wt, light_wt)

        # Embed mutant
        mut_output = self.embedder.encode_batch(heavy_mut, light_mut)

        return {
            'wt_heavy': wt_output.heavy_residue_embeddings,
            'wt_light': wt_output.light_residue_embeddings,
            'wt_global': wt_output.global_embeddings,
            'mut_heavy': mut_output.heavy_residue_embeddings,
            'mut_light': mut_output.light_residue_embeddings,
            'mut_global': mut_output.global_embeddings,
            'wt_heavy_mask': wt_output.heavy_mask,
            'wt_light_mask': wt_output.light_mask,
            'mut_heavy_mask': mut_output.heavy_mask,
            'mut_light_mask': mut_output.light_mask,
        }


def create_dataloaders(
    dataset: AbPairDataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    embedder=None,
    by_antibody: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        dataset: AbPairDataset to split
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        num_workers: Number of data loading workers
        embedder: Optional embedder for on-the-fly embedding
        by_antibody: Whether to split by antibody ID

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = dataset.split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        by_antibody=by_antibody,
    )

    collator = AbPairCollator(embedder=embedder)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return train_loader, val_loader, test_loader

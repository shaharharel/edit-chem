"""
Data loading utilities for RNA experiments.

Handles loading MPRA pairs data and preparing train/val/test splits.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from src.utils.rna_splits import get_rna_splitter
from src.data.rna.sequence_utils import validate_rna_sequence


def load_pairs_data(
    data_file: str,
    property_filter: Optional[List[str]] = None,
    min_pairs_per_property: int = 0
) -> pd.DataFrame:
    """
    Load pairs data from CSV file.

    Args:
        data_file: Path to pairs CSV file
        property_filter: Optional list of property names to keep
        min_pairs_per_property: Minimum pairs required per property

    Returns:
        Filtered pairs DataFrame
    """
    print(f"Loading pairs data from: {data_file}")

    df = pd.read_csv(data_file)
    print(f"  Loaded {len(df):,} pairs")

    # Validate sequences
    valid_mask = df.apply(
        lambda row: validate_rna_sequence(str(row.get('seq_a', '')), allow_n=True) and
                    validate_rna_sequence(str(row.get('seq_b', '')), allow_n=True),
        axis=1
    )
    df = df[valid_mask]
    print(f"  Valid sequences: {len(df):,} pairs")

    # Filter by property
    if property_filter is not None:
        df = df[df['property_name'].isin(property_filter)]
        print(f"  After property filter: {len(df):,} pairs")

    # Filter by minimum pairs per property
    if min_pairs_per_property > 0:
        property_counts = df['property_name'].value_counts()
        valid_properties = property_counts[property_counts >= min_pairs_per_property].index
        df = df[df['property_name'].isin(valid_properties)]
        print(f"  After min pairs filter: {len(df):,} pairs")

    # Print property distribution
    print("\n  Property distribution:")
    for prop, count in df['property_name'].value_counts().items():
        print(f"    {prop}: {count:,} pairs")

    return df


def split_data(
    pairs_df: pd.DataFrame,
    splitter_type: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    **splitter_params
) -> Dict[str, pd.DataFrame]:
    """
    Split pairs data into train/val/test sets.

    Args:
        pairs_df: Pairs DataFrame
        splitter_type: Type of splitter ('random', 'sequence_similarity', etc.)
        train_ratio, val_ratio, test_ratio: Split ratios
        random_seed: Random seed
        **splitter_params: Additional splitter parameters

    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    print(f"\nSplitting data using {splitter_type} splitter...")

    if splitter_type == 'random':
        # Random split
        n = len(pairs_df)
        indices = np.arange(n)
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = pairs_df.iloc[indices[:train_end]].reset_index(drop=True)
        val_df = pairs_df.iloc[indices[train_end:val_end]].reset_index(drop=True)
        test_df = pairs_df.iloc[indices[val_end:]].reset_index(drop=True)

    elif splitter_type == 'sequence_similarity':
        # Split by sequence similarity (ensure dissimilar sequences in test)
        train_df, val_df, test_df = _split_by_sequence_similarity(
            pairs_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_seed=random_seed,
            **splitter_params
        )

    elif splitter_type == 'edit_type':
        # Split by edit type
        train_df, val_df, test_df = _split_by_edit_type(
            pairs_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            random_seed=random_seed,
            **splitter_params
        )

    elif splitter_type in ['motif', 'gc_stratified', 'length_stratified']:
        # Use RNA-specific splitters
        splitter = get_rna_splitter(
            split_type=splitter_type,
            train_size=train_ratio,
            val_size=val_ratio,
            test_size=test_ratio,
            random_state=random_seed,
            **splitter_params
        )
        train_df, val_df, test_df = splitter.split(pairs_df, seq_col='seq_a')

    else:
        # Fallback to random split
        print(f"  Warning: Unknown splitter type '{splitter_type}', using random split")
        n = len(pairs_df)
        indices = np.arange(n)
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_df = pairs_df.iloc[indices[:train_end]].reset_index(drop=True)
        val_df = pairs_df.iloc[indices[train_end:val_end]].reset_index(drop=True)
        test_df = pairs_df.iloc[indices[val_end:]].reset_index(drop=True)

    print(f"  Train: {len(train_df):,} pairs")
    print(f"  Val: {len(val_df):,} pairs")
    print(f"  Test: {len(test_df):,} pairs")

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def _split_by_sequence_similarity(
    pairs_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
    similarity_threshold: float = 0.7,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ensuring test sequences are dissimilar to training sequences.

    Uses sequence clustering to group similar sequences together.
    """
    from collections import Counter

    # Get unique sequences
    all_seqs = set(pairs_df['seq_a'].tolist() + pairs_df['seq_b'].tolist())
    unique_seqs = list(all_seqs)

    print(f"  Unique sequences: {len(unique_seqs)}")

    # Simple clustering by k-mer overlap
    def kmer_set(seq, k=4):
        return set(seq[i:i+k] for i in range(len(seq) - k + 1))

    # Group sequences by length (only compare same-length)
    seqs_by_length = defaultdict(list)
    for seq in unique_seqs:
        seqs_by_length[len(seq)].append(seq)

    # Cluster sequences
    np.random.seed(random_seed)
    clusters = []

    for length, seqs in seqs_by_length.items():
        if len(seqs) == 1:
            clusters.append(seqs)
            continue

        # Compute k-mer sets
        kmer_sets = {seq: kmer_set(seq) for seq in seqs}

        # Greedy clustering
        remaining = set(seqs)
        while remaining:
            # Pick random seed
            seed = np.random.choice(list(remaining))
            cluster = [seed]
            remaining.remove(seed)

            seed_kmers = kmer_sets[seed]

            for other in list(remaining):
                other_kmers = kmer_sets[other]
                overlap = len(seed_kmers & other_kmers) / max(len(seed_kmers | other_kmers), 1)

                if overlap >= similarity_threshold:
                    cluster.append(other)
                    remaining.remove(other)

            clusters.append(cluster)

    print(f"  Created {len(clusters)} sequence clusters")

    # Assign clusters to splits
    np.random.shuffle(clusters)

    target_train = int(len(unique_seqs) * train_ratio)
    target_val = int(len(unique_seqs) * val_ratio)

    train_seqs = set()
    val_seqs = set()
    test_seqs = set()

    train_count = 0
    val_count = 0

    for cluster in clusters:
        if train_count < target_train:
            train_seqs.update(cluster)
            train_count += len(cluster)
        elif val_count < target_val:
            val_seqs.update(cluster)
            val_count += len(cluster)
        else:
            test_seqs.update(cluster)

    # Split pairs based on sequence assignment
    def pair_split(row):
        seq_a, seq_b = row['seq_a'], row['seq_b']
        if seq_a in train_seqs or seq_b in train_seqs:
            return 'train'
        elif seq_a in val_seqs or seq_b in val_seqs:
            return 'val'
        else:
            return 'test'

    pairs_df['_split'] = pairs_df.apply(pair_split, axis=1)

    train_df = pairs_df[pairs_df['_split'] == 'train'].drop('_split', axis=1).reset_index(drop=True)
    val_df = pairs_df[pairs_df['_split'] == 'val'].drop('_split', axis=1).reset_index(drop=True)
    test_df = pairs_df[pairs_df['_split'] == 'test'].drop('_split', axis=1).reset_index(drop=True)

    return train_df, val_df, test_df


def _split_by_edit_type(
    pairs_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
    test_edit_types: Optional[List[str]] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by edit type (train on some edit types, test on others).

    Args:
        test_edit_types: Edit types to hold out for testing
    """
    if test_edit_types is None:
        # Default: hold out 'complex' and 'deletion' for testing
        test_edit_types = ['complex', 'deletion']

    # Split by edit type
    train_val_df = pairs_df[~pairs_df['edit_type'].isin(test_edit_types)]
    test_df = pairs_df[pairs_df['edit_type'].isin(test_edit_types)].reset_index(drop=True)

    print(f"  Held out edit types for test: {test_edit_types}")

    # Split train_val into train and val
    n = len(train_val_df)
    indices = np.arange(n)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    val_size = int(n * val_ratio / (train_ratio + val_ratio))
    train_df = train_val_df.iloc[indices[val_size:]].reset_index(drop=True)
    val_df = train_val_df.iloc[indices[:val_size]].reset_index(drop=True)

    return train_df, val_df, test_df


def load_datasets(
    config
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load and split data according to experiment config.

    Args:
        config: RNAExperimentConfig

    Returns:
        Dict mapping property_name -> {'train': df, 'val': df, 'test': df}
    """
    # Load all pairs
    pairs_df = load_pairs_data(
        data_file=config.data_file,
        property_filter=config.property_filter,
        min_pairs_per_property=config.min_pairs_per_property
    )

    # Get splitter params
    splitter_params = config.splitter_params.get(config.splitter_type, {})

    # Split data
    splits = split_data(
        pairs_df=pairs_df,
        splitter_type=config.splitter_type,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
        **splitter_params
    )

    # Organize by property (for multi-task)
    datasets = {}
    for property_name in pairs_df['property_name'].unique():
        datasets[property_name] = {
            'train': splits['train'][splits['train']['property_name'] == property_name].copy(),
            'val': splits['val'][splits['val']['property_name'] == property_name].copy(),
            'test': splits['test'][splits['test']['property_name'] == property_name].copy()
        }

    return datasets

#!/usr/bin/env python3
"""
Pre-compute embeddings and graphs for ADAR dataset.

Generates:
1. Structure graphs (PyG Data objects) - for GNN models
2. RNA-FM sequence embeddings - for sequence-based models
3. Hand-crafted features - for baseline and hybrid models

These are saved to disk and can be loaded quickly during experiments.
"""

import sys
from pathlib import Path
import argparse
import logging
import time
import pickle
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.adar.models import dot_bracket_to_graph
from experiments.adar.data_loader import (
    extract_sequence_features, extract_structure_features
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_graphs_sharded(
    structures: List[str],
    sequences: List[str],
    output_dir: Path,
    shard_size: int = 100_000
) -> int:
    """
    Pre-compute PyG graph objects and save in shards to avoid memory explosion.

    Args:
        structures: List of dot-bracket structure strings
        sequences: List of RNA sequences
        output_dir: Directory to save shards
        shard_size: Number of graphs per shard file

    Returns:
        Total number of graphs computed
    """
    graphs_dir = output_dir / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)

    total_graphs = 0
    shard_idx = 0
    current_shard = []

    for i in tqdm(range(len(structures)), desc="Computing graphs"):
        graph = dot_bracket_to_graph(structures[i], sequences[i])
        current_shard.append(graph)
        total_graphs += 1

        # Save shard when full
        if len(current_shard) >= shard_size:
            shard_file = graphs_dir / f'shard_{shard_idx:04d}.pkl'
            with open(shard_file, 'wb') as f:
                pickle.dump(current_shard, f)
            logger.info(f"Saved shard {shard_idx} with {len(current_shard)} graphs")
            current_shard = []
            shard_idx += 1

    # Save remaining graphs
    if current_shard:
        shard_file = graphs_dir / f'shard_{shard_idx:04d}.pkl'
        with open(shard_file, 'wb') as f:
            pickle.dump(current_shard, f)
        logger.info(f"Saved final shard {shard_idx} with {len(current_shard)} graphs")

    # Save metadata
    meta = {'total_graphs': total_graphs, 'shard_size': shard_size, 'num_shards': shard_idx + 1}
    with open(graphs_dir / 'metadata.json', 'w') as f:
        import json
        json.dump(meta, f)

    return total_graphs


def precompute_rnafm_embeddings(
    sequences: List[str],
    batch_size: int = 32,
    pooling: str = "mean",
    device: str = "auto"
) -> np.ndarray:
    """
    Pre-compute RNA-FM embeddings for sequences.

    Args:
        sequences: List of RNA sequences
        batch_size: Batch size for encoding
        pooling: Pooling strategy (mean, max, cls)
        device: Device to use (auto, cuda, mps, cpu)

    Returns:
        Embeddings array [N, 640]
    """
    try:
        from src.embedding.rna import RNAFMEmbedder
    except ImportError:
        logger.warning("RNA-FM embedder not available, skipping")
        return None

    logger.info("Loading RNA-FM model...")
    embedder = RNAFMEmbedder(pooling=pooling)

    logger.info(f"Encoding {len(sequences)} sequences...")
    embeddings = embedder.encode(sequences)

    return embeddings


def precompute_features(
    df: pd.DataFrame
) -> Tuple[np.ndarray, List[str]]:
    """
    Pre-compute hand-crafted features.

    Args:
        df: DataFrame with 'sequence' and 'structure' columns

    Returns:
        Feature matrix and feature names
    """
    logger.info("Extracting sequence features...")
    seq_features = extract_sequence_features(df['sequence'])

    logger.info("Extracting structure features...")
    struct_features = extract_structure_features(df['structure'])

    # Combine
    all_features = pd.concat([seq_features, struct_features], axis=1)

    # Add metadata features
    all_features['log_coverage'] = np.log1p(df['coverage'].values)
    all_features['mfe'] = df['mfe'].values

    # Fill NaN
    all_features = all_features.fillna(0)

    return all_features.values, all_features.columns.tolist()


def main():
    parser = argparse.ArgumentParser(description='Pre-compute ADAR embeddings')
    parser.add_argument('--input', type=str,
                        default='data/rna/adar/adar_editing_dataset_with_structure.csv',
                        help='Input CSV file')
    parser.add_argument('--output-dir', type=str,
                        default='data/rna/adar/precomputed',
                        help='Output directory')
    parser.add_argument('--sample', type=int, default=None,
                        help='Only process first N samples (for testing)')
    parser.add_argument('--n-positives', type=int, default=None,
                        help='Number of positive samples (None = all)')
    parser.add_argument('--n-negatives', type=int, default=500_000,
                        help='Number of negative samples')
    parser.add_argument('--skip-rnafm', action='store_true',
                        help='Skip RNA-FM embedding computation')
    parser.add_argument('--skip-graphs', action='store_true',
                        help='Skip graph computation')
    parser.add_argument('--skip-features', action='store_true',
                        help='Skip feature computation')
    parser.add_argument('--no-sampling', action='store_true',
                        help='Use full dataset without any sampling')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Full dataset: {len(df):,} samples")

    # Sample if requested (skip if --no-sampling is set)
    if args.no_sampling:
        logger.info("Using full dataset (no sampling)")
    elif args.sample:
        df = df.head(args.sample)
        logger.info(f"Sampled to {len(df)} samples")
    elif args.n_positives is not None or args.n_negatives < len(df[df['is_edited']==0]):
        positives = df[df['is_edited'] == 1]
        negatives = df[df['is_edited'] == 0]

        if args.n_positives is not None:
            positives = positives.sample(
                n=min(args.n_positives, len(positives)),
                random_state=42
            )

        if args.n_negatives < len(negatives):
            negatives = negatives.sample(n=args.n_negatives, random_state=42)

        df = pd.concat([positives, negatives]).sample(frac=1, random_state=42)
        logger.info(f"Sampled to {len(df):,} samples ({len(positives):,} pos, {len(negatives):,} neg)")

    # Save sampled indices
    df = df.reset_index(drop=True)

    # Compute graphs (sharded to avoid memory explosion)
    if not args.skip_graphs:
        logger.info("Pre-computing structure graphs (sharded)...")
        start = time.time()
        n_graphs = precompute_graphs_sharded(
            df['structure'].tolist(),
            df['sequence'].tolist(),
            output_dir,
            shard_size=100_000  # 100K graphs per shard file
        )
        elapsed = time.time() - start
        logger.info(f"Computed {n_graphs} graphs in {elapsed:.1f}s")

    # Compute features
    if not args.skip_features:
        logger.info("Pre-computing hand-crafted features...")
        start = time.time()
        features, feature_names = precompute_features(df)
        elapsed = time.time() - start
        logger.info(f"Computed {features.shape[1]} features in {elapsed:.1f}s")

        # Save features
        np.save(output_dir / 'features.npy', features)
        with open(output_dir / 'feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_names))
        logger.info(f"Saved features to {output_dir / 'features.npy'}")

    # Compute RNA-FM embeddings
    if not args.skip_rnafm:
        logger.info("Pre-computing RNA-FM embeddings...")
        start = time.time()
        embeddings = precompute_rnafm_embeddings(df['sequence'].tolist())

        if embeddings is not None:
            elapsed = time.time() - start
            logger.info(f"Computed embeddings shape {embeddings.shape} in {elapsed:.1f}s")

            # Save embeddings
            np.save(output_dir / 'rnafm_embeddings.npy', embeddings)
            logger.info(f"Saved embeddings to {output_dir / 'rnafm_embeddings.npy'}")

    # Save labels and metadata
    labels = df['is_edited'].values
    np.save(output_dir / 'labels.npy', labels)

    metadata = df[['chrom', 'position', 'strand', 'coverage', 'editing_rate', 'mfe']].copy()
    metadata.to_csv(output_dir / 'metadata.csv', index=False)

    logger.info(f"\nPre-computation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Files created:")
    for f in output_dir.iterdir():
        size_mb = f.stat().st_size / 1024 / 1024
        logger.info(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()

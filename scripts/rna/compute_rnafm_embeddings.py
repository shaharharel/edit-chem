#!/usr/bin/env python3
"""
Compute RNA-FM embeddings for ADAR dataset.
Standalone script that only requires RNA-FM, not torch_geometric.
"""

import sys
from pathlib import Path
import argparse
import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def compute_rnafm_embeddings(
    sequences: list,
    batch_size: int = 32,
    pooling: str = "mean"
) -> np.ndarray:
    """
    Compute RNA-FM embeddings for sequences.
    """
    from src.embedding.rna import RNAFMEmbedder

    logger.info("Loading RNA-FM model...")
    embedder = RNAFMEmbedder(pooling=pooling)

    logger.info(f"Encoding {len(sequences):,} sequences with batch_size={batch_size}...")

    # Process in batches to show progress
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Batches"):
        batch = sequences[i:i+batch_size]
        batch_emb = embedder.encode(batch)
        all_embeddings.append(batch_emb)

    embeddings = np.vstack(all_embeddings)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Compute RNA-FM embeddings')
    parser.add_argument('--input', type=str,
                        default='data/rna/adar/adar_editing_dataset_with_structure.csv',
                        help='Input CSV file')
    parser.add_argument('--output-dir', type=str,
                        default='data/rna/adar/precomputed_sampled',
                        help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for encoding')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'max', 'cls'],
                        help='Pooling strategy')
    parser.add_argument('--n-negatives', type=int, default=1000000,
                        help='Number of negative samples')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if embeddings already exist
    emb_file = output_dir / 'rnafm_embeddings.npy'
    if emb_file.exists():
        logger.info(f"Embeddings already exist at {emb_file}")
        existing = np.load(emb_file)
        logger.info(f"Existing shape: {existing.shape}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting.")
            return

    # Load data - use the same sampling as precomputed data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Full dataset: {len(df):,} samples")

    # Sample same as before
    positives = df[df['is_edited'] == 1]
    negatives = df[df['is_edited'] == 0]

    if args.n_negatives < len(negatives):
        negatives = negatives.sample(n=args.n_negatives, random_state=42)

    df = pd.concat([positives, negatives]).sample(frac=1, random_state=42)
    df = df.reset_index(drop=True)
    logger.info(f"Sampled to {len(df):,} samples ({len(positives):,} pos, {len(negatives):,} neg)")

    # Compute embeddings
    start = time.time()
    embeddings = compute_rnafm_embeddings(
        df['sequence'].tolist(),
        batch_size=args.batch_size,
        pooling=args.pooling
    )
    elapsed = time.time() - start

    logger.info(f"Computed embeddings shape {embeddings.shape} in {elapsed:.1f}s")

    # Save
    np.save(emb_file, embeddings)
    logger.info(f"Saved embeddings to {emb_file}")

    # Print file size
    size_mb = emb_file.stat().st_size / 1024 / 1024
    logger.info(f"File size: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()

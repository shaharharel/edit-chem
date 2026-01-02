"""
Hard Negative Sampling for ADAR Editing Prediction.

This module addresses critical biases discovered in the initial experiments:

1. Coverage Bias: Positives have 2x higher coverage (mean=352 vs 185)
   - Model can achieve ~0.95 AUROC just from coverage alone

2. Sequence Context Bias: UAG motif is 5x enriched in positives
   - U at position -1: 47% vs 20% (2.4x)
   - G at position +1: 44% vs 18% (2.4x)
   - UAG context: 17.5% vs 3.4% (5x)

This module provides:
- Coverage-matched negative sampling
- Context-matched negative sampling
- Combined coverage+context matching
- "Hard negatives": UAG context + high coverage but NOT edited
- Gene-based splitting for cross-gene generalization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class HardNegativeSamplingConfig:
    """Configuration for hard negative sampling."""
    precomputed_dir: str = "data/rna/adar/precomputed_sampled"
    n_coverage_bins: int = 20
    n_context_bins: int = 16  # 4x4 XAY contexts
    random_state: int = 42

    # Sampling targets
    target_negatives: int = 40000  # Match number of positives
    hard_negative_fraction: float = 0.3  # Fraction of negatives that are "hard"

    # Coverage matching
    coverage_match_tolerance: float = 0.1  # Allow 10% deviation

    # Hard negative thresholds
    hard_neg_min_coverage_percentile: float = 50  # Coverage >= median positive
    hard_neg_contexts: List[str] = None  # UAG, AAG by default


def load_precomputed_data(precomputed_dir: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Load precomputed features, embeddings, and metadata.

    Returns:
        features: [N, D] hand-crafted features
        labels: [N] binary labels
        metadata: DataFrame with chrom, position, coverage, etc.
        feature_names: list of feature names
    """
    precomputed_dir = Path(precomputed_dir)

    features = np.load(precomputed_dir / 'features.npy')
    labels = np.load(precomputed_dir / 'labels.npy')
    metadata = pd.read_csv(precomputed_dir / 'metadata.csv')

    with open(precomputed_dir / 'feature_names.txt') as f:
        feature_names = f.read().strip().split('\n')

    logger.info(f"Loaded {len(labels):,} samples ({(labels==1).sum():,} pos, {(labels==0).sum():,} neg)")

    return features, labels, metadata, feature_names


def extract_context_from_features(features: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """
    Extract trinucleotide context (XAY) from one-hot encoded features.

    Returns:
        contexts: [N] array of context strings like 'UAG', 'AAG', etc.
    """
    # Find left_1_X and right_1_X feature indices
    left_indices = {}
    right_indices = {}
    for i, name in enumerate(feature_names):
        if name.startswith('left_1_'):
            nuc = name.split('_')[-1]
            left_indices[nuc] = i
        elif name.startswith('right_1_'):
            nuc = name.split('_')[-1]
            right_indices[nuc] = i

    contexts = []
    for row in features:
        left_nuc = 'N'
        right_nuc = 'N'
        for nuc, idx in left_indices.items():
            if row[idx] == 1:
                left_nuc = nuc
                break
        for nuc, idx in right_indices.items():
            if row[idx] == 1:
                right_nuc = nuc
                break
        contexts.append(f"{left_nuc}A{right_nuc}")

    return np.array(contexts)


def get_coverage_matched_indices(
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
    coverage: np.ndarray,
    n_samples: int,
    n_bins: int = 20,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample negatives to match positive coverage distribution.

    Args:
        pos_indices: Indices of positive samples
        neg_indices: Indices of negative samples
        coverage: Coverage values for all samples
        n_samples: Number of negatives to sample
        n_bins: Number of coverage bins
        random_state: Random seed

    Returns:
        Selected negative indices
    """
    np.random.seed(random_state)

    pos_cov = coverage[pos_indices]
    neg_cov = coverage[neg_indices]

    # Define bins based on positive coverage quantiles
    quantiles = np.percentile(pos_cov, np.linspace(0, 100, n_bins + 1))

    # Get positive distribution across bins
    pos_bin_assignments = np.digitize(pos_cov, quantiles[1:-1])
    pos_bin_counts = np.bincount(pos_bin_assignments, minlength=n_bins)
    pos_bin_fracs = pos_bin_counts / pos_bin_counts.sum()

    # Assign negatives to bins
    neg_bin_assignments = np.digitize(neg_cov, quantiles[1:-1])

    # Sample from each bin proportionally
    sampled = []
    for bin_id in range(n_bins):
        target_from_bin = int(n_samples * pos_bin_fracs[bin_id])
        if target_from_bin == 0:
            continue

        bin_neg_mask = neg_bin_assignments == bin_id
        bin_neg_indices = neg_indices[bin_neg_mask]

        if len(bin_neg_indices) == 0:
            continue
        elif len(bin_neg_indices) <= target_from_bin:
            sampled.extend(bin_neg_indices)
        else:
            selected = np.random.choice(bin_neg_indices, size=target_from_bin, replace=False)
            sampled.extend(selected)

    return np.array(sampled)


def get_context_matched_indices(
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
    contexts: np.ndarray,
    n_samples: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample negatives to match positive context (XAY) distribution.

    Args:
        pos_indices: Indices of positive samples
        neg_indices: Indices of negative samples
        contexts: Context strings for all samples
        n_samples: Number of negatives to sample
        random_state: Random seed

    Returns:
        Selected negative indices
    """
    np.random.seed(random_state)

    pos_contexts = contexts[pos_indices]
    neg_contexts = contexts[neg_indices]

    # Get positive context distribution
    unique_contexts, pos_counts = np.unique(pos_contexts, return_counts=True)
    context_fracs = {ctx: count / len(pos_contexts) for ctx, count in zip(unique_contexts, pos_counts)}

    # Group negatives by context
    neg_by_context = defaultdict(list)
    for i, idx in enumerate(neg_indices):
        neg_by_context[neg_contexts[i]].append(idx)

    # Sample from each context proportionally
    sampled = []
    for ctx, frac in context_fracs.items():
        target_from_ctx = int(n_samples * frac)
        if target_from_ctx == 0:
            continue

        available = neg_by_context.get(ctx, [])
        if len(available) == 0:
            continue
        elif len(available) <= target_from_ctx:
            sampled.extend(available)
        else:
            selected = np.random.choice(available, size=target_from_ctx, replace=False)
            sampled.extend(selected)

    return np.array(sampled)


def get_combined_matched_indices(
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
    coverage: np.ndarray,
    contexts: np.ndarray,
    n_samples: int,
    n_coverage_bins: int = 10,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample negatives to match both coverage AND context distribution jointly.

    Args:
        pos_indices: Indices of positive samples
        neg_indices: Indices of negative samples
        coverage: Coverage values for all samples
        contexts: Context strings for all samples
        n_samples: Number of negatives to sample
        n_coverage_bins: Number of coverage bins
        random_state: Random seed

    Returns:
        Selected negative indices
    """
    np.random.seed(random_state)

    pos_cov = coverage[pos_indices]
    pos_ctx = contexts[pos_indices]

    # Define coverage bins
    cov_quantiles = np.percentile(pos_cov, np.linspace(0, 100, n_coverage_bins + 1))

    # Get positive joint distribution
    pos_cov_bins = np.digitize(pos_cov, cov_quantiles[1:-1])
    pos_joint = defaultdict(int)
    for cov_bin, ctx in zip(pos_cov_bins, pos_ctx):
        pos_joint[(ctx, cov_bin)] += 1

    total_pos = sum(pos_joint.values())
    joint_fracs = {key: count / total_pos for key, count in pos_joint.items()}

    # Filter negatives to positive coverage range
    neg_cov = coverage[neg_indices]
    in_range_mask = (neg_cov >= cov_quantiles[0]) & (neg_cov <= cov_quantiles[-1])
    neg_indices_filtered = neg_indices[in_range_mask]
    neg_cov_filtered = neg_cov[in_range_mask]
    neg_ctx_filtered = contexts[neg_indices_filtered]
    neg_cov_bins_filtered = np.digitize(neg_cov_filtered, cov_quantiles[1:-1])

    # Group negatives by (context, coverage_bin)
    neg_by_joint = defaultdict(list)
    for i, idx in enumerate(neg_indices_filtered):
        key = (neg_ctx_filtered[i], neg_cov_bins_filtered[i])
        neg_by_joint[key].append(idx)

    # Sample from each cell proportionally
    sampled = []
    for key, frac in joint_fracs.items():
        target_from_cell = int(n_samples * frac)
        if target_from_cell == 0:
            continue

        available = neg_by_joint.get(key, [])
        if len(available) == 0:
            continue
        elif len(available) <= target_from_cell:
            sampled.extend(available)
        else:
            selected = np.random.choice(available, size=target_from_cell, replace=False)
            sampled.extend(selected)

    return np.array(sampled)


def get_hard_negative_indices(
    neg_indices: np.ndarray,
    coverage: np.ndarray,
    contexts: np.ndarray,
    pos_coverage: np.ndarray,
    n_samples: int,
    min_coverage_percentile: float = 50,
    hard_contexts: List[str] = None,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample "hard negatives" - high coverage + ADAR-preferred context but NOT edited.

    These are adenosines that SHOULD be edited based on known ADAR preferences
    (UAG/AAG context, dsRNA structure, high coverage) but are NOT edited.

    Args:
        neg_indices: Indices of negative samples
        coverage: Coverage values for all samples
        contexts: Context strings for all samples
        pos_coverage: Coverage values of positive samples (for threshold)
        n_samples: Number of hard negatives to sample
        min_coverage_percentile: Minimum coverage percentile (of positives)
        hard_contexts: List of "ADAR-preferred" contexts (default: UAG, AAG)
        random_state: Random seed

    Returns:
        Selected hard negative indices
    """
    np.random.seed(random_state)

    if hard_contexts is None:
        hard_contexts = ['UAG', 'AAG']  # Most enriched in positives

    # Get coverage threshold
    cov_threshold = np.percentile(pos_coverage, min_coverage_percentile)

    # Filter negatives
    neg_cov = coverage[neg_indices]
    neg_ctx = contexts[neg_indices]

    hard_mask = (neg_cov >= cov_threshold) & np.isin(neg_ctx, hard_contexts)
    hard_neg_indices = neg_indices[hard_mask]

    logger.info(f"Found {len(hard_neg_indices):,} hard negatives "
                f"(coverage >= {cov_threshold:.0f}, context in {hard_contexts})")

    if len(hard_neg_indices) == 0:
        return np.array([])
    elif len(hard_neg_indices) <= n_samples:
        return hard_neg_indices
    else:
        return np.random.choice(hard_neg_indices, size=n_samples, replace=False)


def create_balanced_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: List[str],
    config: HardNegativeSamplingConfig,
    sampling_strategy: str = 'coverage_matched'
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Create a balanced dataset with various negative sampling strategies.

    Args:
        features: [N, D] features
        labels: [N] labels
        metadata: Metadata DataFrame
        feature_names: Feature names
        config: Sampling configuration
        sampling_strategy: One of:
            - 'random': Random sampling (baseline, biased)
            - 'coverage_matched': Match positive coverage distribution
            - 'context_matched': Match positive XAY context distribution
            - 'combined_matched': Match both coverage AND context jointly
            - 'hard_negatives': Include hard negatives (ADAR-preferred but not edited)
            - 'mixed': Combination of matched + hard negatives

    Returns:
        features_balanced: [M, D] balanced features
        labels_balanced: [M] balanced labels
        metadata_balanced: Balanced metadata
        indices: Original indices of selected samples
    """
    np.random.seed(config.random_state)

    # Get positive and negative indices
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    # Extract coverage and context
    coverage = metadata['coverage'].values
    contexts = extract_context_from_features(features, feature_names)

    n_pos = len(pos_indices)
    n_neg_target = config.target_negatives

    logger.info(f"Creating balanced dataset: {n_pos:,} positives, targeting {n_neg_target:,} negatives")
    logger.info(f"Sampling strategy: {sampling_strategy}")

    if sampling_strategy == 'random':
        # Baseline: random sampling (biased!)
        if len(neg_indices) > n_neg_target:
            selected_neg = np.random.choice(neg_indices, size=n_neg_target, replace=False)
        else:
            selected_neg = neg_indices

    elif sampling_strategy == 'coverage_matched':
        selected_neg = get_coverage_matched_indices(
            pos_indices, neg_indices, coverage, n_neg_target,
            n_bins=config.n_coverage_bins, random_state=config.random_state
        )

    elif sampling_strategy == 'context_matched':
        selected_neg = get_context_matched_indices(
            pos_indices, neg_indices, contexts, n_neg_target,
            random_state=config.random_state
        )

    elif sampling_strategy == 'combined_matched':
        selected_neg = get_combined_matched_indices(
            pos_indices, neg_indices, coverage, contexts, n_neg_target,
            n_coverage_bins=10, random_state=config.random_state
        )

    elif sampling_strategy == 'hard_negatives':
        # Only hard negatives
        selected_neg = get_hard_negative_indices(
            neg_indices, coverage, contexts,
            pos_coverage=coverage[pos_indices],
            n_samples=n_neg_target,
            min_coverage_percentile=config.hard_neg_min_coverage_percentile,
            hard_contexts=config.hard_neg_contexts,
            random_state=config.random_state
        )

    elif sampling_strategy == 'mixed':
        # Combination: some matched, some hard
        n_hard = int(n_neg_target * config.hard_negative_fraction)
        n_matched = n_neg_target - n_hard

        # Get hard negatives
        hard_neg = get_hard_negative_indices(
            neg_indices, coverage, contexts,
            pos_coverage=coverage[pos_indices],
            n_samples=n_hard,
            min_coverage_percentile=config.hard_neg_min_coverage_percentile,
            hard_contexts=config.hard_neg_contexts,
            random_state=config.random_state
        )

        # Get matched negatives (excluding hard ones)
        remaining_neg = np.setdiff1d(neg_indices, hard_neg)
        matched_neg = get_combined_matched_indices(
            pos_indices, remaining_neg, coverage, contexts, n_matched,
            n_coverage_bins=10, random_state=config.random_state + 1
        )

        selected_neg = np.concatenate([hard_neg, matched_neg])

    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    # Combine positives and selected negatives
    all_indices = np.concatenate([pos_indices, selected_neg])
    np.random.shuffle(all_indices)

    features_balanced = features[all_indices]
    labels_balanced = labels[all_indices]
    metadata_balanced = metadata.iloc[all_indices].reset_index(drop=True)

    logger.info(f"Balanced dataset: {len(all_indices):,} samples "
                f"({(labels_balanced==1).sum():,} pos, {(labels_balanced==0).sum():,} neg)")

    return features_balanced, labels_balanced, metadata_balanced, all_indices


def analyze_dataset_biases(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: List[str],
    dataset_name: str = "Dataset"
) -> Dict:
    """
    Analyze biases in a dataset for coverage and context.

    Returns:
        Dictionary with bias statistics
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    coverage = metadata['coverage'].values
    contexts = extract_context_from_features(features, feature_names)

    stats = {
        'name': dataset_name,
        'n_total': len(labels),
        'n_pos': pos_mask.sum(),
        'n_neg': neg_mask.sum(),
    }

    # Coverage statistics
    stats['pos_cov_mean'] = coverage[pos_mask].mean()
    stats['pos_cov_median'] = np.median(coverage[pos_mask])
    stats['neg_cov_mean'] = coverage[neg_mask].mean()
    stats['neg_cov_median'] = np.median(coverage[neg_mask])
    stats['cov_ratio'] = stats['pos_cov_mean'] / stats['neg_cov_mean']

    # Log coverage
    log_cov = np.log1p(coverage)
    stats['pos_log_cov_mean'] = log_cov[pos_mask].mean()
    stats['neg_log_cov_mean'] = log_cov[neg_mask].mean()

    # Context statistics
    pos_contexts = contexts[pos_mask]
    neg_contexts = contexts[neg_mask]

    for ctx in ['UAG', 'AAG', 'UAA', 'GAA']:
        pos_frac = (pos_contexts == ctx).mean()
        neg_frac = (neg_contexts == ctx).mean()
        ratio = pos_frac / neg_frac if neg_frac > 0 else float('inf')
        stats[f'{ctx}_pos_frac'] = pos_frac
        stats[f'{ctx}_neg_frac'] = neg_frac
        stats[f'{ctx}_ratio'] = ratio

    return stats


def print_bias_comparison(stats_list: List[Dict]):
    """Print comparison of biases across datasets."""
    print("\n" + "=" * 90)
    print("BIAS COMPARISON ACROSS SAMPLING STRATEGIES")
    print("=" * 90)

    # Coverage comparison
    print("\n📊 COVERAGE BIAS:")
    print(f"{'Dataset':<25} {'Pos Mean':>10} {'Neg Mean':>10} {'Ratio':>8}")
    print("-" * 55)
    for stats in stats_list:
        print(f"{stats['name']:<25} {stats['pos_cov_mean']:>10.1f} {stats['neg_cov_mean']:>10.1f} {stats['cov_ratio']:>7.2f}x")

    # Context comparison
    print("\n📊 CONTEXT BIAS (UAG - most ADAR-preferred):")
    print(f"{'Dataset':<25} {'Pos %':>10} {'Neg %':>10} {'Ratio':>8}")
    print("-" * 55)
    for stats in stats_list:
        print(f"{stats['name']:<25} {stats['UAG_pos_frac']*100:>9.1f}% {stats['UAG_neg_frac']*100:>9.1f}% {stats['UAG_ratio']:>7.2f}x")

    print("\n📊 CONTEXT BIAS (AAG - second most ADAR-preferred):")
    print(f"{'Dataset':<25} {'Pos %':>10} {'Neg %':>10} {'Ratio':>8}")
    print("-" * 55)
    for stats in stats_list:
        print(f"{stats['name']:<25} {stats['AAG_pos_frac']*100:>9.1f}% {stats['AAG_neg_frac']*100:>9.1f}% {stats['AAG_ratio']:>7.2f}x")

    print("=" * 90)


# ============================================================================
# ADAR-Specific Generalization Splits
# ============================================================================

def create_adar_gene_split(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    gtf_file: str = "data/rna/adar/sacCer3.ncbiRefSeq.gtf",
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/val/test split by holding out entire genes.

    Tests whether the model generalizes to unseen genes.

    Args:
        metadata: DataFrame with chrom, position, strand
        labels: Sample labels
        gtf_file: Path to GTF file with gene annotations
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation set
        random_state: Random seed

    Returns:
        train_indices, val_indices, test_indices
    """
    np.random.seed(random_state)

    # Parse GTF to get gene boundaries
    genes = []
    with open(gtf_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 9 and parts[2] == 'transcript':
                chrom = parts[0].replace('chr', '')
                start = int(parts[3])
                end = int(parts[4])
                strand = parts[6]
                attrs = parts[8]
                gene_id = None
                for attr in attrs.split(';'):
                    attr = attr.strip()
                    if attr.startswith('gene_id'):
                        gene_id = attr.split('"')[1]
                        break
                if gene_id:
                    genes.append({
                        'chrom': chrom,
                        'start': start,
                        'end': end,
                        'strand': strand,
                        'gene_id': gene_id
                    })

    genes_df = pd.DataFrame(genes)

    # Aggregate by gene
    gene_bounds = genes_df.groupby(['chrom', 'gene_id', 'strand']).agg({
        'start': 'min',
        'end': 'max'
    }).reset_index()

    logger.info(f"Loaded {len(gene_bounds)} genes from GTF")

    # Map each sample to a gene
    def map_to_gene(row):
        chrom = str(row['chrom'])
        pos = row['position']
        strand = row['strand']

        mask = (gene_bounds['chrom'] == chrom) & (gene_bounds['strand'] == strand)
        cand = gene_bounds[mask]
        hits = cand[(cand['start'] <= pos) & (cand['end'] >= pos)]

        if len(hits) > 0:
            return hits.iloc[0]['gene_id']
        return f"intergenic_{chrom}_{pos // 10000}"  # Group intergenic by 10kb region

    metadata = metadata.copy()
    metadata['gene_id'] = metadata.apply(map_to_gene, axis=1)

    # Get unique genes
    unique_genes = metadata['gene_id'].unique()
    np.random.shuffle(unique_genes)

    n_test = max(1, int(len(unique_genes) * test_ratio))
    n_val = max(1, int(len(unique_genes) * val_ratio))

    test_genes = set(unique_genes[:n_test])
    val_genes = set(unique_genes[n_test:n_test + n_val])
    train_genes = set(unique_genes[n_test + n_val:])

    # Assign samples to splits
    gene_ids = metadata['gene_id'].values

    train_mask = np.array([g in train_genes for g in gene_ids])
    val_mask = np.array([g in val_genes for g in gene_ids])
    test_mask = np.array([g in test_genes for g in gene_ids])

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]

    logger.info(f"Gene split - Train: {len(train_indices):,} ({len(train_genes)} genes), "
                f"Val: {len(val_indices):,} ({len(val_genes)} genes), "
                f"Test: {len(test_indices):,} ({len(test_genes)} genes)")

    return train_indices, val_indices, test_indices


def create_coverage_holdout_split(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    test_coverage_range: Tuple[float, float] = (0.75, 1.0),
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create split that holds out a coverage range for testing.

    Tests whether model generalizes to different coverage regimes.

    Args:
        metadata: DataFrame with coverage column
        labels: Sample labels
        test_coverage_range: Tuple of (min_percentile, max_percentile) for test set
        val_ratio: Fraction for validation (from train set)
        random_state: Random seed

    Returns:
        train_indices, val_indices, test_indices
    """
    np.random.seed(random_state)

    coverage = metadata['coverage'].values

    # Get coverage thresholds
    min_pct, max_pct = test_coverage_range
    min_cov = np.percentile(coverage, min_pct * 100)
    max_cov = np.percentile(coverage, max_pct * 100)

    # Split by coverage
    test_mask = (coverage >= min_cov) & (coverage <= max_cov)
    train_val_mask = ~test_mask

    test_indices = np.where(test_mask)[0]
    train_val_indices = np.where(train_val_mask)[0]

    # Split train/val
    np.random.shuffle(train_val_indices)
    n_val = int(len(train_val_indices) * val_ratio)

    val_indices = train_val_indices[:n_val]
    train_indices = train_val_indices[n_val:]

    logger.info(f"Coverage holdout split - Test coverage: {min_cov:.0f}-{max_cov:.0f}")
    logger.info(f"Train: {len(train_indices):,}, Val: {len(val_indices):,}, Test: {len(test_indices):,}")

    return train_indices, val_indices, test_indices


def create_context_holdout_split(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    test_contexts: List[str] = None,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create split that holds out specific sequence contexts for testing.

    Tests whether model generalizes across sequence contexts.

    Args:
        features: Feature array
        labels: Sample labels
        feature_names: Feature names
        test_contexts: Contexts to hold out (default: UAG - most ADAR-preferred)
        val_ratio: Fraction for validation
        random_state: Random seed

    Returns:
        train_indices, val_indices, test_indices
    """
    np.random.seed(random_state)

    if test_contexts is None:
        test_contexts = ['UAG']  # Most ADAR-preferred

    contexts = extract_context_from_features(features, feature_names)

    # Split by context
    test_mask = np.isin(contexts, test_contexts)
    train_val_mask = ~test_mask

    test_indices = np.where(test_mask)[0]
    train_val_indices = np.where(train_val_mask)[0]

    # Split train/val
    np.random.shuffle(train_val_indices)
    n_val = int(len(train_val_indices) * val_ratio)

    val_indices = train_val_indices[:n_val]
    train_indices = train_val_indices[n_val:]

    logger.info(f"Context holdout split - Test contexts: {test_contexts}")
    logger.info(f"Train: {len(train_indices):,}, Val: {len(val_indices):,}, Test: {len(test_indices):,}")

    return train_indices, val_indices, test_indices


def create_editing_rate_extrapolation_split(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    train_percentile: float = 75,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create split to test extrapolation to high editing rates.

    Train on low-medium editing rates, test on high editing rates.

    Args:
        metadata: DataFrame with editing_rate column
        labels: Sample labels
        train_percentile: Train on positives below this percentile
        val_ratio: Fraction for validation
        random_state: Random seed

    Returns:
        train_indices, val_indices, test_indices
    """
    np.random.seed(random_state)

    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    # Split positives by editing rate
    editing_rates = metadata['editing_rate'].values
    pos_rates = editing_rates[pos_indices]
    rate_threshold = np.percentile(pos_rates, train_percentile)

    pos_train_mask = pos_rates <= rate_threshold
    pos_test_mask = pos_rates > rate_threshold

    pos_train_indices = pos_indices[pos_train_mask]
    pos_test_indices = pos_indices[pos_test_mask]

    # Split positives train into train/val
    np.random.shuffle(pos_train_indices)
    n_pos_val = int(len(pos_train_indices) * val_ratio)
    pos_val_indices = pos_train_indices[:n_pos_val]
    pos_train_indices = pos_train_indices[n_pos_val:]

    # Split negatives randomly
    np.random.shuffle(neg_indices)
    n_neg_test = int(len(neg_indices) * (len(pos_test_indices) / len(pos_indices)))
    n_neg_val = int(len(neg_indices) * val_ratio)

    neg_test_indices = neg_indices[:n_neg_test]
    neg_val_indices = neg_indices[n_neg_test:n_neg_test + n_neg_val]
    neg_train_indices = neg_indices[n_neg_test + n_neg_val:]

    # Combine
    train_indices = np.concatenate([pos_train_indices, neg_train_indices])
    val_indices = np.concatenate([pos_val_indices, neg_val_indices])
    test_indices = np.concatenate([pos_test_indices, neg_test_indices])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    logger.info(f"Editing rate extrapolation split - Train on rate <= {rate_threshold:.3f}")
    logger.info(f"Train: {len(train_indices):,}, Val: {len(val_indices):,}, Test: {len(test_indices):,}")
    logger.info(f"Test positives have editing rate > {rate_threshold:.3f}")

    return train_indices, val_indices, test_indices


# ============================================================================
# Main function for generating all datasets
# ============================================================================

def generate_all_challenging_datasets(
    precomputed_dir: str = "data/rna/adar/precomputed_sampled",
    output_dir: str = "data/rna/adar/challenging_datasets",
    random_state: int = 42
):
    """
    Generate all challenging datasets for stress-testing ADAR models.

    Creates:
    1. Coverage-matched dataset
    2. Context-matched dataset
    3. Combined-matched dataset
    4. Hard negatives dataset
    5. Mixed dataset (matched + hard)

    Also analyzes and compares biases across all datasets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features, labels, metadata, feature_names = load_precomputed_data(precomputed_dir)

    # Configuration
    config = HardNegativeSamplingConfig(
        precomputed_dir=precomputed_dir,
        random_state=random_state,
        target_negatives=40000,
        hard_negative_fraction=0.3,
        hard_neg_contexts=['UAG', 'AAG']
    )

    # Generate datasets with different strategies
    strategies = [
        'random',
        'coverage_matched',
        'context_matched',
        'combined_matched',
        'hard_negatives',
        'mixed'
    ]

    all_stats = []

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Generating: {strategy}")
        print(f"{'='*60}")

        feat_bal, labels_bal, meta_bal, indices = create_balanced_dataset(
            features, labels, metadata, feature_names, config, sampling_strategy=strategy
        )

        # Analyze biases
        stats = analyze_dataset_biases(feat_bal, labels_bal, meta_bal, feature_names, strategy)
        all_stats.append(stats)

        # Save dataset
        strategy_dir = output_dir / strategy
        strategy_dir.mkdir(exist_ok=True)

        np.save(strategy_dir / 'features.npy', feat_bal)
        np.save(strategy_dir / 'labels.npy', labels_bal)
        np.save(strategy_dir / 'indices.npy', indices)
        meta_bal.to_csv(strategy_dir / 'metadata.csv', index=False)

        with open(strategy_dir / 'feature_names.txt', 'w') as f:
            f.write('\n'.join(feature_names))

        print(f"Saved to {strategy_dir}")

    # Print bias comparison
    print_bias_comparison(all_stats)

    # Save comparison
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(output_dir / 'bias_comparison.csv', index=False)

    return all_stats


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    stats = generate_all_challenging_datasets()

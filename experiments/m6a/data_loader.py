"""
Data loading for m6A YTH binding experiments.

Handles the specific structure of paired A/M binding data with:
- Multiple technical replicates
- Variance weighting
- Various evaluation splits (context-held-out, motif-held-out, etc.)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import ast


def load_m6a_data(
    data_file: str,
    target_column: str = 'log2_fold_change_median',
    min_replicates: int = 5,
    use_variance_weighting: bool = True
) -> pd.DataFrame:
    """
    Load m6A paired binding data.

    Args:
        data_file: Path to processed paired CSV file
        target_column: Column to use as prediction target
        min_replicates: Minimum replicates required per sequence
        use_variance_weighting: Calculate sample weights from variance

    Returns:
        DataFrame with required columns for modeling
    """
    print(f"Loading m6A data from: {data_file}")

    df = pd.read_csv(data_file)
    print(f"  Loaded {len(df):,} paired sequences")

    # Filter by minimum replicates
    if 'n_replicates_A' in df.columns and 'n_replicates_M' in df.columns:
        mask = (df['n_replicates_A'] >= min_replicates) & (df['n_replicates_M'] >= min_replicates)
        df = df[mask].copy()
        print(f"  After replicate filter (min={min_replicates}): {len(df):,} pairs")

    # Validate target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {df.columns.tolist()}")

    # Remove rows with missing targets
    df = df.dropna(subset=[target_column])
    print(f"  After removing NaN targets: {len(df):,} pairs")

    # Calculate sample weights from variance if requested
    if use_variance_weighting:
        df = _calculate_sample_weights(df)

    # Standardize column names for modeling
    df = _standardize_columns(df, target_column)

    # Print summary statistics
    _print_data_summary(df)

    return df


def _calculate_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sample weights based on measurement variance."""
    # Combine variance from both A and M measurements
    if 'intensity_std_A' in df.columns and 'intensity_std_M' in df.columns:
        # Propagate variance for log2 fold change
        # Var(log2(M/A)) ≈ (var_M/μ_M² + var_A/μ_A²) / ln(2)²
        var_A = df['intensity_std_A'] ** 2
        var_M = df['intensity_std_M'] ** 2
        mean_A = df['intensity_mean_A'].clip(lower=1e-6)
        mean_M = df['intensity_mean_M'].clip(lower=1e-6)

        # Coefficient of variation squared
        cv2_A = var_A / (mean_A ** 2)
        cv2_M = var_M / (mean_M ** 2)

        # Propagated variance for log2 ratio
        propagated_var = (cv2_A + cv2_M) / (np.log(2) ** 2)
        propagated_var = propagated_var.clip(lower=1e-6)

        # Weight is inverse variance (normalized)
        df['sample_weight'] = 1.0 / propagated_var
        df['sample_weight'] = df['sample_weight'] / df['sample_weight'].mean()

        print(f"  Calculated sample weights: mean={df['sample_weight'].mean():.2f}, "
              f"std={df['sample_weight'].std():.2f}")
    else:
        df['sample_weight'] = 1.0

    return df


def _standardize_columns(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Standardize column names for modeling compatibility."""
    # Create standard columns expected by models
    df['seq_a'] = df['loop_A']  # Use loop sequence (7-mer)
    df['seq_b'] = df['loop_M']  # Methylated version
    df['delta_value'] = df[target_column]  # Target
    df['property_name'] = 'm6a_binding'  # Property identifier

    # Extract edit information (A→M at center position)
    df['edit_position'] = 3  # Center of 7-mer (0-indexed)
    df['edit_from'] = 'A'
    df['edit_to'] = 'M'  # Represents m6A

    # Keep full sequence for context models
    if 'sequence_A' in df.columns:
        df['full_seq_a'] = df['sequence_A']
        df['full_seq_b'] = df['sequence_M']

    return df


def _print_data_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of the data."""
    print("\n  === Data Summary ===")
    print(f"  Total pairs: {len(df):,}")

    if 'is_drach' in df.columns:
        n_drach = df['is_drach'].sum()
        print(f"  DRACH motif: {n_drach:,} ({100*n_drach/len(df):.1f}%)")
        print(f"  Non-DRACH: {len(df)-n_drach:,} ({100*(len(df)-n_drach)/len(df):.1f}%)")

    if 'delta_value' in df.columns:
        print(f"  Target (Δ) range: [{df['delta_value'].min():.3f}, {df['delta_value'].max():.3f}]")
        print(f"  Target (Δ) mean: {df['delta_value'].mean():.3f} ± {df['delta_value'].std():.3f}")

    if 'context_3nt_upstream' in df.columns:
        print(f"  Unique upstream contexts: {df['context_3nt_upstream'].nunique()}")


def create_m6a_splits(
    df: pd.DataFrame,
    split_type: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test splits for m6A data.

    Args:
        df: DataFrame from load_m6a_data
        split_type: Type of split:
            - 'random': Standard random split
            - 'context_held_out': Hold out specific 3nt contexts for testing
            - 'motif_held_out': Hold out non-DRACH for testing (train on DRACH)
            - 'motif_held_out_reverse': Hold out DRACH for testing (train on non-DRACH)
            - 'rbp_motif_held_out': Hold out sequences with specific RBP motif
            - 'replicate_robustness': Split by replicates (requires raw data)
        train_ratio, val_ratio, test_ratio: Split ratios
        random_seed: Random seed
        **kwargs: Additional split-specific parameters

    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    print(f"\nCreating {split_type} split...")
    np.random.seed(random_seed)

    if split_type == 'random':
        return _random_split(df, train_ratio, val_ratio, random_seed)

    elif split_type == 'context_held_out':
        held_out_contexts = kwargs.get('held_out_contexts', ['GAC', 'AAC'])
        return _context_held_out_split(df, held_out_contexts, train_ratio, val_ratio, random_seed)

    elif split_type == 'motif_held_out':
        # Train on DRACH, test on non-DRACH
        return _motif_held_out_split(df, held_out_drach=False, train_ratio=train_ratio,
                                      val_ratio=val_ratio, random_seed=random_seed)

    elif split_type == 'motif_held_out_reverse':
        # Train on non-DRACH, test on DRACH
        return _motif_held_out_split(df, held_out_drach=True, train_ratio=train_ratio,
                                      val_ratio=val_ratio, random_seed=random_seed)

    elif split_type == 'rbp_motif_held_out':
        # Hold out sequences with a specific RBP motif
        rbp_motif = kwargs.get('rbp_motif', 'has_FTO_ALKBH5')
        return _rbp_motif_held_out_split(df, rbp_motif, train_ratio, val_ratio, random_seed)

    elif split_type == 'replicate_robustness':
        return _replicate_robustness_split(df, train_ratio, val_ratio, random_seed)

    else:
        print(f"  Warning: Unknown split type '{split_type}', using random split")
        return _random_split(df, train_ratio, val_ratio, random_seed)


def _random_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    random_seed: int
) -> Dict[str, pd.DataFrame]:
    """Standard random split."""
    n = len(df)
    indices = np.arange(n)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[indices[:train_end]].reset_index(drop=True)
    val_df = df.iloc[indices[train_end:val_end]].reset_index(drop=True)
    test_df = df.iloc[indices[val_end:]].reset_index(drop=True)

    _print_split_stats(train_df, val_df, test_df)

    return {'train': train_df, 'val': val_df, 'test': test_df}


def _context_held_out_split(
    df: pd.DataFrame,
    held_out_contexts: List[str],
    train_ratio: float,
    val_ratio: float,
    random_seed: int
) -> Dict[str, pd.DataFrame]:
    """
    Hold out specific 3nt contexts for testing.

    This evaluates generalization to unseen sequence contexts.
    """
    print(f"  Held out contexts: {held_out_contexts}")

    # Identify held out sequences (by upstream context)
    if 'context_3nt_upstream' in df.columns:
        context_col = 'context_3nt_upstream'
    else:
        # Extract from loop sequence
        df['context_3nt_upstream'] = df['seq_a'].str[:3]
        context_col = 'context_3nt_upstream'

    test_mask = df[context_col].isin(held_out_contexts)
    test_df = df[test_mask].reset_index(drop=True)
    train_val_df = df[~test_mask].copy()

    print(f"  Test (held out contexts): {len(test_df):,} pairs")

    # Split remaining into train/val
    n = len(train_val_df)
    indices = np.arange(n)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Adjust ratios since test is already separated
    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    val_end = int(n * adjusted_val_ratio)

    val_df = train_val_df.iloc[indices[:val_end]].reset_index(drop=True)
    train_df = train_val_df.iloc[indices[val_end:]].reset_index(drop=True)

    _print_split_stats(train_df, val_df, test_df)

    return {'train': train_df, 'val': val_df, 'test': test_df}


def _motif_held_out_split(
    df: pd.DataFrame,
    held_out_drach: bool,
    train_ratio: float,
    val_ratio: float,
    random_seed: int
) -> Dict[str, pd.DataFrame]:
    """
    Hold out DRACH or non-DRACH sequences for testing.

    This evaluates whether models learn DRACH-specific vs general rules.
    """
    motif_type = "DRACH" if held_out_drach else "non-DRACH"
    print(f"  Held out motif type: {motif_type}")

    if 'is_drach' not in df.columns:
        raise ValueError("DataFrame must have 'is_drach' column for motif_held_out split")

    if held_out_drach:
        test_mask = df['is_drach']
    else:
        test_mask = ~df['is_drach']

    test_df = df[test_mask].reset_index(drop=True)
    train_val_df = df[~test_mask].copy()

    print(f"  Test ({motif_type}): {len(test_df):,} pairs")

    # Split remaining into train/val
    n = len(train_val_df)
    indices = np.arange(n)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    val_end = int(n * adjusted_val_ratio)

    val_df = train_val_df.iloc[indices[:val_end]].reset_index(drop=True)
    train_df = train_val_df.iloc[indices[val_end:]].reset_index(drop=True)

    _print_split_stats(train_df, val_df, test_df)

    return {'train': train_df, 'val': val_df, 'test': test_df}


def _rbp_motif_held_out_split(
    df: pd.DataFrame,
    rbp_motif: str,
    train_ratio: float,
    val_ratio: float,
    random_seed: int
) -> Dict[str, pd.DataFrame]:
    """
    Hold out sequences with a specific RBP motif for testing.

    Args:
        rbp_motif: Column name like 'has_FTO_ALKBH5', 'has_SRSF_motif', etc.
    """
    print(f"  Held out RBP motif: {rbp_motif}")

    if rbp_motif not in df.columns:
        # Try to add the motif column from the enriched data
        print(f"  Warning: {rbp_motif} not in columns, checking for RBP motif patterns...")
        # Fall back to random split
        return _random_split(df, train_ratio, val_ratio, random_seed)

    test_mask = df[rbp_motif] == True
    test_df = df[test_mask].reset_index(drop=True)
    train_val_df = df[~test_mask].copy()

    print(f"  Test (with {rbp_motif}): {len(test_df):,} pairs")
    print(f"  Train/Val (without {rbp_motif}): {len(train_val_df):,} pairs")

    if len(test_df) < 10:
        print(f"  Warning: Only {len(test_df)} test samples, split may not be reliable")

    # Split remaining into train/val
    n = len(train_val_df)
    indices = np.arange(n)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    val_end = int(n * adjusted_val_ratio)

    val_df = train_val_df.iloc[indices[:val_end]].reset_index(drop=True)
    train_df = train_val_df.iloc[indices[val_end:]].reset_index(drop=True)

    _print_split_stats(train_df, val_df, test_df)

    return {'train': train_df, 'val': val_df, 'test': test_df}


def _replicate_robustness_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    random_seed: int
) -> Dict[str, pd.DataFrame]:
    """
    Split data by replicates for robustness testing.

    This uses subset of replicates for training and different subset for testing
    to evaluate model stability across technical variation.

    Note: This requires the intensity_all columns with list of all replicate values.
    """
    print("  Replicate robustness split: using different replicate subsets")

    # Check if we have replicate data
    if 'intensity_all_A' not in df.columns or 'intensity_all_M' not in df.columns:
        print("  Warning: No replicate data found, falling back to random split")
        return _random_split(df, train_ratio, val_ratio, random_seed)

    # Parse list strings if needed
    def parse_list(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x

    # Split replicates: use 8 for train/val, 3 for test
    np.random.seed(random_seed)

    # Create new aggregated values from replicate subsets
    train_val_rows = []
    test_rows = []

    for idx, row in df.iterrows():
        try:
            reps_A = parse_list(row['intensity_all_A'])
            reps_M = parse_list(row['intensity_all_M'])
        except:
            continue

        n_reps = min(len(reps_A), len(reps_M))
        if n_reps < 5:
            continue

        # Shuffle and split replicates
        indices = np.arange(n_reps)
        np.random.shuffle(indices)

        train_end = int(n_reps * 0.7)

        train_reps_A = [reps_A[i] for i in indices[:train_end]]
        train_reps_M = [reps_M[i] for i in indices[:train_end]]
        test_reps_A = [reps_A[i] for i in indices[train_end:]]
        test_reps_M = [reps_M[i] for i in indices[train_end:]]

        # Create train row with subset of replicates
        train_row = row.copy()
        train_row['intensity_median_A'] = np.median(train_reps_A)
        train_row['intensity_median_M'] = np.median(train_reps_M)
        train_row['log2_fold_change_median'] = np.log2(
            train_row['intensity_median_M'] / train_row['intensity_median_A']
        )
        train_row['delta_value'] = train_row['log2_fold_change_median']
        train_val_rows.append(train_row)

        # Create test row with remaining replicates
        test_row = row.copy()
        test_row['intensity_median_A'] = np.median(test_reps_A)
        test_row['intensity_median_M'] = np.median(test_reps_M)
        test_row['log2_fold_change_median'] = np.log2(
            test_row['intensity_median_M'] / test_row['intensity_median_A']
        )
        test_row['delta_value'] = test_row['log2_fold_change_median']
        test_rows.append(test_row)

    train_val_df = pd.DataFrame(train_val_rows).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).reset_index(drop=True)

    # Split train_val into train and val
    n = len(train_val_df)
    indices = np.arange(n)
    np.random.shuffle(indices)

    adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
    val_end = int(n * adjusted_val_ratio)

    val_df = train_val_df.iloc[indices[:val_end]].reset_index(drop=True)
    train_df = train_val_df.iloc[indices[val_end:]].reset_index(drop=True)

    _print_split_stats(train_df, val_df, test_df)

    return {'train': train_df, 'val': val_df, 'test': test_df}


def _print_split_stats(train_df, val_df, test_df) -> None:
    """Print statistics for each split."""
    print(f"  Train: {len(train_df):,} pairs")
    print(f"  Val: {len(val_df):,} pairs")
    print(f"  Test: {len(test_df):,} pairs")

    # Print DRACH distribution if available
    if 'is_drach' in train_df.columns:
        for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            n_drach = split_df['is_drach'].sum()
            print(f"    {name} DRACH: {n_drach} ({100*n_drach/len(split_df):.1f}%)")


def prepare_model_inputs(
    df: pd.DataFrame,
    use_full_sequence: bool = False
) -> Dict:
    """
    Prepare data in format expected by models.

    Args:
        df: DataFrame from load_m6a_data/create_m6a_splits
        use_full_sequence: Use full hairpin sequence instead of loop

    Returns:
        Dict with model inputs
    """
    seq_col = 'full_seq_a' if use_full_sequence and 'full_seq_a' in df.columns else 'seq_a'

    return {
        'sequences': df[seq_col].tolist(),
        'targets': df['delta_value'].values,
        'sample_weights': df.get('sample_weight', pd.Series([1.0] * len(df))).values,
        'edit_positions': df.get('edit_position', pd.Series([3] * len(df))).values,
        'is_drach': df.get('is_drach', pd.Series([False] * len(df))).values,
        'contexts_upstream': df.get('context_3nt_upstream', pd.Series(['XXX'] * len(df))).tolist(),
        'contexts_downstream': df.get('context_3nt_downstream', pd.Series(['XXX'] * len(df))).tolist(),
        'pair_ids': df.get('pair_id', df.index).tolist()
    }

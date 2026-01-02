"""
Preprocess m6A YTH protein binding data.

This script processes the raw m6A binding assay data:
1. Merges design file with measurement files
2. Pairs methylated (M) and unmethylated (A) sequences
3. Aggregates 11 technical replicates per sequence (median, mean, all values)
4. Outputs processed paired dataset with metadata

Dataset background:
- ~90,000 measurements (11 technical replicates x ~8,200 sequences)
- Hairpin RNA structures with 7-mer loops containing either A or m6A (M) at center
- YTH protein binding intensities measured at two sensitivity settings
- Goal: Understand sequence-context effects on m6A-mediated YTH protein binding
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_design_file(design_path: Path) -> pd.DataFrame:
    """Load and process the sequence design file."""
    logger.info(f"Loading design file: {design_path}")
    design = pd.read_csv(design_path)
    logger.info(f"Design file shape: {design.shape}")
    logger.info(f"Center distribution:\n{design['center'].value_counts()}")
    return design


def load_measurement_file(measurement_path: Path) -> pd.DataFrame:
    """Load and process measurement TSV file."""
    logger.info(f"Loading measurement file: {measurement_path}")

    # Skip the header comment line
    measurements = pd.read_csv(measurement_path, sep='\t', skiprows=1)

    # Clean column names
    measurements.columns = measurements.columns.str.strip()

    # Extract relevant columns
    # SEQ_ID is the sequence name, PM_635_26112025_nm is the intensity measurement
    relevant_cols = ['SEQ_ID', 'PM_635_26112025_nm', 'X', 'Y', 'MATCH_INDEX']
    measurements = measurements[relevant_cols].copy()
    measurements.rename(columns={'SEQ_ID': 'name'}, inplace=True)

    logger.info(f"Measurement file shape: {measurements.shape}")
    logger.info(f"Unique sequences: {measurements['name'].nunique()}")

    return measurements


def aggregate_replicates(measurements: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 11 technical replicates per sequence.

    Returns DataFrame with:
    - intensity_median: Median intensity across replicates
    - intensity_mean: Mean intensity across replicates
    - intensity_std: Standard deviation across replicates
    - intensity_all: List of all replicate values
    - n_replicates: Number of replicates
    """
    logger.info("Aggregating technical replicates...")

    agg_df = measurements.groupby('name').agg(
        intensity_median=('PM_635_26112025_nm', 'median'),
        intensity_mean=('PM_635_26112025_nm', 'mean'),
        intensity_std=('PM_635_26112025_nm', 'std'),
        intensity_all=('PM_635_26112025_nm', list),
        n_replicates=('PM_635_26112025_nm', 'count')
    ).reset_index()

    logger.info(f"Aggregated shape: {agg_df.shape}")
    logger.info(f"Replicate counts:\n{agg_df['n_replicates'].value_counts()}")

    return agg_df


def extract_pair_id(name: str) -> str:
    """Extract the pair ID from sequence name (e.g., HP15-7-15_A_000001 -> 000001)."""
    parts = name.split('_')
    return parts[-1]  # Last part is the numeric ID


def extract_center_type(name: str) -> str:
    """Extract center type (A, M, G, U) from sequence name."""
    parts = name.split('_')
    return parts[-2]  # Second to last is A/M/G/U


def create_paired_dataset(design: pd.DataFrame, measurements_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Create paired dataset matching A (unmethylated) and M (methylated) sequences.

    Returns DataFrame with columns for both A and M versions of each sequence pair.
    """
    logger.info("Creating paired A/M dataset...")

    # Merge design with aggregated measurements
    merged = design.merge(measurements_agg, on='name', how='inner')
    logger.info(f"Merged shape: {merged.shape}")

    # Extract pair ID and center type
    merged['pair_id'] = merged['name'].apply(extract_pair_id)
    merged['center_type'] = merged['name'].apply(extract_center_type)

    # Separate A and M sequences
    a_seqs = merged[merged['center'] == 'A'].copy()
    m_seqs = merged[merged['center'] == 'M'].copy()

    logger.info(f"A sequences: {len(a_seqs)}, M sequences: {len(m_seqs)}")

    # Rename columns for A sequences
    a_cols = {
        'name': 'name_A',
        'sequence_5to3': 'sequence_A',
        'loop': 'loop_A',
        'intensity_median': 'intensity_median_A',
        'intensity_mean': 'intensity_mean_A',
        'intensity_std': 'intensity_std_A',
        'intensity_all': 'intensity_all_A',
        'n_replicates': 'n_replicates_A'
    }
    a_seqs = a_seqs.rename(columns=a_cols)

    # Rename columns for M sequences
    m_cols = {
        'name': 'name_M',
        'sequence_5to3': 'sequence_M',
        'loop': 'loop_M',
        'intensity_median': 'intensity_median_M',
        'intensity_mean': 'intensity_mean_M',
        'intensity_std': 'intensity_std_M',
        'intensity_all': 'intensity_all_M',
        'n_replicates': 'n_replicates_M'
    }
    m_seqs = m_seqs.rename(columns=m_cols)

    # Merge A and M by pair_id
    paired = a_seqs.merge(
        m_seqs[['pair_id', 'name_M', 'sequence_M', 'loop_M',
                'intensity_median_M', 'intensity_mean_M', 'intensity_std_M',
                'intensity_all_M', 'n_replicates_M']],
        on='pair_id',
        how='inner'
    )

    logger.info(f"Paired dataset shape: {paired.shape}")

    # Calculate differential binding metrics
    paired['delta_intensity_median'] = paired['intensity_median_M'] - paired['intensity_median_A']
    paired['delta_intensity_mean'] = paired['intensity_mean_M'] - paired['intensity_mean_A']
    paired['fold_change_median'] = paired['intensity_median_M'] / paired['intensity_median_A']
    paired['fold_change_mean'] = paired['intensity_mean_M'] / paired['intensity_mean_A']
    paired['log2_fold_change_median'] = np.log2(paired['fold_change_median'])
    paired['log2_fold_change_mean'] = np.log2(paired['fold_change_mean'])

    # Extract context features from loop sequence
    # The loop is 7 nucleotides with center at position 4 (0-indexed: 3)
    paired['context_3nt_upstream'] = paired['loop_A'].str[:3]
    paired['context_3nt_downstream'] = paired['loop_A'].str[4:]

    # Check for DRACH motif (D=A/G/U, R=A/G, A, C, H=A/C/U)
    # Position mapping in 7mer: -3, -2, -1, center, +1, +2, +3
    # DRACH: D at -2, R at -1, A at center, C at +1, H at +2
    def is_drach(loop):
        if len(loop) != 7:
            return False
        d_pos = loop[1]  # -2 position
        r_pos = loop[2]  # -1 position
        center = loop[3]  # center (A)
        c_pos = loop[4]  # +1 position
        h_pos = loop[5]  # +2 position

        d_valid = d_pos in ['A', 'G', 'U']
        r_valid = r_pos in ['A', 'G']
        a_valid = center == 'A'
        c_valid = c_pos == 'C'
        h_valid = h_pos in ['A', 'C', 'U']

        return d_valid and r_valid and a_valid and c_valid and h_valid

    paired['is_drach'] = paired['loop_A'].apply(is_drach)

    # Reorder columns for clarity
    col_order = [
        # Identifiers
        'pair_id', 'name_A', 'name_M',
        # Sequences
        'sequence_A', 'sequence_M', 'loop_A', 'loop_M',
        # Metadata
        'type', 'type_custom', 'stem_len_each', 'total_len',
        # Intensity - A (unmethylated)
        'intensity_median_A', 'intensity_mean_A', 'intensity_std_A', 'n_replicates_A',
        # Intensity - M (methylated)
        'intensity_median_M', 'intensity_mean_M', 'intensity_std_M', 'n_replicates_M',
        # Differential metrics
        'delta_intensity_median', 'delta_intensity_mean',
        'fold_change_median', 'fold_change_mean',
        'log2_fold_change_median', 'log2_fold_change_mean',
        # Context features
        'context_3nt_upstream', 'context_3nt_downstream', 'is_drach',
        # All replicate values (for detailed analysis)
        'intensity_all_A', 'intensity_all_M'
    ]

    # Only include columns that exist
    col_order = [c for c in col_order if c in paired.columns]
    paired = paired[col_order]

    return paired


def process_control_sequences(design: pd.DataFrame, measurements_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Process G and U control sequences (no methylation site).
    These are negative controls without a center A.
    """
    logger.info("Processing G/U control sequences...")

    merged = design.merge(measurements_agg, on='name', how='inner')

    # Filter for G and U center sequences
    controls = merged[merged['center'].isin(['G', 'U'])].copy()
    logger.info(f"Control sequences: {len(controls)}")

    if len(controls) == 0:
        return pd.DataFrame()

    # Rename columns for consistency
    controls = controls.rename(columns={
        'sequence_5to3': 'sequence',
        'PM_635_26112025_nm': 'intensity'
    })

    return controls


def main():
    parser = argparse.ArgumentParser(description='Preprocess m6A YTH binding data')
    parser.add_argument('--data_dir', type=str,
                        default='/Users/shaharharel/Documents/github/edit-chem/data/rna/m6a',
                        help='Directory containing raw data files')
    parser.add_argument('--output_dir', type=str,
                        default='/Users/shaharharel/Documents/github/edit-chem/data/rna/m6a/processed',
                        help='Directory for processed output files')
    parser.add_argument('--measurement_file', type=str,
                        default='635_500_50_2.5res_YTHDF2100nM_26112025.tsv',
                        help='Measurement file to use (default: 500 sensitivity setting)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load files
    design_path = data_dir / 'hairpin_15_7_15_all7mers_centerA_and_M_plus_label.csv'
    measurement_path = data_dir / args.measurement_file

    design = load_design_file(design_path)
    measurements = load_measurement_file(measurement_path)

    # Aggregate replicates
    measurements_agg = aggregate_replicates(measurements)

    # Create paired dataset
    paired_df = create_paired_dataset(design, measurements_agg)

    # Process control sequences
    controls_df = process_control_sequences(design, measurements_agg)

    # Determine output suffix based on measurement file
    if '500' in args.measurement_file:
        suffix = 'sensitivity_500'
    elif '600' in args.measurement_file:
        suffix = 'sensitivity_600'
    else:
        suffix = 'default'

    # Save outputs
    paired_output = output_dir / f'm6a_paired_binding_{suffix}.csv'
    paired_df.to_csv(paired_output, index=False)
    logger.info(f"Saved paired dataset to: {paired_output}")

    # Save version without list columns for easier loading
    paired_simple = paired_df.drop(columns=['intensity_all_A', 'intensity_all_M'])
    paired_simple_output = output_dir / f'm6a_paired_binding_{suffix}_simple.csv'
    paired_simple.to_csv(paired_simple_output, index=False)
    logger.info(f"Saved simplified paired dataset to: {paired_simple_output}")

    if len(controls_df) > 0:
        controls_output = output_dir / f'm6a_controls_{suffix}.csv'
        controls_df.to_csv(controls_output, index=False)
        logger.info(f"Saved controls dataset to: {controls_output}")

    # Print summary statistics
    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Total paired sequences: {len(paired_df)}")
    logger.info(f"DRACH sequences: {paired_df['is_drach'].sum()}")
    logger.info(f"Non-DRACH sequences: {(~paired_df['is_drach']).sum()}")

    logger.info("\n=== Binding Statistics (Median) ===")
    logger.info(f"A (unmethylated) intensity: {paired_df['intensity_median_A'].mean():.2f} +/- {paired_df['intensity_median_A'].std():.2f}")
    logger.info(f"M (methylated) intensity: {paired_df['intensity_median_M'].mean():.2f} +/- {paired_df['intensity_median_M'].std():.2f}")
    logger.info(f"Log2 fold change: {paired_df['log2_fold_change_median'].mean():.3f} +/- {paired_df['log2_fold_change_median'].std():.3f}")

    logger.info("\n=== By Sequence Type ===")
    for seq_type in paired_df['type'].unique():
        subset = paired_df[paired_df['type'] == seq_type]
        logger.info(f"{seq_type}: n={len(subset)}, log2FC={subset['log2_fold_change_median'].mean():.3f}")

    return paired_df, controls_df


if __name__ == '__main__':
    main()

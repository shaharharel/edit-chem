#!/usr/bin/env python3
"""
Download and process MPRA 5' UTR data from Sample et al. 2019.

Paper: "Human 5' UTR design and variant effect prediction from a massively parallel translation assay"
GEO Accession: GSE114002

This script downloads the processed MPRA data and prepares it for
the edit-chem RNA modality experiments.

Usage:
    python scripts/extraction/download_mpra_5utr.py --output-dir data/rna/raw
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# GEO Dataset Information
# =============================================================================

GEO_ACCESSION = "GSE114002"

# Direct FTP links to supplementary files
# These are the processed data files from the paper
SUPPLEMENTARY_FILES = {
    # Random 5' UTR library with MRL measurements
    "random_library": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114002/suppl/GSE114002_random_library.csv.gz",
        "filename": "GSE114002_random_library.csv.gz",
        "description": "280k random 5' UTR sequences with Mean Ribosome Load (MRL)"
    },
    # Human 5' UTR variants
    "human_utrs": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114002/suppl/GSE114002_human_5utrs.csv.gz",
        "filename": "GSE114002_human_5utrs.csv.gz",
        "description": "Human 5' UTR sequences (first 50nt) with MRL"
    },
    # SNV variants with measured effects
    "snv_variants": {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE114nnn/GSE114002/suppl/GSE114002_snv_variants.csv.gz",
        "filename": "GSE114002_snv_variants.csv.gz",
        "description": "SNV variants with WT and variant MRL (direct Δ-labels)"
    }
}

# Alternative: Use GEOparse to download
def download_with_geoparse(output_dir: Path):
    """Download using GEOparse library (alternative method)."""
    try:
        import GEOparse

        logger.info(f"Downloading GSE114002 using GEOparse...")
        gse = GEOparse.get_GEO(geo=GEO_ACCESSION, destdir=str(output_dir))

        logger.info(f"GSE title: {gse.metadata['title'][0]}")
        logger.info(f"Samples: {len(gse.gsms)}")

        return gse

    except ImportError:
        logger.warning("GEOparse not installed. Install with: pip install GEOparse")
        return None


def download_file(url: str, output_path: Path, description: str = ""):
    """Download a file from URL with progress."""
    import urllib.request
    import gzip
    import shutil

    logger.info(f"Downloading: {description}")
    logger.info(f"  URL: {url}")
    logger.info(f"  Output: {output_path}")

    try:
        # Download
        temp_path = output_path.with_suffix('.tmp')

        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                if count % 100 == 0:
                    print(f"\r  Progress: {percent}%", end='', flush=True)

        urllib.request.urlretrieve(url, temp_path, reporthook)
        print()  # New line after progress

        # Move to final location
        shutil.move(temp_path, output_path)
        logger.info(f"  Downloaded: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        return True

    except Exception as e:
        logger.error(f"  Download failed: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def download_all_files(output_dir: Path):
    """Download all supplementary files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for key, info in SUPPLEMENTARY_FILES.items():
        output_path = output_dir / info['filename']

        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            downloaded[key] = output_path
        else:
            success = download_file(
                url=info['url'],
                output_path=output_path,
                description=info['description']
            )
            if success:
                downloaded[key] = output_path

    return downloaded


def create_sample_data(output_dir: Path):
    """
    Create sample MPRA data for testing when real data is not available.

    This generates synthetic data with realistic structure for development.
    """
    logger.info("Creating sample MPRA data for testing...")

    np.random.seed(42)
    n_sequences = 10000

    # Generate random 5' UTR sequences (50nt each)
    nucleotides = ['A', 'C', 'G', 'U']

    sequences = []
    for _ in range(n_sequences):
        seq = ''.join(np.random.choice(nucleotides, size=50))
        sequences.append(seq)

    # Generate MRL values (log-normal distribution, typical for MPRA)
    # MRL typically ranges from ~1 to ~10 (log2 scale: 0 to 3.3)
    mrl_values = np.random.lognormal(mean=0.5, sigma=0.5, size=n_sequences)
    mrl_values = np.clip(mrl_values, 0.5, 15)  # Realistic range

    # Create random library dataframe
    random_df = pd.DataFrame({
        'sequence': sequences,
        'MRL': mrl_values,
        'log2_MRL': np.log2(mrl_values)
    })

    # Create SNV pairs (reference → variant)
    n_pairs = 2000
    pair_data = []

    for i in range(n_pairs):
        # Pick a random reference sequence
        ref_idx = np.random.randint(0, n_sequences)
        ref_seq = sequences[ref_idx]
        ref_mrl = mrl_values[ref_idx]

        # Create SNV variant
        pos = np.random.randint(0, 50)
        old_nuc = ref_seq[pos]
        new_nuc = np.random.choice([n for n in nucleotides if n != old_nuc])

        var_seq = ref_seq[:pos] + new_nuc + ref_seq[pos+1:]

        # Generate variant MRL (correlated with reference, but with noise)
        delta_mrl = np.random.normal(0, 0.5)  # Most SNVs have small effect

        # Some positions have larger effects (simulate regulatory elements)
        if pos < 10:  # 5' end often important
            delta_mrl *= 1.5

        var_mrl = ref_mrl * np.exp(delta_mrl)
        var_mrl = np.clip(var_mrl, 0.5, 15)

        pair_data.append({
            'ref_sequence': ref_seq,
            'var_sequence': var_seq,
            'position': pos,
            'ref_nucleotide': old_nuc,
            'var_nucleotide': new_nuc,
            'ref_MRL': ref_mrl,
            'var_MRL': var_mrl,
            'delta_MRL': var_mrl - ref_mrl,
            'log2_delta_MRL': np.log2(var_mrl) - np.log2(ref_mrl)
        })

    snv_df = pd.DataFrame(pair_data)

    # Save files
    output_dir.mkdir(parents=True, exist_ok=True)

    random_path = output_dir / 'sample_random_library.csv'
    snv_path = output_dir / 'sample_snv_variants.csv'

    random_df.to_csv(random_path, index=False)
    snv_df.to_csv(snv_path, index=False)

    logger.info(f"Created sample random library: {random_path} ({len(random_df)} sequences)")
    logger.info(f"Created sample SNV variants: {snv_path} ({len(snv_df)} pairs)")

    return {
        'random_library': random_path,
        'snv_variants': snv_path
    }


def process_random_library(input_path: Path, output_dir: Path) -> pd.DataFrame:
    """
    Process the random 5' UTR library.

    Expected columns: sequence, MRL, replicate info, etc.
    """
    logger.info(f"Processing random library: {input_path}")

    # Read file (handle gzip)
    if str(input_path).endswith('.gz'):
        df = pd.read_csv(input_path, compression='gzip')
    else:
        df = pd.read_csv(input_path)

    logger.info(f"  Loaded {len(df)} sequences")
    logger.info(f"  Columns: {df.columns.tolist()}")

    # Standardize column names
    column_mapping = {
        'seq': 'sequence',
        'Sequence': 'sequence',
        'UTR': 'sequence',
        'utr': 'sequence',
        'mrl': 'MRL',
        'mean_ribosome_load': 'MRL',
        'Mean_Ribosome_Load': 'MRL'
    }

    df = df.rename(columns=column_mapping)

    # Ensure we have required columns
    if 'sequence' not in df.columns:
        # Try to find sequence column
        for col in df.columns:
            if df[col].dtype == object:
                sample = str(df[col].iloc[0]).upper()
                if set(sample) <= set('ACGUT'):
                    df['sequence'] = df[col]
                    break

    if 'MRL' not in df.columns:
        # Try to find MRL column
        for col in df.columns:
            if 'mrl' in col.lower() or 'ribosome' in col.lower():
                df['MRL'] = df[col]
                break

    # Add log2 MRL if not present
    if 'log2_MRL' not in df.columns and 'MRL' in df.columns:
        df['log2_MRL'] = np.log2(df['MRL'].clip(lower=0.001))

    # Convert to RNA (T -> U)
    if 'sequence' in df.columns:
        df['sequence'] = df['sequence'].str.upper().str.replace('T', 'U')

    # Save processed file
    output_path = output_dir / 'random_library_processed.csv'
    df.to_csv(output_path, index=False)

    logger.info(f"  Saved processed file: {output_path}")

    return df


def process_snv_variants(input_path: Path, output_dir: Path) -> pd.DataFrame:
    """
    Process SNV variants with WT and variant MRL.

    This creates the Δ-labeled pairs needed for training.
    """
    logger.info(f"Processing SNV variants: {input_path}")

    # Read file
    if str(input_path).endswith('.gz'):
        df = pd.read_csv(input_path, compression='gzip')
    else:
        df = pd.read_csv(input_path)

    logger.info(f"  Loaded {len(df)} variants")
    logger.info(f"  Columns: {df.columns.tolist()}")

    # Standardize column names
    column_mapping = {
        'wt_sequence': 'ref_sequence',
        'WT_sequence': 'ref_sequence',
        'reference': 'ref_sequence',
        'variant_sequence': 'var_sequence',
        'mutant_sequence': 'var_sequence',
        'wt_MRL': 'ref_MRL',
        'WT_MRL': 'ref_MRL',
        'variant_MRL': 'var_MRL',
        'mutant_MRL': 'var_MRL'
    }

    df = df.rename(columns=column_mapping)

    # Calculate delta if not present
    if 'delta_MRL' not in df.columns:
        if 'var_MRL' in df.columns and 'ref_MRL' in df.columns:
            df['delta_MRL'] = df['var_MRL'] - df['ref_MRL']

    if 'log2_delta_MRL' not in df.columns:
        if 'var_MRL' in df.columns and 'ref_MRL' in df.columns:
            df['log2_delta_MRL'] = np.log2(df['var_MRL'].clip(lower=0.001)) - \
                                   np.log2(df['ref_MRL'].clip(lower=0.001))

    # Convert to RNA
    for col in ['ref_sequence', 'var_sequence']:
        if col in df.columns:
            df[col] = df[col].str.upper().str.replace('T', 'U')

    # Save processed file
    output_path = output_dir / 'snv_variants_processed.csv'
    df.to_csv(output_path, index=False)

    logger.info(f"  Saved processed file: {output_path}")

    return df


def create_pairs_long_format(
    snv_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """
    Create long-format pairs file compatible with edit-chem framework.

    Output schema matches the small molecule pairs format:
    - seq_a, seq_b: Reference and variant sequences
    - edit_type, edit_position, edit_from, edit_to: Edit information
    - value_a, value_b, delta: Property values
    - property_name: Name of the measured property
    """
    logger.info("Creating long-format pairs file...")

    pairs_data = []

    for _, row in snv_df.iterrows():
        # Extract edit information
        ref_seq = row.get('ref_sequence', '')
        var_seq = row.get('var_sequence', '')

        # Find the SNV position
        edit_pos = -1
        edit_from = ''
        edit_to = ''

        if len(ref_seq) == len(var_seq):
            for i, (r, v) in enumerate(zip(ref_seq, var_seq)):
                if r != v:
                    edit_pos = i
                    edit_from = r
                    edit_to = v
                    break

        # Create pair record
        pair = {
            'seq_a': ref_seq,
            'seq_b': var_seq,
            'edit_type': 'SNV' if edit_pos >= 0 else 'complex',
            'edit_position': edit_pos,
            'edit_from': edit_from,
            'edit_to': edit_to,
            'value_a': row.get('ref_MRL', np.nan),
            'value_b': row.get('var_MRL', np.nan),
            'delta': row.get('delta_MRL', np.nan),
            'log2_delta': row.get('log2_delta_MRL', np.nan),
            'property_name': 'MRL_5UTR',
            'cell_type': 'HEK293T',
            'experiment_id': 'GSE114002'
        }

        pairs_data.append(pair)

    pairs_df = pd.DataFrame(pairs_data)

    # Remove rows with missing data
    pairs_df = pairs_df.dropna(subset=['seq_a', 'seq_b', 'delta'])

    # Save
    output_path = output_dir / 'mpra_5utr_pairs_long.csv'
    pairs_df.to_csv(output_path, index=False)

    logger.info(f"  Created {len(pairs_df)} pairs")
    logger.info(f"  Saved to: {output_path}")

    return pairs_df


def main():
    parser = argparse.ArgumentParser(
        description="Download and process MPRA 5' UTR data"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/rna/raw',
        help='Output directory for downloaded files'
    )
    parser.add_argument(
        '--processed-dir', '-p',
        type=str,
        default='data/rna/processed',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--pairs-dir',
        type=str,
        default='data/rna/pairs',
        help='Output directory for pairs files'
    )
    parser.add_argument(
        '--sample-data',
        action='store_true',
        help='Create sample data for testing (instead of downloading)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download, process existing files only'
    )

    args = parser.parse_args()

    # Convert to Path
    output_dir = Path(args.output_dir)
    processed_dir = Path(args.processed_dir)
    pairs_dir = Path(args.pairs_dir)

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MPRA 5' UTR Data Download and Processing")
    logger.info("=" * 70)
    logger.info(f"GEO Accession: {GEO_ACCESSION}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Processed directory: {processed_dir}")
    logger.info(f"Pairs directory: {pairs_dir}")
    logger.info("=" * 70)

    if args.sample_data:
        # Create sample data for testing
        downloaded = create_sample_data(processed_dir)

        # Process sample data
        if 'snv_variants' in downloaded:
            snv_df = pd.read_csv(downloaded['snv_variants'])
            create_pairs_long_format(snv_df, pairs_dir)

    else:
        # Download real data
        if not args.skip_download:
            downloaded = download_all_files(output_dir)
        else:
            downloaded = {
                key: output_dir / info['filename']
                for key, info in SUPPLEMENTARY_FILES.items()
                if (output_dir / info['filename']).exists()
            }

        # Process downloaded files
        if 'random_library' in downloaded:
            process_random_library(downloaded['random_library'], processed_dir)

        if 'snv_variants' in downloaded:
            snv_df = process_snv_variants(downloaded['snv_variants'], processed_dir)
            create_pairs_long_format(snv_df, pairs_dir)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Download and processing complete!")
    logger.info("=" * 70)
    logger.info(f"Raw data: {output_dir}")
    logger.info(f"Processed data: {processed_dir}")
    logger.info(f"Pairs data: {pairs_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Process complete MPRAu 3'UTR data from Griesemer/Ulirsch et al. (Cell 2021).

This script creates a comprehensive dataset including:
1. Natural variant pairs (ref/alt alleles) from Variant MPRAu Results
2. SNV tiling experiments (systematic single-nucleotide perturbations)
3. Deletion tiling experiments (systematic 5bp deletions)

All data supports multiple cell lines and is formatted for edit prediction
and direct property prediction workflows.

Data source:
    Griesemer et al. "Genome-wide functional screen of 3'UTR variants
    uncovers causal variants for human disease and evolution" Cell 2021
    https://doi.org/10.1016/j.cell.2021.08.025

Usage:
    python scripts/extraction/process_mprau_complete.py --output data/rna/pairs/mprau_3utr_complete.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import time
import logging
import json
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cell lines available in MPRAu data
CELL_LINES = ['HEK293FT', 'HEPG2', 'HMEC', 'K562', 'GM12878', 'SKNSH']


def fetch_sequence_ucsc(chrom: str, start: int, end: int, genome: str = 'hg19') -> str:
    """Fetch DNA sequence from UCSC DAS server."""
    if not chrom.startswith('chr'):
        chrom = f'chr{chrom}'

    url = f"https://genome.ucsc.edu/cgi-bin/das/{genome}/dna?segment={chrom}:{start},{end}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        for dna in root.iter('DNA'):
            seq = ''.join(dna.text.split()).upper()
            return seq
    except Exception as e:
        logger.warning(f"Failed to fetch {chrom}:{start}-{end}: {e}")
        return None
    return None


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq.upper()))


def dna_to_rna(seq: str) -> str:
    """Convert DNA to RNA (T -> U)."""
    return seq.upper().replace('T', 'U')


def load_sequence_cache(cache_path: Path) -> dict:
    """Load cached sequences."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_sequence_cache(cache: dict, cache_path: Path):
    """Save sequences to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f)


def process_variant_results(xlsx_path: str, cache: dict, fetch_seqs: bool = True) -> pd.DataFrame:
    """
    Process natural variant pairs from 'Variant MPRAu Results' sheet.

    Returns pairs for all cell lines with skew (allelic difference) measurements.
    """
    logger.info("Processing Variant MPRAu Results...")

    # Load sheets
    results_df = pd.read_excel(xlsx_path, sheet_name='Variant MPRAu Results')
    oligo_df = pd.read_excel(xlsx_path, sheet_name='Oligo Variant Info')

    # Merge on mpra_variant_id
    merged = pd.merge(results_df, oligo_df, on='mpra_variant_id', how='inner')

    # Filter to SNVs
    merged['is_snv'] = (merged['ref_allele'].str.len() == 1) & (merged['alt_allele'].str.len() == 1)
    snv_df = merged[merged['is_snv']].copy()

    # De-duplicate by variant_id (some have multiple oligo contexts)
    snv_df = snv_df.drop_duplicates(subset=['variant_id'])
    logger.info(f"Processing {len(snv_df)} unique SNVs")

    all_pairs = []

    for idx, row in tqdm(snv_df.iterrows(), total=len(snv_df), desc="Variant pairs"):
        try:
            chrom = str(row['chrom'])
            oligo_start = int(str(row['oligo_starts']).split(',')[0])
            oligo_end = int(str(row['oligo_ends']).split(',')[0])
            var_pos = int(row['var_start'])
            strand = row['strand']
            ref_allele = row['ref_allele']
            alt_allele = row['alt_allele']

            # Calculate variant position within oligo
            if strand == '+':
                var_idx = var_pos - oligo_start
            else:
                var_idx = oligo_end - var_pos

            # Get sequence
            cache_key = f"{chrom}:{oligo_start}-{oligo_end}"

            if fetch_seqs:
                if cache_key not in cache:
                    ref_seq = fetch_sequence_ucsc(chrom, oligo_start, oligo_end)
                    if ref_seq:
                        cache[cache_key] = ref_seq
                    time.sleep(0.05)
                else:
                    ref_seq = cache.get(cache_key)

                if not ref_seq or var_idx < 0 or var_idx >= len(ref_seq):
                    continue

                # Handle strand orientation
                if strand == '-':
                    ref_seq = reverse_complement(ref_seq)
                    ref_check = reverse_complement(ref_allele)
                    alt_for_sub = reverse_complement(alt_allele)
                else:
                    ref_check = ref_allele
                    alt_for_sub = alt_allele

                # Verify and create sequences
                if ref_seq[var_idx] != ref_check:
                    if ref_seq[var_idx] != ref_allele:
                        continue
                    alt_for_sub = alt_allele

                alt_seq = ref_seq[:var_idx] + alt_for_sub + ref_seq[var_idx+1:]
                ref_rna = dna_to_rna(ref_seq)
                alt_rna = dna_to_rna(alt_seq)
            else:
                ref_rna = None
                alt_rna = None

            # Create pair for each cell line
            for cell_line in CELL_LINES:
                skew_col = f'log2FoldChange_Skew_{cell_line}'
                padj_col = f'padj_Skew_{cell_line}'
                ref_col = f'log2FoldChange_Ref_{cell_line}'
                alt_col = f'log2FoldChange_Alt_{cell_line}'

                if pd.isna(row.get(skew_col)):
                    continue

                skew = row[skew_col]
                padj = row.get(padj_col, np.nan)
                ref_expr = row.get(ref_col, 0)
                alt_expr = row.get(alt_col, 0)

                pair = {
                    'seq_a': ref_rna,
                    'seq_b': alt_rna,
                    'edit_type': 'SNV',
                    'edit_position': var_idx,
                    'edit_from': ref_allele,
                    'edit_to': alt_allele,
                    'value_a': ref_expr if pd.notna(ref_expr) else 0,
                    'value_b': alt_expr if pd.notna(alt_expr) else skew,
                    'delta': skew,
                    'log2_delta': skew,
                    'padj': padj,
                    'property_name': f'3UTR_skew_{cell_line}',
                    'variant_id': row['variant_id'],
                    'mpra_variant_id': row['mpra_variant_id'],
                    'chrom': chrom,
                    'genomic_pos': f"chr{chrom}:{var_pos}",
                    'strand': strand,
                    'gene': row.get('gene_symbols', ''),
                    'cell_type': cell_line,
                    'experiment_id': 'MPRAu_Griesemer2021',
                    'data_type': 'variant',
                    'source': '3UTR'
                }
                all_pairs.append(pair)

        except Exception as e:
            logger.debug(f"Error processing variant {row.get('variant_id', 'unknown')}: {e}")
            continue

    pairs_df = pd.DataFrame(all_pairs)
    logger.info(f"Created {len(pairs_df)} variant pairs across {len(CELL_LINES)} cell lines")
    return pairs_df


def process_snv_tiling(xlsx_path: str, cache: dict, fetch_seqs: bool = True) -> pd.DataFrame:
    """
    Process SNV tiling data - systematic single nucleotide perturbations.

    This data contains mutations at every position around key variants,
    providing dense coverage for learning position-specific effects.
    """
    logger.info("Processing SNV Tiling Results...")

    tiling_df = pd.read_excel(xlsx_path, sheet_name='SNV Tiling Results')
    oligo_df = pd.read_excel(xlsx_path, sheet_name='Oligo Variant Info')

    logger.info(f"Loaded {len(tiling_df)} SNV tiling experiments")

    all_pairs = []

    # Get unique unperturbed oligo backgrounds
    oligo_backgrounds = tiling_df['oligo_id_unperturbed'].unique()
    logger.info(f"Processing {len(oligo_backgrounds)} oligo backgrounds")

    for _, row in tqdm(tiling_df.iterrows(), total=len(tiling_df), desc="SNV tiling"):
        try:
            oligo_id = row['oligo_id']
            unperturbed_id = row['oligo_id_unperturbed']
            offset = row['num_bases_offset_from_var']
            base_sub = row['base_substiution']
            chrom = str(row['chrom'])
            strand = row['strand']

            # Skip if no substitution info
            if pd.isna(base_sub):
                continue

            # Create pairs for each cell line
            for cell_line in CELL_LINES:
                lfc_col = f'log2FoldChange_{cell_line}'
                padj_col = f'padj_{cell_line}'

                if pd.isna(row.get(lfc_col)):
                    continue

                lfc = row[lfc_col]
                padj = row.get(padj_col, np.nan)

                pair = {
                    'seq_a': None,  # Would need to fetch unperturbed sequence
                    'seq_b': None,  # Would need to construct perturbed sequence
                    'edit_type': 'SNV',
                    'edit_position': int(offset) if pd.notna(offset) else 0,
                    'edit_from': '-',  # Original base unknown without sequence
                    'edit_to': base_sub,
                    'value_a': 0,
                    'value_b': lfc,
                    'delta': lfc,
                    'log2_delta': lfc,
                    'padj': padj,
                    'property_name': f'3UTR_tiling_snv_{cell_line}',
                    'variant_id': oligo_id,
                    'mpra_variant_id': unperturbed_id,
                    'chrom': chrom,
                    'genomic_pos': '',
                    'strand': strand,
                    'gene': '',
                    'cell_type': cell_line,
                    'experiment_id': 'MPRAu_Griesemer2021_tiling',
                    'data_type': 'snv_tiling',
                    'source': '3UTR'
                }
                all_pairs.append(pair)

        except Exception as e:
            logger.debug(f"Error processing tiling row: {e}")
            continue

    pairs_df = pd.DataFrame(all_pairs)
    logger.info(f"Created {len(pairs_df)} SNV tiling pairs")
    return pairs_df


def process_deletion_tiling(xlsx_path: str) -> pd.DataFrame:
    """
    Process deletion tiling data - systematic 5bp deletions.

    This provides data on deletion effects at different positions.
    """
    logger.info("Processing Deletion Tiling Results...")

    tiling_df = pd.read_excel(xlsx_path, sheet_name='Deletion Tiling Results')
    logger.info(f"Loaded {len(tiling_df)} deletion tiling experiments")

    all_pairs = []

    for _, row in tqdm(tiling_df.iterrows(), total=len(tiling_df), desc="Deletion tiling"):
        try:
            oligo_id = row['oligo_id']
            unperturbed_id = row['oligo_id_unperturbed']
            del_start = row['del_start_pos']
            del_end = row['del_end_pos']
            chrom = str(row['chrom'])
            strand = row['strand']

            del_size = int(del_end - del_start) if pd.notna(del_end) and pd.notna(del_start) else 5

            for cell_line in CELL_LINES:
                lfc_col = f'log2FoldChange_{cell_line}'
                padj_col = f'padj_{cell_line}'

                if pd.isna(row.get(lfc_col)):
                    continue

                lfc = row[lfc_col]
                padj = row.get(padj_col, np.nan)

                pair = {
                    'seq_a': None,
                    'seq_b': None,
                    'edit_type': 'deletion',
                    'edit_position': int(del_start) if pd.notna(del_start) else 0,
                    'edit_from': f'{del_size}bp',
                    'edit_to': '-',
                    'value_a': 0,
                    'value_b': lfc,
                    'delta': lfc,
                    'log2_delta': lfc,
                    'padj': padj,
                    'property_name': f'3UTR_tiling_del_{cell_line}',
                    'variant_id': oligo_id,
                    'mpra_variant_id': unperturbed_id,
                    'chrom': chrom,
                    'genomic_pos': '',
                    'strand': strand,
                    'gene': '',
                    'cell_type': cell_line,
                    'experiment_id': 'MPRAu_Griesemer2021_tiling',
                    'data_type': 'deletion_tiling',
                    'source': '3UTR'
                }
                all_pairs.append(pair)

        except Exception as e:
            logger.debug(f"Error processing deletion row: {e}")
            continue

    pairs_df = pd.DataFrame(all_pairs)
    logger.info(f"Created {len(pairs_df)} deletion tiling pairs")
    return pairs_df


def analyze_dataset(df: pd.DataFrame):
    """Print comprehensive analysis of the dataset."""
    print("\n" + "="*80)
    print("MPRAU 3'UTR COMPLETE DATASET ANALYSIS")
    print("="*80)

    print(f"\nTotal pairs: {len(df):,}")

    print(f"\n=== Data Types ===")
    print(df['data_type'].value_counts().to_string())

    print(f"\n=== Cell Lines ===")
    print(df['cell_type'].value_counts().to_string())

    print(f"\n=== Edit Types ===")
    print(df['edit_type'].value_counts().to_string())

    print(f"\n=== Properties ===")
    print(df['property_name'].value_counts().head(20).to_string())

    # Stats per data type
    for dtype in df['data_type'].unique():
        subset = df[df['data_type'] == dtype]
        delta = subset['delta'].dropna()
        print(f"\n=== {dtype} Delta Statistics ===")
        print(f"  Count: {len(delta):,}")
        print(f"  Mean: {delta.mean():.4f}")
        print(f"  Std: {delta.std():.4f}")
        print(f"  Range: [{delta.min():.4f}, {delta.max():.4f}]")

        sig = subset[subset['padj'] < 0.1]
        print(f"  Significant (padj<0.1): {len(sig):,} ({100*len(sig)/len(subset):.1f}%)")

    # Sequence coverage
    if 'seq_a' in df.columns:
        with_seqs = df['seq_a'].notna().sum()
        print(f"\n=== Sequence Coverage ===")
        print(f"  Pairs with sequences: {with_seqs:,} ({100*with_seqs/len(df):.1f}%)")

        if with_seqs > 0:
            seq_lens = df['seq_a'].dropna().str.len()
            print(f"  Sequence length range: {seq_lens.min()}-{seq_lens.max()} bp")


def main():
    parser = argparse.ArgumentParser(description='Process complete MPRAu 3\'UTR data')
    parser.add_argument('--input', '-i',
                       default='data/rna/raw/mprau_3utr/Table_S1.xlsx',
                       help='Path to Table S1 Excel file')
    parser.add_argument('--output', '-o',
                       default='data/rna/pairs/mprau_3utr_complete.csv',
                       help='Output path for pairs CSV')
    parser.add_argument('--output-variants-only', '-v',
                       default='data/rna/pairs/mprau_3utr_variants_multicell.csv',
                       help='Output path for variant pairs only')
    parser.add_argument('--no-sequences', action='store_true',
                       help='Skip fetching sequences')
    parser.add_argument('--variants-only', action='store_true',
                       help='Only process variant data, skip tiling')
    parser.add_argument('--cache-dir',
                       default='data/rna/raw/mprau_3utr/cache',
                       help='Directory for sequence cache')

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    input_path = project_root / args.input
    output_path = project_root / args.output
    output_variants_path = project_root / args.output_variants_only
    cache_path = project_root / args.cache_dir / 'sequence_cache.json'

    # Load sequence cache
    cache = load_sequence_cache(cache_path)
    logger.info(f"Loaded {len(cache)} cached sequences")

    all_dfs = []

    # Process variant results (natural variants)
    variant_df = process_variant_results(
        str(input_path),
        cache,
        fetch_seqs=not args.no_sequences
    )
    all_dfs.append(variant_df)

    # Save variants-only file
    output_variants_path.parent.mkdir(parents=True, exist_ok=True)
    variant_df.to_csv(output_variants_path, index=False)
    logger.info(f"Saved variant pairs to: {output_variants_path}")

    if not args.variants_only:
        # Process SNV tiling (systematic perturbations)
        snv_tiling_df = process_snv_tiling(
            str(input_path),
            cache,
            fetch_seqs=not args.no_sequences
        )
        all_dfs.append(snv_tiling_df)

        # Process deletion tiling
        del_tiling_df = process_deletion_tiling(str(input_path))
        all_dfs.append(del_tiling_df)

    # Combine all data
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Analyze
    analyze_dataset(combined_df)

    # Save cache
    save_sequence_cache(cache, cache_path)
    logger.info(f"Saved {len(cache)} sequences to cache")

    # Save combined dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved complete dataset to: {output_path}")

    return combined_df


if __name__ == '__main__':
    main()

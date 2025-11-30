#!/usr/bin/env python3
"""
Process MPRAu 3'UTR data from Griesemer et al. (Cell 2021).

This script:
1. Loads the MPRAu variant results from Table S1
2. Fetches 100bp oligo sequences from hg19 reference genome
3. Creates ref/alt pairs with allelic skew measurements
4. Outputs long-format pairs compatible with edit-chem framework

Data source:
    Griesemer et al. "Genome-wide functional screen of 3'UTR variants
    uncovers causal variants for human disease and evolution" Cell 2021
    https://doi.org/10.1016/j.cell.2021.08.025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_sequence_ucsc(chrom: str, start: int, end: int, genome: str = 'hg19') -> str:
    """
    Fetch DNA sequence from UCSC DAS server.

    Args:
        chrom: Chromosome (e.g., '1' or 'chr1')
        start: 1-based start position
        end: 1-based end position (inclusive)
        genome: Genome assembly (default: hg19)

    Returns:
        DNA sequence string (uppercase)
    """
    # Ensure chromosome has 'chr' prefix
    if not chrom.startswith('chr'):
        chrom = f'chr{chrom}'

    # UCSC DAS uses 0-based coordinates for start
    url = f"https://genome.ucsc.edu/cgi-bin/das/{genome}/dna?segment={chrom}:{start},{end}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse XML response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)

        # Find DNA sequence in response
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


def load_mprau_data(xlsx_path: str) -> pd.DataFrame:
    """
    Load and merge MPRAu data from Excel file.

    Returns DataFrame with variant results and oligo info merged.
    """
    logger.info(f"Loading MPRAu data from {xlsx_path}")

    # Load sheets
    results_df = pd.read_excel(xlsx_path, sheet_name='Variant MPRAu Results')
    oligo_df = pd.read_excel(xlsx_path, sheet_name='Oligo Variant Info')

    # Merge on mpra_variant_id
    merged = pd.merge(results_df, oligo_df, on='mpra_variant_id', how='inner')

    # Filter to SNVs only
    merged['is_snv'] = (merged['ref_allele'].str.len() == 1) & (merged['alt_allele'].str.len() == 1)
    snv_df = merged[merged['is_snv']].copy()

    logger.info(f"Loaded {len(snv_df)} SNVs from {len(merged)} total variants")

    return snv_df


def extract_pairs_with_sequences(
    snv_df: pd.DataFrame,
    cell_line: str = 'HEK293FT',
    fetch_sequences: bool = True,
    cache_dir: str = None
) -> pd.DataFrame:
    """
    Extract ref/alt pairs with sequences and allelic skew values.

    Args:
        snv_df: DataFrame with SNV variant info
        cell_line: Cell line to use for allelic skew (default: HEK293FT)
        fetch_sequences: Whether to fetch sequences from UCSC
        cache_dir: Directory to cache fetched sequences

    Returns:
        Long-format DataFrame with pairs
    """
    logger.info(f"Extracting pairs for cell line: {cell_line}")

    # Select relevant columns
    skew_col = f'log2FoldChange_Skew_{cell_line}'
    padj_col = f'padj_Skew_{cell_line}'
    ref_col = f'log2FoldChange_Ref_{cell_line}'
    alt_col = f'log2FoldChange_Alt_{cell_line}'

    # Drop rows with missing skew data
    valid_df = snv_df[snv_df[skew_col].notna()].copy()
    logger.info(f"Variants with valid skew data: {len(valid_df)}")

    pairs = []
    sequence_cache = {}

    # Load cache if exists
    if cache_dir:
        cache_path = Path(cache_dir) / 'sequence_cache.csv'
        if cache_path.exists():
            cache_df = pd.read_csv(cache_path)
            sequence_cache = dict(zip(cache_df['key'], cache_df['sequence']))
            logger.info(f"Loaded {len(sequence_cache)} cached sequences")

    # Group by unique variant_id to avoid duplicates from alternative backgrounds
    unique_variants = valid_df.drop_duplicates(subset=['variant_id'])
    logger.info(f"Processing {len(unique_variants)} unique variants")

    for idx, row in tqdm(unique_variants.iterrows(), total=len(unique_variants), desc="Processing variants"):
        try:
            # Parse coordinates
            chrom = str(row['chrom'])
            oligo_start = int(str(row['oligo_starts']).split(',')[0])
            oligo_end = int(str(row['oligo_ends']).split(',')[0])
            var_pos = int(row['var_start'])
            strand = row['strand']
            ref_allele = row['ref_allele']
            alt_allele = row['alt_allele']

            # Calculate variant position within oligo (0-indexed)
            if strand == '+':
                var_idx = var_pos - oligo_start
            else:
                var_idx = oligo_end - var_pos

            # Fetch or lookup sequence
            cache_key = f"{chrom}:{oligo_start}-{oligo_end}"

            if fetch_sequences:
                if cache_key in sequence_cache:
                    ref_seq = sequence_cache[cache_key]
                else:
                    ref_seq = fetch_sequence_ucsc(chrom, oligo_start, oligo_end)
                    if ref_seq:
                        sequence_cache[cache_key] = ref_seq
                    time.sleep(0.1)  # Rate limiting

                if not ref_seq:
                    continue

                # Handle strand
                if strand == '-':
                    ref_seq = reverse_complement(ref_seq)
                    # Also reverse complement the alleles for comparison
                    ref_check = reverse_complement(ref_allele)
                    alt_for_sub = reverse_complement(alt_allele)
                else:
                    ref_check = ref_allele
                    alt_for_sub = alt_allele

                # Verify ref allele matches
                if var_idx < 0 or var_idx >= len(ref_seq):
                    continue

                if ref_seq[var_idx] != ref_check:
                    # Try with original allele (some cases)
                    if ref_seq[var_idx] != ref_allele:
                        logger.debug(f"Ref mismatch at {cache_key}: expected {ref_check}, got {ref_seq[var_idx]}")
                        continue
                    alt_for_sub = alt_allele

                # Create alt sequence
                alt_seq = ref_seq[:var_idx] + alt_for_sub + ref_seq[var_idx+1:]

                # Convert to RNA
                ref_rna = dna_to_rna(ref_seq)
                alt_rna = dna_to_rna(alt_seq)
            else:
                ref_rna = None
                alt_rna = None

            # Get expression values
            skew = row[skew_col]  # log2(alt/ref) allelic skew
            padj = row[padj_col]

            # The skew is log2(alt/ref), so delta = skew
            # Positive skew = alt has higher expression

            pair = {
                'seq_a': ref_rna,  # Reference sequence
                'seq_b': alt_rna,  # Alternate sequence
                'edit_type': 'SNV',
                'edit_position': var_idx,
                'edit_from': ref_allele,
                'edit_to': alt_allele,
                'value_a': 0,  # Baseline (ref)
                'value_b': skew,  # Relative to ref (log2 scale)
                'delta': skew,  # log2(alt/ref)
                'log2_delta': skew,
                'padj': padj,
                'property_name': f'3UTR_activity_{cell_line}',
                'variant_id': row['variant_id'],
                'mpra_variant_id': row['mpra_variant_id'],
                'chrom': chrom,
                'genomic_pos': f"chr{chrom}:{var_pos}",
                'strand': strand,
                'gene_symbols': row.get('gene_symbols', ''),
                'cell_type': cell_line,
                'experiment_id': 'MPRAu_Griesemer2021',
                'source': '3UTR'
            }

            pairs.append(pair)

        except Exception as e:
            logger.debug(f"Error processing {row['variant_id']}: {e}")
            continue

    # Save cache
    if cache_dir and sequence_cache:
        cache_path = Path(cache_dir) / 'sequence_cache.csv'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df = pd.DataFrame([
            {'key': k, 'sequence': v} for k, v in sequence_cache.items()
        ])
        cache_df.to_csv(cache_path, index=False)
        logger.info(f"Saved {len(sequence_cache)} sequences to cache")

    pairs_df = pd.DataFrame(pairs)
    logger.info(f"Created {len(pairs_df)} pairs")

    return pairs_df


def analyze_pairs(pairs_df: pd.DataFrame):
    """Print analysis of extracted pairs."""
    print("\n" + "="*80)
    print("MPRAu 3'UTR PAIRS ANALYSIS")
    print("="*80)

    print(f"\nTotal pairs: {len(pairs_df)}")

    if 'seq_a' in pairs_df.columns and pairs_df['seq_a'].notna().any():
        seq_lens = pairs_df['seq_a'].dropna().str.len()
        print(f"Sequence length: {seq_lens.min()}-{seq_lens.max()} bp")

    print(f"\n=== Delta (log2 allelic skew) Statistics ===")
    delta = pairs_df['delta'].dropna()
    print(f"Mean: {delta.mean():.4f}")
    print(f"Std: {delta.std():.4f}")
    print(f"Range: [{delta.min():.4f}, {delta.max():.4f}]")
    print(f"Median: {delta.median():.4f}")

    print(f"\n=== Significant Variants (padj < 0.1) ===")
    sig = pairs_df[pairs_df['padj'] < 0.1]
    print(f"Count: {len(sig)} ({100*len(sig)/len(pairs_df):.1f}%)")

    print(f"\n=== Mutation Type Distribution ===")
    pairs_df['mutation_type'] = pairs_df['edit_from'] + '→' + pairs_df['edit_to']
    print(pairs_df['mutation_type'].value_counts().head(12).to_string())

    print(f"\n=== Mean Effect by Mutation Type ===")
    effect = pairs_df.groupby('mutation_type')['delta'].agg(['mean', 'count'])
    effect = effect.sort_values('mean', ascending=False)
    print(effect.to_string())


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Process MPRAu 3\'UTR data')
    parser.add_argument('--input', '-i',
                       default='data/rna/raw/mprau_3utr/Table_S1.xlsx',
                       help='Path to Table S1 Excel file')
    parser.add_argument('--output', '-o',
                       default='data/rna/pairs/mprau_3utr_pairs_long.csv',
                       help='Output path for pairs CSV')
    parser.add_argument('--cell-line', '-c',
                       default='HEK293FT',
                       choices=['HEK293FT', 'HEPG2', 'HMEC', 'K562', 'GM12878', 'SKNSH'],
                       help='Cell line to use')
    parser.add_argument('--no-sequences', action='store_true',
                       help='Skip fetching sequences (faster)')
    parser.add_argument('--cache-dir',
                       default='data/rna/raw/mprau_3utr/cache',
                       help='Directory for sequence cache')

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    input_path = project_root / args.input
    output_path = project_root / args.output
    cache_dir = project_root / args.cache_dir

    # Load data
    snv_df = load_mprau_data(input_path)

    # Extract pairs
    pairs_df = extract_pairs_with_sequences(
        snv_df,
        cell_line=args.cell_line,
        fetch_sequences=not args.no_sequences,
        cache_dir=cache_dir
    )

    # Analyze
    analyze_pairs(pairs_df)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_csv(output_path, index=False)
    print(f"\nSaved pairs to: {output_path}")


if __name__ == '__main__':
    main()

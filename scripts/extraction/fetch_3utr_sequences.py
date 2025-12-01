#!/usr/bin/env python3
"""
Fetch 3'UTR sequences from UCSC for MPRAu dataset.
Uses batch requests for efficiency.
"""

import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time
from pathlib import Path
import json


def fetch_sequence_ucsc(chrom, start: int, end: int, genome: str = 'hg19') -> str:
    """Fetch DNA sequence from UCSC DAS server."""
    chrom = str(chrom)
    if not chrom.startswith('chr'):
        chrom = f'chr{chrom}'

    url = f"https://genome.ucsc.edu/cgi-bin/das/{genome}/dna?segment={chrom}:{start},{end}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        for dna in root.iter('DNA'):
            return ''.join(dna.text.split()).upper()
    except Exception as e:
        return None

    return None


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq.upper()))


def dna_to_rna(seq: str) -> str:
    """Convert DNA to RNA (T -> U)."""
    return seq.upper().replace('T', 'U')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/rna/pairs/mprau_3utr_multitask.csv')
    parser.add_argument('--output', default='data/rna/pairs/mprau_3utr_multitask_with_seq.csv')
    parser.add_argument('--cache', default='data/rna/raw/mprau_3utr/sequence_cache.json')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of variants to process')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}")

    # Get unique genomic regions
    unique_regions = df.drop_duplicates(subset=['chrom', 'oligo_start', 'oligo_end', 'strand'])[
        ['variant_id', 'chrom', 'oligo_start', 'oligo_end', 'strand', 'edit_position', 'edit_from', 'edit_to']
    ].copy()
    print(f"  Unique regions: {len(unique_regions)}")

    if args.limit:
        unique_regions = unique_regions.head(args.limit)
        print(f"  Limited to: {len(unique_regions)}")

    # Load cache
    cache = {}
    cache_path = Path(args.cache)
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached sequences")

    # Fetch sequences
    seq_a_map = {}
    seq_b_map = {}

    for idx, row in tqdm(unique_regions.iterrows(), total=len(unique_regions), desc="Fetching sequences"):
        cache_key = f"{row['chrom']}:{row['oligo_start']}-{row['oligo_end']}"

        # Get or fetch sequence
        if cache_key in cache:
            ref_dna = cache[cache_key]
        else:
            ref_dna = fetch_sequence_ucsc(row['chrom'], row['oligo_start'], row['oligo_end'])
            if ref_dna:
                cache[cache_key] = ref_dna
            time.sleep(0.1)  # Rate limiting

        if not ref_dna:
            continue

        # Handle strand
        if row['strand'] == '-':
            ref_dna = reverse_complement(ref_dna)
            ref_check = reverse_complement(row['edit_from'])
            alt_sub = reverse_complement(row['edit_to'])
        else:
            ref_check = row['edit_from']
            alt_sub = row['edit_to']

        # Validate and create alt
        pos = int(row['edit_position'])
        if pos < 0 or pos >= len(ref_dna):
            continue

        if ref_dna[pos] != ref_check:
            continue

        alt_dna = ref_dna[:pos] + alt_sub + ref_dna[pos+1:]

        # Convert to RNA
        ref_rna = dna_to_rna(ref_dna)
        alt_rna = dna_to_rna(alt_dna)

        seq_a_map[row['variant_id']] = ref_rna
        seq_b_map[row['variant_id']] = alt_rna

    print(f"\nSuccessfully fetched {len(seq_a_map)} sequence pairs")

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f"Saved cache to {cache_path}")

    # Map sequences to dataframe
    df['seq_a'] = df['variant_id'].map(seq_a_map)
    df['seq_b'] = df['variant_id'].map(seq_b_map)

    # Filter to rows with sequences
    df_with_seq = df[df['seq_a'].notna()].copy()
    print(f"\nRows with sequences: {len(df_with_seq)} / {len(df)}")

    # Save
    df_with_seq.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    # Print sample
    print("\n--- Sample sequences ---")
    sample = df_with_seq.head(3)
    for _, row in sample.iterrows():
        pos = int(row['edit_position'])
        print(f"\n{row['variant_id']} ({row['edit_from']}→{row['edit_to']}):")
        print(f"  Ref: ...{row['seq_a'][max(0,pos-5):pos]}[{row['seq_a'][pos]}]{row['seq_a'][pos+1:min(len(row['seq_a']),pos+6)]}...")
        print(f"  Alt: ...{row['seq_b'][max(0,pos-5):pos]}[{row['seq_b'][pos]}]{row['seq_b'][pos+1:min(len(row['seq_b']),pos+6)]}...")


if __name__ == '__main__':
    main()

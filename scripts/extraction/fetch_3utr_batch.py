#!/usr/bin/env python3
"""
Batch fetch 3'UTR sequences from UCSC with parallel requests.
Optimized for speed while respecting rate limits.
"""

import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from pathlib import Path
import json
import argparse


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


def fetch_single_region(args):
    """Fetch a single region (for parallel execution)."""
    cache_key, chrom, start, end, strand, edit_pos, edit_from, edit_to, variant_id = args

    ref_dna = fetch_sequence_ucsc(chrom, start, end)
    if not ref_dna:
        return None

    # Handle strand
    if strand == '-':
        ref_dna = reverse_complement(ref_dna)
        ref_check = reverse_complement(edit_from) if len(edit_from) == 1 else edit_from
        alt_sub = reverse_complement(edit_to) if len(edit_to) == 1 else edit_to
    else:
        ref_check = edit_from
        alt_sub = edit_to

    # Validate position
    pos = int(edit_pos)
    if pos < 0 or pos >= len(ref_dna):
        return None

    # Check ref allele matches
    if len(ref_check) == 1 and ref_dna[pos] != ref_check:
        return None

    # Create alt sequence
    alt_dna = ref_dna[:pos] + alt_sub + ref_dna[pos+len(edit_from):]

    # Convert to RNA
    ref_rna = dna_to_rna(ref_dna)
    alt_rna = dna_to_rna(alt_dna)

    return {
        'cache_key': cache_key,
        'variant_id': variant_id,
        'seq_a': ref_rna,
        'seq_b': alt_rna,
        'ref_dna': ref_dna  # For caching
    }


def main():
    parser = argparse.ArgumentParser(description='Batch fetch 3UTR sequences')
    parser.add_argument('--input', default='data/rna/pairs/mprau_3utr_multitask.csv')
    parser.add_argument('--output', default='data/rna/pairs/mprau_3utr_multitask_with_seq.csv')
    parser.add_argument('--cache', default='data/rna/raw/mprau_3utr/sequence_cache.json')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    parser.add_argument('--batch-delay', type=float, default=0.5, help='Delay between batches (seconds)')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Total rows: {len(df)}")

    # Get unique regions
    unique_regions = df.drop_duplicates(subset=['variant_id'])[
        ['variant_id', 'chrom', 'oligo_start', 'oligo_end', 'strand', 'edit_position', 'edit_from', 'edit_to']
    ].copy()
    print(f"  Unique variants: {len(unique_regions)}")

    # Load cache
    cache = {}
    cache_path = Path(args.cache)
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached sequences")

    # Prepare fetch tasks (only for uncached)
    tasks = []
    cached_results = {}

    for _, row in unique_regions.iterrows():
        cache_key = f"{row['chrom']}:{row['oligo_start']}-{row['oligo_end']}"

        if cache_key in cache:
            # Already cached - process locally
            ref_dna = cache[cache_key]
            strand = row['strand']
            edit_pos = int(row['edit_position'])
            edit_from = row['edit_from']
            edit_to = row['edit_to']

            # Handle strand
            if strand == '-':
                ref_dna_proc = reverse_complement(ref_dna)
                ref_check = reverse_complement(edit_from) if len(edit_from) == 1 else edit_from
                alt_sub = reverse_complement(edit_to) if len(edit_to) == 1 else edit_to
            else:
                ref_dna_proc = ref_dna
                ref_check = edit_from
                alt_sub = edit_to

            if edit_pos >= 0 and edit_pos < len(ref_dna_proc):
                if len(ref_check) == 1 and ref_dna_proc[edit_pos] == ref_check:
                    alt_dna = ref_dna_proc[:edit_pos] + alt_sub + ref_dna_proc[edit_pos+len(edit_from):]
                    cached_results[row['variant_id']] = {
                        'seq_a': dna_to_rna(ref_dna_proc),
                        'seq_b': dna_to_rna(alt_dna)
                    }
        else:
            tasks.append((
                cache_key,
                row['chrom'],
                row['oligo_start'],
                row['oligo_end'],
                row['strand'],
                row['edit_position'],
                row['edit_from'],
                row['edit_to'],
                row['variant_id']
            ))

    print(f"  Already processed from cache: {len(cached_results)}")
    print(f"  Need to fetch: {len(tasks)}")

    # Fetch in parallel
    new_results = {}
    new_cache = {}

    if tasks:
        print(f"\nFetching {len(tasks)} sequences with {args.workers} workers...")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit tasks in batches to manage rate limiting
            batch_size = args.workers * 10
            futures = []

            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i+batch_size]
                batch_futures = {executor.submit(fetch_single_region, task): task for task in batch}
                futures.extend(batch_futures.items())

                # Small delay between batches
                if i + batch_size < len(tasks):
                    time.sleep(args.batch_delay)

            # Process results
            for future, task in tqdm(futures, desc="Processing"):
                try:
                    result = future.result(timeout=60)
                    if result:
                        new_results[result['variant_id']] = {
                            'seq_a': result['seq_a'],
                            'seq_b': result['seq_b']
                        }
                        new_cache[result['cache_key']] = result['ref_dna']
                except Exception as e:
                    pass

    # Combine results
    all_results = {**cached_results, **new_results}
    print(f"\nTotal variants with sequences: {len(all_results)}")

    # Update cache
    cache.update(new_cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f"Updated cache: {len(cache)} total sequences")

    # Map sequences to dataframe
    df['seq_a'] = df['variant_id'].map(lambda v: all_results.get(v, {}).get('seq_a'))
    df['seq_b'] = df['variant_id'].map(lambda v: all_results.get(v, {}).get('seq_b'))

    # Filter to rows with sequences
    df_with_seq = df[df['seq_a'].notna()].copy()
    print(f"Rows with sequences: {len(df_with_seq)} / {len(df)}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_seq.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Unique variants: {len(unique_regions)}")
    print(f"Variants with sequences: {len(all_results)}")
    print(f"Missing: {len(unique_regions) - len(all_results)}")
    print(f"Total rows in output: {len(df_with_seq)}")

    if len(df_with_seq) > 0:
        print(f"\nSequence length: {df_with_seq['seq_a'].str.len().iloc[0]} bp")
        print(f"Cell lines: {df_with_seq['cell_type'].nunique()}")


if __name__ == '__main__':
    main()

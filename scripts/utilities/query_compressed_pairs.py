#!/usr/bin/env python3
"""
Utility script to query compressed pairs data without decompressing the entire file.
"""

import csv
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import argparse


def load_indices(index_dir):
    """Load index mappings."""
    index_dir = Path(index_dir)

    with open(index_dir / 'smiles_index.json', 'r') as f:
        smiles_to_idx = json.load(f)
    idx_to_smiles = {v: k for k, v in smiles_to_idx.items()}

    with open(index_dir / 'chembl_to_target.json', 'r') as f:
        chembl_id_to_target = json.load(f)

    return smiles_to_idx, idx_to_smiles, chembl_id_to_target


def query_by_smiles(compressed_file, smiles, smiles_to_idx, idx_to_smiles,
                    chembl_id_to_target, limit=100):
    """Find all pairs involving a specific SMILES."""
    if smiles not in smiles_to_idx:
        print(f"SMILES not found in index: {smiles}")
        return []

    target_idx = smiles_to_idx[smiles]
    results = []

    with open(compressed_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mol_a_idx = int(row['mol_a_idx'])
            mol_b_idx = int(row['mol_b_idx'])

            if mol_a_idx == target_idx or mol_b_idx == target_idx:
                results.append({
                    'mol_a': idx_to_smiles[mol_a_idx],
                    'mol_b': idx_to_smiles[mol_b_idx],
                    'edit_name': row['edit_name'],
                    'property': row['property_name'],
                    'delta': float(row['delta']),
                    'target': chembl_id_to_target.get(row['target_chembl_id'], 'Unknown')
                })

                if len(results) >= limit:
                    break

    return results


def query_by_target(compressed_file, target_id, idx_to_smiles, chembl_id_to_target,
                    limit=100):
    """Find all pairs for a specific target."""
    results = []

    with open(compressed_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['target_chembl_id'] == target_id:
                mol_a_idx = int(row['mol_a_idx'])
                mol_b_idx = int(row['mol_b_idx'])

                results.append({
                    'mol_a': idx_to_smiles[mol_a_idx],
                    'mol_b': idx_to_smiles[mol_b_idx],
                    'edit_name': row['edit_name'],
                    'property': row['property_name'],
                    'delta': float(row['delta']),
                    'target': chembl_id_to_target.get(row['target_chembl_id'], 'Unknown')
                })

                if len(results) >= limit:
                    break

    return results


def query_by_property(compressed_file, property_name, idx_to_smiles,
                      chembl_id_to_target, limit=100):
    """Find all pairs for a specific property."""
    results = []

    with open(compressed_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['property_name'] == property_name:
                mol_a_idx = int(row['mol_a_idx'])
                mol_b_idx = int(row['mol_b_idx'])

                results.append({
                    'mol_a': idx_to_smiles[mol_a_idx],
                    'mol_b': idx_to_smiles[mol_b_idx],
                    'edit_name': row['edit_name'],
                    'property': row['property_name'],
                    'delta': float(row['delta']),
                    'target': chembl_id_to_target.get(row['target_chembl_id'], 'Unknown')
                })

                if len(results) >= limit:
                    break

    return results


def compute_statistics(compressed_file, idx_to_smiles, chembl_id_to_target):
    """Compute statistics about the compressed file."""
    print("Computing statistics (this may take a few minutes)...\n")

    stats = {
        'total_rows': 0,
        'unique_mol_a': set(),
        'unique_mol_b': set(),
        'unique_edits': set(),
        'properties': Counter(),
        'targets': Counter(),
        'delta_distribution': defaultdict(list),
    }

    with open(compressed_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            stats['total_rows'] += 1
            stats['unique_mol_a'].add(int(row['mol_a_idx']))
            stats['unique_mol_b'].add(int(row['mol_b_idx']))
            stats['unique_edits'].add(row['edit_name'])
            stats['properties'][row['property_name']] += 1
            stats['targets'][row['target_chembl_id']] += 1
            stats['delta_distribution'][row['property_name']].append(float(row['delta']))

            if i % 1000000 == 0 and i > 0:
                print(f"  Processed {i:,} rows...", end='\r')

    print(f"  Processed {stats['total_rows']:,} rows.    \n")

    # Print statistics
    print("="*70)
    print("COMPRESSED PAIRS STATISTICS")
    print("="*70)
    print(f"\nTotal pairs:           {stats['total_rows']:,}")
    print(f"Unique molecules (A):  {len(stats['unique_mol_a']):,}")
    print(f"Unique molecules (B):  {len(stats['unique_mol_b']):,}")
    print(f"Unique molecules (total): {len(stats['unique_mol_a'] | stats['unique_mol_b']):,}")
    print(f"Unique edits:          {len(stats['unique_edits']):,}")
    print(f"Unique properties:     {len(stats['properties']):,}")
    print(f"Unique targets:        {len(stats['targets']):,}")

    print(f"\nTop 10 properties by count:")
    for prop, count in stats['properties'].most_common(10):
        print(f"  {prop:50s} {count:>10,}")

    print(f"\nTop 10 targets by count:")
    for target_id, count in stats['targets'].most_common(10):
        target_name = chembl_id_to_target.get(target_id, 'Unknown')
        print(f"  {target_id:15s} {target_name[:40]:40s} {count:>10,}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Query compressed pairs data'
    )
    parser.add_argument('--smiles', type=str, help='Query by SMILES')
    parser.add_argument('--target', type=str, help='Query by target ChEMBL ID')
    parser.add_argument('--property', type=str, help='Query by property name')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--limit', type=int, default=100, help='Max results (default: 100)')
    parser.add_argument('--index-dir', type=str, help='Directory with index files')
    parser.add_argument('--compressed-file', type=str, help='Path to compressed CSV')

    args = parser.parse_args()

    # Default paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'

    index_dir = Path(args.index_dir) if args.index_dir else (
        data_dir / 'pairs' / 'checkpoints_50k_filtered' / 'compressed'
    )
    compressed_file = Path(args.compressed_file) if args.compressed_file else (
        index_dir / 'pairs_checkpoint_compressed.csv'
    )

    # Verify files exist
    if not compressed_file.exists():
        print(f"Error: Compressed file not found: {compressed_file}")
        sys.exit(1)

    if not index_dir.exists():
        print(f"Error: Index directory not found: {index_dir}")
        sys.exit(1)

    # Load indices
    print("Loading indices...")
    smiles_to_idx, idx_to_smiles, chembl_id_to_target = load_indices(index_dir)
    print(f"Loaded {len(idx_to_smiles):,} SMILES and {len(chembl_id_to_target):,} targets\n")

    # Execute query
    if args.stats:
        compute_statistics(compressed_file, idx_to_smiles, chembl_id_to_target)

    elif args.smiles:
        results = query_by_smiles(
            compressed_file, args.smiles, smiles_to_idx, idx_to_smiles,
            chembl_id_to_target, args.limit
        )
        print(f"Found {len(results)} pairs involving SMILES: {args.smiles}\n")
        for i, result in enumerate(results[:args.limit], 1):
            print(f"{i}. {result['property']}")
            print(f"   mol_a: {result['mol_a'][:60]}...")
            print(f"   mol_b: {result['mol_b'][:60]}...")
            print(f"   edit:  {result['edit_name'][:60]}...")
            print(f"   delta: {result['delta']:+.3f}")
            print(f"   target: {result['target'][:50]}")
            print()

    elif args.target:
        results = query_by_target(
            compressed_file, args.target, idx_to_smiles, chembl_id_to_target, args.limit
        )
        print(f"Found {len(results)} pairs for target: {args.target}\n")
        for i, result in enumerate(results[:args.limit], 1):
            print(f"{i}. {result['property']}")
            print(f"   mol_a: {result['mol_a'][:60]}...")
            print(f"   mol_b: {result['mol_b'][:60]}...")
            print(f"   edit:  {result['edit_name'][:60]}...")
            print(f"   delta: {result['delta']:+.3f}")
            print()

    elif args.property:
        results = query_by_property(
            compressed_file, args.property, idx_to_smiles, chembl_id_to_target, args.limit
        )
        print(f"Found {len(results)} pairs for property: {args.property}\n")
        for i, result in enumerate(results[:args.limit], 1):
            print(f"{i}. Target: {result['target'][:50]}")
            print(f"   mol_a: {result['mol_a'][:60]}...")
            print(f"   mol_b: {result['mol_b'][:60]}...")
            print(f"   edit:  {result['edit_name'][:60]}...")
            print(f"   delta: {result['delta']:+.3f}")
            print()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()

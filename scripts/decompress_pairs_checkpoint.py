#!/usr/bin/env python3
"""
Decompress pairs_checkpoint_compressed.csv back to original format.

This script reconstructs the original pairs_checkpoint.csv from the compressed version
using the saved index mappings.
"""

import csv
import json
import sys
from pathlib import Path
from tqdm import tqdm


def load_indices(index_dir):
    """Load index mappings from JSON files."""
    print("Loading index mappings...")

    index_dir = Path(index_dir)

    # Load SMILES index
    smiles_file = index_dir / 'smiles_index.json'
    print(f"  Loading SMILES index from {smiles_file}...")
    with open(smiles_file, 'r') as f:
        smiles_to_idx = json.load(f)
    idx_to_smiles = {v: k for k, v in smiles_to_idx.items()}
    print(f"    Loaded {len(idx_to_smiles):,} SMILES")

    # Load target index
    target_file = index_dir / 'target_index.json'
    print(f"  Loading target index from {target_file}...")
    with open(target_file, 'r') as f:
        target_id_to_idx = json.load(f)
    idx_to_target = {v: k for k, v in target_id_to_idx.items()}
    print(f"    Loaded {len(idx_to_target):,} targets")

    # Load ChEMBL ID to target name mapping
    mapping_file = index_dir / 'chembl_to_target.json'
    print(f"  Loading ChEMBL to target name mapping from {mapping_file}...")
    with open(mapping_file, 'r') as f:
        chembl_id_to_target = json.load(f)
    print(f"    Loaded {len(chembl_id_to_target):,} mappings")

    return idx_to_smiles, idx_to_target, chembl_id_to_target


def decompress_file(compressed_file, output_file, idx_to_smiles, chembl_id_to_target):
    """
    Decompress the compressed pairs file back to original format.

    Args:
        compressed_file: Path to compressed CSV
        output_file: Path to output decompressed CSV
        idx_to_smiles: Dict mapping index to SMILES
        chembl_id_to_target: Dict mapping target_chembl_id to target_name
    """
    print(f"\nDecompressing {compressed_file}...")
    print(f"  Output: {output_file}")

    # Count lines for progress bar
    print("  Counting lines...")
    with open(compressed_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f) - 1  # Subtract header

    print(f"  Processing {num_lines:,} rows...")

    with open(compressed_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)

        # Original header
        fieldnames = [
            'mol_a', 'mol_b', 'edit_smiles', 'edit_name',
            'property_name', 'value_a', 'value_b', 'delta',
            'target_name', 'target_chembl_id'
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process line by line
        for row in tqdm(reader, total=num_lines, desc="  Decompressing"):
            mol_a_idx = int(row['mol_a_idx'])
            mol_b_idx = int(row['mol_b_idx'])
            target_id = row['target_chembl_id']

            # Reconstruct original row
            decompressed_row = {
                'mol_a': idx_to_smiles[mol_a_idx],
                'mol_b': idx_to_smiles[mol_b_idx],
                'edit_smiles': row['edit_smiles'],
                'edit_name': row['edit_name'],
                'property_name': row['property_name'],
                'value_a': row['value_a'],
                'value_b': row['value_b'],
                'delta': row['delta'],
                'target_name': chembl_id_to_target.get(target_id, 'Unknown'),
                'target_chembl_id': target_id,
            }

            writer.writerow(decompressed_row)

    print(f"\n  Decompression complete!")


def main():
    """Main decompression pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Decompress pairs_checkpoint_compressed.csv back to original format'
    )
    parser.add_argument(
        '--compressed-file',
        type=str,
        help='Path to compressed CSV file',
        default=None
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='Path to output decompressed CSV file',
        default=None
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        help='Directory containing index JSON files',
        default=None
    )

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
    output_file = Path(args.output_file) if args.output_file else (
        data_dir / 'pairs' / 'checkpoints_50k_filtered' / 'pairs_checkpoint_decompressed.csv'
    )

    # Verify files exist
    if not compressed_file.exists():
        print(f"Error: Compressed file not found: {compressed_file}")
        sys.exit(1)

    if not index_dir.exists():
        print(f"Error: Index directory not found: {index_dir}")
        sys.exit(1)

    print("="*70)
    print("PAIRS CHECKPOINT DECOMPRESSION TOOL")
    print("="*70)
    print(f"\nInput:  {compressed_file}")
    print(f"Output: {output_file}")
    print(f"Indices: {index_dir}")
    print("\n" + "="*70 + "\n")

    # Load indices
    idx_to_smiles, idx_to_target, chembl_id_to_target = load_indices(index_dir)

    # Decompress file
    decompress_file(compressed_file, output_file, idx_to_smiles, chembl_id_to_target)

    print("\n✅ Decompression complete!")
    print(f"\nDecompressed file: {output_file}")


if __name__ == '__main__':
    main()

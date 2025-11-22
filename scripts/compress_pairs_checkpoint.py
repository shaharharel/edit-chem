#!/usr/bin/env python3
"""
Compress pairs_checkpoint.csv by creating index mappings.

This script processes a large CSV file line-by-line to avoid memory issues.
It creates three index mappings:
1. SMILES -> index
2. Target ChEMBL ID -> index
3. ChEMBL ID -> target name (for reference)

Original file: ~31GB with 91M rows
Compressed file: Replaces SMILES and target names with integer indices
"""

import csv
import sys
from pathlib import Path
from collections import OrderedDict
import json
from tqdm import tqdm


def build_indices_from_bioactivity(bioactivity_file, molecules_file):
    """
    Build indices from the source ChEMBL files.

    Returns:
        smiles_to_idx: dict mapping SMILES to integer index
        target_id_to_idx: dict mapping target_chembl_id to integer index
        chembl_id_to_target: dict mapping target_chembl_id to target_name
    """
    print("Step 1: Building indices from source files...")

    # Build SMILES index from molecules file
    print(f"  Reading SMILES from {molecules_file}...")
    smiles_to_idx = OrderedDict()
    smiles_idx = 0

    with open(molecules_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row['smiles']
            if smiles not in smiles_to_idx:
                smiles_to_idx[smiles] = smiles_idx
                smiles_idx += 1

    print(f"    Found {len(smiles_to_idx):,} unique SMILES")

    # Build target indices from bioactivity file
    print(f"  Reading targets from {bioactivity_file}...")
    target_id_to_idx = OrderedDict()
    chembl_id_to_target = {}
    target_idx = 0

    with open(bioactivity_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_id = row['target_chembl_id']
            target_name = row['target_name']

            if target_id not in target_id_to_idx:
                target_id_to_idx[target_id] = target_idx
                chembl_id_to_target[target_id] = target_name
                target_idx += 1

    print(f"    Found {len(target_id_to_idx):,} unique targets")

    return smiles_to_idx, target_id_to_idx, chembl_id_to_target


def save_indices(smiles_to_idx, target_id_to_idx, chembl_id_to_target, output_dir):
    """Save index mappings to JSON files for later reference."""
    print("\nStep 2: Saving index mappings...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save SMILES index
    smiles_file = output_dir / 'smiles_index.json'
    print(f"  Saving SMILES index to {smiles_file}...")
    with open(smiles_file, 'w') as f:
        json.dump(smiles_to_idx, f, indent=2)

    # Save target index
    target_file = output_dir / 'target_index.json'
    print(f"  Saving target index to {target_file}...")
    with open(target_file, 'w') as f:
        json.dump(target_id_to_idx, f, indent=2)

    # Save ChEMBL ID to target name mapping
    mapping_file = output_dir / 'chembl_to_target.json'
    print(f"  Saving ChEMBL ID to target name mapping to {mapping_file}...")
    with open(mapping_file, 'w') as f:
        json.dump(chembl_id_to_target, f, indent=2)

    print("  Index files saved successfully!")


def compress_pairs_file(input_file, output_file, smiles_to_idx, target_id_to_idx,
                        chembl_id_to_target):
    """
    Process pairs_checkpoint.csv line by line and create compressed version.

    Changes:
    - mol_a: SMILES -> integer index
    - mol_b: SMILES -> integer index
    - target_name: removed (can be recovered from target_chembl_id)
    - target_chembl_id: kept as-is (short strings like CHEMBL243)

    Optional: Could also replace target_chembl_id with index, but the IDs are short.
    """
    print(f"\nStep 3: Compressing {input_file}...")
    print(f"  Output: {output_file}")

    # Count lines for progress bar
    print("  Counting lines...")
    with open(input_file, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for _ in f) - 1  # Subtract header

    print(f"  Processing {num_lines:,} rows...")

    # Statistics
    stats = {
        'processed': 0,
        'missing_smiles': 0,
        'missing_target': 0,
        'new_smiles_added': 0,
    }

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)

        # New header (removed target_name, mol_a and mol_b are now indices)
        fieldnames = [
            'mol_a_idx', 'mol_b_idx', 'edit_smiles', 'edit_name',
            'property_name', 'value_a', 'value_b', 'delta', 'target_chembl_id'
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process line by line
        for row in tqdm(reader, total=num_lines, desc="  Compressing"):
            mol_a = row['mol_a']
            mol_b = row['mol_b']
            target_id = row['target_chembl_id']

            # Get or create SMILES indices
            if mol_a not in smiles_to_idx:
                smiles_to_idx[mol_a] = len(smiles_to_idx)
                stats['new_smiles_added'] += 1

            if mol_b not in smiles_to_idx:
                smiles_to_idx[mol_b] = len(smiles_to_idx)
                stats['new_smiles_added'] += 1

            mol_a_idx = smiles_to_idx[mol_a]
            mol_b_idx = smiles_to_idx[mol_b]

            # Check if target exists (should always exist if built correctly)
            if target_id not in target_id_to_idx:
                stats['missing_target'] += 1
                # Add it
                target_id_to_idx[target_id] = len(target_id_to_idx)
                chembl_id_to_target[target_id] = row.get('target_name', 'Unknown')

            # Write compressed row
            compressed_row = {
                'mol_a_idx': mol_a_idx,
                'mol_b_idx': mol_b_idx,
                'edit_smiles': row['edit_smiles'],
                'edit_name': row['edit_name'],
                'property_name': row['property_name'],
                'value_a': row['value_a'],
                'value_b': row['value_b'],
                'delta': row['delta'],
                'target_chembl_id': target_id,
            }

            writer.writerow(compressed_row)
            stats['processed'] += 1

    print(f"\n  Compression complete!")
    print(f"    Rows processed: {stats['processed']:,}")
    print(f"    New SMILES added: {stats['new_smiles_added']:,}")
    print(f"    Missing targets: {stats['missing_target']:,}")
    print(f"    Final SMILES vocabulary size: {len(smiles_to_idx):,}")
    print(f"    Final target count: {len(target_id_to_idx):,}")

    return stats


def estimate_compression_ratio(input_file, output_file):
    """Estimate compression ratio based on file sizes."""
    input_size = Path(input_file).stat().st_size
    output_size = Path(output_file).stat().st_size

    compression_ratio = input_size / output_size
    space_saved = input_size - output_size

    print(f"\n" + "="*70)
    print("COMPRESSION STATISTICS")
    print("="*70)
    print(f"Original file size:   {input_size / (1024**3):.2f} GB")
    print(f"Compressed file size: {output_size / (1024**3):.2f} GB")
    print(f"Compression ratio:    {compression_ratio:.2f}x")
    print(f"Space saved:          {space_saved / (1024**3):.2f} GB ({100 * space_saved / input_size:.1f}%)")
    print("="*70)


def main():
    """Main compression pipeline."""

    # File paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'

    bioactivity_file = data_dir / 'chembl_bulk' / 'chembl_bioactivity_long_563784.csv'
    molecules_file = data_dir / 'chembl_bulk' / 'chembl_molecules_563784.csv'
    input_file = data_dir / 'pairs' / 'checkpoints_50k_filtered' / 'pairs_checkpoint.csv'

    output_dir = data_dir / 'pairs' / 'checkpoints_50k_filtered' / 'compressed'
    output_file = output_dir / 'pairs_checkpoint_compressed.csv'

    # Verify input files exist
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    if not bioactivity_file.exists():
        print(f"Error: Bioactivity file not found: {bioactivity_file}")
        sys.exit(1)

    if not molecules_file.exists():
        print(f"Error: Molecules file not found: {molecules_file}")
        sys.exit(1)

    print("="*70)
    print("PAIRS CHECKPOINT COMPRESSION TOOL")
    print("="*70)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")
    print(f"\nInput file size: {input_file.stat().st_size / (1024**3):.2f} GB")
    print(f"Number of rows:  {sum(1 for _ in open(input_file)) - 1:,}")
    print("\n" + "="*70 + "\n")

    # Step 1: Build indices from source files
    smiles_to_idx, target_id_to_idx, chembl_id_to_target = build_indices_from_bioactivity(
        bioactivity_file, molecules_file
    )

    # Step 2: Save indices
    save_indices(smiles_to_idx, target_id_to_idx, chembl_id_to_target, output_dir)

    # Step 3: Compress pairs file
    stats = compress_pairs_file(
        input_file, output_file, smiles_to_idx, target_id_to_idx, chembl_id_to_target
    )

    # Update indices if new SMILES were added
    if stats['new_smiles_added'] > 0:
        print("\n  Updating SMILES index with newly discovered molecules...")
        save_indices(smiles_to_idx, target_id_to_idx, chembl_id_to_target, output_dir)

    # Step 4: Show compression statistics
    estimate_compression_ratio(input_file, output_file)

    print("\n✅ Compression complete!")
    print(f"\nCompressed file: {output_file}")
    print(f"Index files saved in: {output_dir}")


if __name__ == '__main__':
    main()

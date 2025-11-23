#!/usr/bin/env python3
"""
Analyze ChEMBL bioactivity data to find all biological properties
with more than 1,000 molecules.

This script reads the bioactivity CSV and provides detailed statistics
about which properties have sufficient data for ML training.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List
import argparse


def analyze_properties(bioactivity_file: Path, min_molecules: int = 1000) -> pd.DataFrame:
    """
    Analyze properties in ChEMBL bioactivity data.

    Args:
        bioactivity_file: Path to chembl_bioactivity_long_*.csv
        min_molecules: Minimum number of unique molecules per property

    Returns:
        DataFrame with property statistics
    """
    print(f"Loading bioactivity data from: {bioactivity_file}")
    df = pd.read_csv(bioactivity_file)

    print(f"Total bioactivity measurements: {len(df):,}")
    print(f"Unique molecules (chembl_id): {df['chembl_id'].nunique():,}")
    print(f"Unique properties: {df['property_name'].nunique():,}")
    print()

    # Group by property and count unique molecules
    property_stats = df.groupby('property_name').agg({
        'chembl_id': 'nunique',  # unique molecules
        'pchembl_value': ['count', 'mean', 'std', 'min', 'max'],
        'target_name': lambda x: x.iloc[0] if len(x) > 0 else '',
        'target_chembl_id': lambda x: x.iloc[0] if len(x) > 0 else '',
        'activity_type': lambda x: x.iloc[0] if len(x) > 0 else ''
    }).reset_index()

    # Flatten column names
    property_stats.columns = [
        'property_name',
        'unique_molecules',
        'total_measurements',
        'mean_value',
        'std_value',
        'min_value',
        'max_value',
        'target_name',
        'target_chembl_id',
        'activity_type'
    ]

    # Sort by unique molecules
    property_stats = property_stats.sort_values('unique_molecules', ascending=False)

    # Filter for properties with enough molecules
    filtered_stats = property_stats[property_stats['unique_molecules'] >= min_molecules].copy()

    # Categorize properties
    def categorize_property(row):
        activity = row['activity_type']
        if activity in ['IC50', 'Ki', 'EC50', 'Kd']:
            return f'experimental_{activity}'
        elif activity == 'computed':
            return 'computed'
        else:
            return 'other'

    filtered_stats['category'] = filtered_stats.apply(categorize_property, axis=1)

    return filtered_stats, property_stats


def print_summary(filtered_stats: pd.DataFrame, min_molecules: int):
    """Print summary statistics."""
    print("=" * 100)
    print(f"PROPERTIES WITH >= {min_molecules:,} MOLECULES")
    print("=" * 100)
    print()

    # By category
    print("BY CATEGORY:")
    print("-" * 100)
    category_counts = filtered_stats.groupby('category').agg({
        'property_name': 'count',
        'unique_molecules': ['sum', 'mean', 'max']
    })
    print(category_counts)
    print()

    # Computed properties
    computed = filtered_stats[filtered_stats['category'] == 'computed']
    if len(computed) > 0:
        print("COMPUTED PROPERTIES:")
        print("-" * 100)
        for _, row in computed.iterrows():
            print(f"  {row['property_name']:30s}  {row['unique_molecules']:>8,} molecules")
        print()

    # Experimental properties by activity type
    for activity_type in ['IC50', 'Ki', 'EC50', 'Kd']:
        exp_props = filtered_stats[filtered_stats['category'] == f'experimental_{activity_type}']
        if len(exp_props) > 0:
            print(f"EXPERIMENTAL PROPERTIES ({activity_type}):")
            print("-" * 100)
            print(f"{'Property Name':<50} {'Target':<30} {'Molecules':>10} {'Measurements':>12}")
            print("-" * 100)
            for _, row in exp_props.head(20).iterrows():  # Top 20
                target = row['target_name'][:28] if len(row['target_name']) > 28 else row['target_name']
                print(f"{row['property_name']:<50} {target:<30} {row['unique_molecules']:>10,} {row['total_measurements']:>12,}")

            if len(exp_props) > 20:
                print(f"  ... and {len(exp_props) - 20} more {activity_type} properties")
            print()


def save_results(filtered_stats: pd.DataFrame, all_stats: pd.DataFrame, output_dir: Path):
    """Save analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save filtered properties
    filtered_path = output_dir / "properties_with_1k_molecules.csv"
    filtered_stats.to_csv(filtered_path, index=False)
    print(f"✓ Saved filtered properties to: {filtered_path}")

    # Save all properties
    all_path = output_dir / "all_properties_stats.csv"
    all_stats.to_csv(all_path, index=False)
    print(f"✓ Saved all property statistics to: {all_path}")

    # Save property lists by category
    for category in filtered_stats['category'].unique():
        props = filtered_stats[filtered_stats['category'] == category]['property_name'].tolist()
        category_path = output_dir / f"properties_{category}.txt"
        with open(category_path, 'w') as f:
            for prop in props:
                f.write(f"{prop}\n")
        print(f"✓ Saved {category} properties to: {category_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ChEMBL bioactivity properties")
    parser.add_argument(
        '--bioactivity-file',
        type=Path,
        default=Path('data/chembl_bulk/chembl_bioactivity_long_563784.csv'),
        help='Path to bioactivity CSV file'
    )
    parser.add_argument(
        '--min-molecules',
        type=int,
        default=1000,
        help='Minimum number of molecules per property'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/analysis'),
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Check file exists
    if not args.bioactivity_file.exists():
        print(f"Error: File not found: {args.bioactivity_file}")
        print("\nAvailable bioactivity files:")
        for f in Path('data/chembl_bulk').glob('chembl_bioactivity_long_*.csv'):
            print(f"  {f}")
        sys.exit(1)

    # Analyze
    filtered_stats, all_stats = analyze_properties(args.bioactivity_file, args.min_molecules)

    # Print summary
    print_summary(filtered_stats, args.min_molecules)

    # Print totals
    print("=" * 100)
    print("SUMMARY:")
    print(f"  Total properties with >= {args.min_molecules:,} molecules: {len(filtered_stats)}")
    print(f"  Total unique molecules covered: {filtered_stats['unique_molecules'].sum():,}")
    print("=" * 100)
    print()

    # Save results
    save_results(filtered_stats, all_stats, args.output_dir)


if __name__ == '__main__':
    main()

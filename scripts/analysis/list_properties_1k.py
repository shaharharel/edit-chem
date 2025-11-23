#!/usr/bin/env python3
"""
Quick script to list all biological properties with >1k molecules.
"""

import pandas as pd
from pathlib import Path

def main():
    # Load data
    props_file = Path('data/analysis/properties_with_1k_molecules.csv')

    if not props_file.exists():
        print("Error: Run 'python scripts/analyze_chembl_properties.py' first")
        return

    df = pd.read_csv(props_file)

    print("=" * 100)
    print("CHEMBL BIOLOGICAL PROPERTIES WITH >1,000 MOLECULES")
    print("=" * 100)
    print()

    # Summary
    print(f"Total properties: {len(df)}")
    print()

    # Group by activity type
    for activity_type in ['IC50', 'Ki', 'EC50', 'Kd']:
        subset = df[df['activity_type'] == activity_type].sort_values('unique_molecules', ascending=False)

        if len(subset) > 0:
            print(f"\n{activity_type} PROPERTIES ({len(subset)} total):")
            print("-" * 100)
            print(f"{'#':<4} {'Property Name':<50} {'Target':<35} {'Molecules':>10}")
            print("-" * 100)

            for idx, (_, row) in enumerate(subset.iterrows(), 1):
                target = row['target_name'][:33] if len(row['target_name']) > 33 else row['target_name']
                prop_name = row['property_name'][:48] if len(row['property_name']) > 48 else row['property_name']
                print(f"{idx:<4} {prop_name:<50} {target:<35} {row['unique_molecules']:>10,}")

    print()
    print("=" * 100)
    print(f"See data/analysis/CHEMBL_PROPERTIES_SUMMARY.md for detailed analysis")
    print("=" * 100)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Generate property list with molecule counts for Kinases, GPCRs, and Enzymes.
"""

import pandas as pd
from pathlib import Path

def main():
    # Load data
    props_file = Path('data/analysis/kinases_gpcrs_enzymes_1k.csv')

    if not props_file.exists():
        print("Error: Run 'python scripts/filter_kinases_gpcrs_enzymes.py' first")
        return

    df = pd.read_csv(props_file)
    df = df.sort_values(['target_class', 'unique_molecules'], ascending=[True, False])

    output_dir = Path('data/analysis')

    # Generate formatted property list with counts
    output_file = output_dir / 'properties_kinases_gpcrs_enzymes_with_counts.txt'

    with open(output_file, 'w') as f:
        f.write("# Kinases, GPCRs, and Enzymes - Properties with Molecule Counts\n")
        f.write("# Format: property_name,molecules,target_class,activity_type,target_name\n")
        f.write("#\n")
        f.write(f"# Total: {len(df)} properties\n")
        f.write(f"# Kinases: {len(df[df['target_class'] == 'Kinase'])}\n")
        f.write(f"# GPCRs: {len(df[df['target_class'] == 'GPCR'])}\n")
        f.write(f"# Enzymes: {len(df[df['target_class'] == 'Enzyme'])}\n")
        f.write("#\n\n")

        # Write each category
        for target_class in ['Kinase', 'GPCR', 'Enzyme']:
            subset = df[df['target_class'] == target_class]

            if len(subset) > 0:
                f.write(f"# === {target_class.upper()}S ({len(subset)} properties) ===\n")
                f.write(f"# Total molecules: {subset['unique_molecules'].sum():,}\n\n")

                for _, row in subset.iterrows():
                    f.write(f"{row['property_name']},{row['unique_molecules']},{row['target_class']},{row['activity_type']},{row['target_name']}\n")

                f.write("\n")

    print(f"✓ Generated: {output_file}")

    # Also generate a simple CSV version
    csv_file = output_dir / 'properties_kinases_gpcrs_enzymes_with_counts.csv'
    df[['property_name', 'unique_molecules', 'target_class', 'activity_type', 'target_name']].to_csv(csv_file, index=False)
    print(f"✓ Generated: {csv_file}")

    # Generate summary by target class
    summary_file = output_dir / 'properties_summary_by_class.txt'

    with open(summary_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("KINASES, GPCRs, AND ENZYMES - SUMMARY BY CLASS\n")
        f.write("=" * 100 + "\n\n")

        for target_class in ['Kinase', 'GPCR', 'Enzyme']:
            subset = df[df['target_class'] == target_class].sort_values('unique_molecules', ascending=False)

            if len(subset) > 0:
                f.write(f"\n{target_class.upper()}S ({len(subset)} properties)\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Property':<50} {'Molecules':>10}  {'Type':>6}  {'Target'}\n")
                f.write("-" * 100 + "\n")

                for _, row in subset.iterrows():
                    prop_name = row['property_name'][:48] if len(row['property_name']) > 48 else row['property_name']
                    target = row['target_name'][:40] if len(row['target_name']) > 40 else row['target_name']
                    f.write(f"{prop_name:<50} {row['unique_molecules']:>10,}  {row['activity_type']:>6}  {target}\n")

                f.write("\n")
                f.write(f"Total molecules in {target_class}s: {subset['unique_molecules'].sum():,}\n")
                f.write(f"Average molecules per property: {subset['unique_molecules'].mean():.0f}\n")

    print(f"✓ Generated: {summary_file}")

    print(f"\n✓ All files generated in: {output_dir}/")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Filter ChEMBL properties for Kinases, GPCRs, and Enzymes with >1,000 molecules.
"""

import pandas as pd
from pathlib import Path

def classify_target(target_name):
    """Classify target into Kinase, GPCR, Enzyme, or Other."""
    target_lower = target_name.lower()

    # Kinases
    kinase_keywords = ['kinase', 'jak', 'tyrosine-protein', 'serine/threonine']
    if any(kw in target_lower for kw in kinase_keywords):
        return 'Kinase'

    # GPCRs
    gpcr_keywords = [
        'receptor', 'adrenergic', 'dopamine', 'serotonin', 'opioid',
        'adenosine', 'cannabinoid', 'chemokine', 'orexin', 'apelin',
        'melanocortin', 'metabotropic', 'purinoceptor', 'sphingosine',
        'free fatty acid receptor', 'bile acid receptor'
    ]
    if any(kw in target_lower for kw in gpcr_keywords):
        # Exclude kinase receptors
        if 'kinase' not in target_lower and 'tyrosine-protein' not in target_lower:
            return 'GPCR'

    # Enzymes
    enzyme_keywords = [
        'dehydrogenase', 'oxidase', 'transferase', 'hydrolase', 'synthase',
        'reductase', 'hydroxylase', 'peptidase', 'protease', 'esterase',
        'deacetylase', 'demethylase', 'carboxylase', 'polymerase',
        'secretase', 'carbonic anhydrase', 'cytochrome', 'cholinesterase',
        'acetylcholinesterase', 'aromatase', 'renin', 'kallikrein',
        'collagenase', 'metalloproteinase', 'phosphodiesterase'
    ]
    if any(kw in target_lower for kw in enzyme_keywords):
        return 'Enzyme'

    return 'Other'

def main():
    # Load data
    props_file = Path('data/analysis/properties_with_1k_molecules.csv')

    if not props_file.exists():
        print("Error: Run 'python scripts/analyze_chembl_properties.py' first")
        return

    df = pd.read_csv(props_file)

    # Classify targets
    df['target_class'] = df['target_name'].apply(classify_target)

    # Filter for Kinases, GPCRs, and Enzymes
    filtered = df[df['target_class'].isin(['Kinase', 'GPCR', 'Enzyme'])].copy()
    filtered = filtered.sort_values(['target_class', 'unique_molecules'], ascending=[True, False])

    print("=" * 100)
    print("KINASES, GPCRs, AND ENZYMES WITH >1,000 MOLECULES")
    print("=" * 100)
    print()

    # Summary stats
    summary = filtered.groupby('target_class').agg({
        'property_name': 'count',
        'unique_molecules': ['sum', 'mean', 'min', 'max']
    })
    print("SUMMARY:")
    print("-" * 100)
    print(summary)
    print()

    # Print each category
    for target_class in ['Kinase', 'GPCR', 'Enzyme']:
        subset = filtered[filtered['target_class'] == target_class]

        if len(subset) > 0:
            print(f"\n{target_class.upper()}S ({len(subset)} properties):")
            print("-" * 100)
            print(f"{'#':<4} {'Property Name':<50} {'Target':<35} {'Type':>6} {'Molecules':>10}")
            print("-" * 100)

            for idx, (_, row) in enumerate(subset.iterrows(), 1):
                target = row['target_name'][:33] if len(row['target_name']) > 33 else row['target_name']
                prop_name = row['property_name'][:48] if len(row['property_name']) > 48 else row['property_name']
                print(f"{idx:<4} {prop_name:<50} {target:<35} {row['activity_type']:>6} {row['unique_molecules']:>10,}")

    # Save results
    output_dir = Path('data/analysis')

    # Save full filtered dataset
    output_file = output_dir / 'kinases_gpcrs_enzymes_1k.csv'
    filtered.to_csv(output_file, index=False)
    print(f"\n✓ Saved full dataset to: {output_file}")

    # Save property name lists by class
    for target_class in ['Kinase', 'GPCR', 'Enzyme']:
        subset = filtered[filtered['target_class'] == target_class]
        if len(subset) > 0:
            props_file = output_dir / f'properties_{target_class.lower()}.txt'
            with open(props_file, 'w') as f:
                for prop in subset['property_name']:
                    f.write(f"{prop}\n")
            print(f"✓ Saved {target_class} property names to: {props_file}")

    # Save combined property list
    combined_file = output_dir / 'properties_kinases_gpcrs_enzymes.txt'
    with open(combined_file, 'w') as f:
        for prop in filtered['property_name']:
            f.write(f"{prop}\n")
    print(f"✓ Saved combined property list to: {combined_file}")

    print()
    print("=" * 100)
    print(f"TOTAL: {len(filtered)} properties across Kinases, GPCRs, and Enzymes")
    print(f"  Kinases: {len(filtered[filtered['target_class'] == 'Kinase'])}")
    print(f"  GPCRs: {len(filtered[filtered['target_class'] == 'GPCR'])}")
    print(f"  Enzymes: {len(filtered[filtered['target_class'] == 'Enzyme'])}")
    print("=" * 100)

if __name__ == '__main__':
    main()

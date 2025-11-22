"""
Analyze ambiguous edit SMILES in the dataset.

An edit is ambiguous if the reactant fragment appears multiple times in mol_a,
making it unclear where to perform the edit.

Example:
  mol_a: CC(C)CC(C)C  (two identical branching patterns)
  edit_smiles: C(C)>>C(C)O
  → Ambiguous! Which C(C) group should be modified?

Run: python scripts/analyze_ambiguous_edits.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from tqdm import tqdm


def parse_edit_smiles(edit_smiles):
    """Parse edit_smiles into reactant and product fragments."""
    if not isinstance(edit_smiles, str) or '>>' not in edit_smiles:
        return None, None

    parts = edit_smiles.split('>>')
    if len(parts) != 2:
        return None, None

    reactant_smiles = parts[0].strip()
    product_smiles = parts[1].strip()

    return reactant_smiles, product_smiles


def count_substructure_matches(mol, pattern):
    """
    Count how many times a substructure pattern appears in a molecule.

    Args:
        mol: RDKit molecule
        pattern: RDKit molecule (substructure pattern)

    Returns:
        Number of matches (0 if pattern not found or invalid)
    """
    if mol is None or pattern is None:
        return 0

    try:
        matches = mol.GetSubstructMatches(pattern)
        return len(matches)
    except:
        return 0


def is_edit_ambiguous(mol_a_smiles, edit_smiles, verbose=False):
    """
    Check if an edit is ambiguous.

    An edit is ambiguous if the reactant fragment appears multiple times
    in mol_a, making it unclear where to apply the edit.

    Args:
        mol_a_smiles: SMILES of molecule A (starting molecule)
        edit_smiles: Edit SMILES (reactant>>product)
        verbose: If True, print details

    Returns:
        Tuple of (is_ambiguous, num_matches, reactant_smiles)
    """
    # Parse edit
    reactant_smiles, product_smiles = parse_edit_smiles(edit_smiles)

    if reactant_smiles is None:
        return False, 0, None

    # Convert to RDKit molecules
    mol_a = Chem.MolFromSmiles(mol_a_smiles)
    reactant = Chem.MolFromSmiles(reactant_smiles)

    if mol_a is None or reactant is None:
        return False, 0, None

    # Count matches
    num_matches = count_substructure_matches(mol_a, reactant)

    # Ambiguous if matches > 1
    is_ambiguous = num_matches > 1

    if verbose and is_ambiguous:
        print(f"\n{'='*80}")
        print(f"AMBIGUOUS EDIT FOUND")
        print(f"{'='*80}")
        print(f"Molecule A:     {mol_a_smiles}")
        print(f"Edit SMILES:    {edit_smiles}")
        print(f"Reactant frag:  {reactant_smiles}")
        print(f"Matches found:  {num_matches}")
        print(f"→ The reactant fragment appears {num_matches} times in mol_a!")

    return is_ambiguous, num_matches, reactant_smiles


def analyze_dataset(df, sample_size=None, verbose_examples=5):
    """
    Analyze entire dataset for ambiguous edits.

    Args:
        df: DataFrame with columns ['mol_a', 'edit_smiles']
        sample_size: If provided, only analyze this many rows (for speed)
        verbose_examples: Number of ambiguous examples to print

    Returns:
        Dictionary with analysis results
    """
    print("="*80)
    print("ANALYZING DATASET FOR AMBIGUOUS EDITS")
    print("="*80)

    # Sample if needed
    if sample_size and len(df) > sample_size:
        print(f"\nSampling {sample_size:,} rows from {len(df):,} total rows")
        df = df.sample(n=sample_size, random_state=42)
    else:
        print(f"\nAnalyzing all {len(df):,} rows")

    # Filter to rows with edit_smiles
    df_with_edits = df[df['edit_smiles'].notna()].copy()
    print(f"Rows with edit_smiles: {len(df_with_edits):,}")

    if len(df_with_edits) == 0:
        print("\n⚠️  No rows with edit_smiles found!")
        return {}

    # Analyze each row
    results = []
    ambiguous_examples = []

    print("\nAnalyzing edits...")
    for idx, row in tqdm(df_with_edits.iterrows(), total=len(df_with_edits)):
        mol_a = row.get('mol_a')
        edit_smiles = row.get('edit_smiles')

        if pd.isna(mol_a) or pd.isna(edit_smiles):
            continue

        is_ambig, num_matches, reactant = is_edit_ambiguous(
            mol_a, edit_smiles, verbose=False
        )

        results.append({
            'is_ambiguous': is_ambig,
            'num_matches': num_matches,
            'reactant': reactant
        })

        # Store examples
        if is_ambig and len(ambiguous_examples) < verbose_examples:
            ambiguous_examples.append({
                'mol_a': mol_a,
                'edit_smiles': edit_smiles,
                'num_matches': num_matches,
                'reactant': reactant
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate statistics
    total = len(results_df)
    ambiguous = results_df['is_ambiguous'].sum()
    unambiguous = total - ambiguous
    ambiguous_pct = (ambiguous / total * 100) if total > 0 else 0

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nTotal edits analyzed:     {total:>8,}")
    print(f"Unambiguous edits:        {unambiguous:>8,} ({100-ambiguous_pct:>5.2f}%)")
    print(f"Ambiguous edits:          {ambiguous:>8,} ({ambiguous_pct:>5.2f}%)")

    # Distribution of matches
    if len(results_df) > 0:
        print(f"\n{'='*80}")
        print("DISTRIBUTION OF MATCHES")
        print(f"{'='*80}")
        match_counts = results_df[results_df['num_matches'] > 0]['num_matches'].value_counts().sort_index()

        print(f"\n{'Matches':<15} {'Count':>10} {'Percentage':>12}")
        print("-"*80)
        for matches, count in match_counts.items():
            pct = count / total * 100
            ambig_marker = " ⚠️" if matches > 1 else ""
            print(f"{matches:<15} {count:>10,} {pct:>11.2f}%{ambig_marker}")

    # Show examples
    if ambiguous_examples:
        print(f"\n{'='*80}")
        print(f"EXAMPLE AMBIGUOUS EDITS (showing {len(ambiguous_examples)})")
        print(f"{'='*80}")

        for i, example in enumerate(ambiguous_examples, 1):
            print(f"\n{i}. Molecule A: {example['mol_a'][:80]}")
            if len(example['mol_a']) > 80:
                print(f"   {' '*13}{example['mol_a'][80:]}")
            print(f"   Edit:       {example['edit_smiles']}")
            print(f"   Reactant:   {example['reactant']}")
            print(f"   Matches:    {example['num_matches']} occurrences ⚠️")

    # Additional statistics
    if len(results_df[results_df['is_ambiguous']]) > 0:
        print(f"\n{'='*80}")
        print("AMBIGUOUS EDIT STATISTICS")
        print(f"{'='*80}")

        ambig_df = results_df[results_df['is_ambiguous']]
        print(f"\nAverage matches (ambiguous only): {ambig_df['num_matches'].mean():.2f}")
        print(f"Max matches:                       {ambig_df['num_matches'].max()}")
        print(f"Min matches (ambiguous):           {ambig_df['num_matches'].min()}")

        # Most common reactant patterns in ambiguous edits
        print(f"\nTop 10 reactant patterns in ambiguous edits:")
        top_reactants = ambig_df['reactant'].value_counts().head(10)
        for j, (reactant, count) in enumerate(top_reactants.items(), 1):
            pct = count / len(ambig_df) * 100
            print(f"  {j:2d}. {reactant:<30} {count:>5,} ({pct:>5.1f}%)")

    print("\n" + "="*80)

    return {
        'total': total,
        'ambiguous': ambiguous,
        'unambiguous': unambiguous,
        'ambiguous_pct': ambiguous_pct,
        'results_df': results_df,
        'ambiguous_examples': ambiguous_examples
    }


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description='Analyze ambiguous edit SMILES')
    parser.add_argument(
        '--data-file',
        type=str,
        default='data/pairs/chembl_pairs_long_sample.csv',
        help='Path to data file'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of rows to sample (default: all)'
    )
    parser.add_argument(
        '--verbose-examples',
        type=int,
        default=10,
        help='Number of ambiguous examples to print'
    )
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Path to save detailed results CSV'
    )

    args = parser.parse_args()

    # Load data
    print(f"\nLoading data from: {args.data_file}")
    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df):,} rows")

    # Analyze
    results = analyze_dataset(
        df,
        sample_size=args.sample_size,
        verbose_examples=args.verbose_examples
    )

    # Save detailed results if requested
    if args.save_results and 'results_df' in results:
        # Add original data
        results_df = results['results_df']
        df_with_results = df.iloc[results_df.index].copy()
        df_with_results['is_ambiguous'] = results_df['is_ambiguous'].values
        df_with_results['num_matches'] = results_df['num_matches'].values

        # Save
        df_with_results.to_csv(args.save_results, index=False)
        print(f"\n✓ Saved detailed results to: {args.save_results}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if results:
        print(f"""
Key Findings:
- {results['ambiguous_pct']:.1f}% of edits are ambiguous
- {results['unambiguous']:,} edits can be applied unambiguously
- {results['ambiguous']:,} edits have multiple possible locations

Implications:
- Ambiguous edits may lead to:
  1. Undefined behavior (which match is used?)
  2. Potential data quality issues
  3. Model confusion during training

Recommendations:
- If ambiguous_pct < 5%: Probably acceptable, can filter out
- If ambiguous_pct > 20%: Consider using position-specific edits
- If ambiguous_pct > 50%: Dataset may need redesign
        """)

    print("="*80)


if __name__ == '__main__':
    main()

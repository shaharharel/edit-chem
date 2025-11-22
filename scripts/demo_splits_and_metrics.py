"""
Demonstration of challenging molecular splits and comprehensive metrics.

This script shows how to use the new evaluation framework with:
1. Multiple challenging splitting strategies
2. Comprehensive metrics evaluation
3. Comparison across different splits

Run: python scripts/demo_splits_and_metrics.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils import (
    get_splitter,
    RegressionMetrics,
    MultiTaskMetrics,
    print_metrics_summary
)


def demo_splits():
    """Demonstrate all splitting strategies."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION: CHALLENGING MOLECULAR SPLITS")
    print("=" * 80)

    # Create synthetic dataset
    print("\nCreating synthetic molecular dataset...")
    np.random.seed(42)

    # Generate fake SMILES (for demo purposes)
    smiles_list = [
        "CCO",  # ethanol
        "CC(C)O",  # isopropanol
        "CCCO",  # propanol
        "c1ccccc1",  # benzene
        "c1ccccc1O",  # phenol
        "c1ccccc1C",  # toluene
        "CC(=O)O",  # acetic acid
        "CCC(=O)O",  # propanoic acid
        "c1ccccc1N",  # aniline
        "c1ccccc1Cl",  # chlorobenzene
    ] * 100  # Repeat to get 1000 molecules

    df = pd.DataFrame({
        'smiles': smiles_list[:1000],
        'property_value': np.random.randn(1000) * 2 + 5,
        'target_id': np.random.choice(['TARGET_A', 'TARGET_B', 'TARGET_C'], 1000),
        'timestamp': pd.date_range('2020-01-01', periods=1000, freq='D')
    })

    print(f"  Created dataset: {len(df)} molecules")
    print(f"  Unique SMILES: {df['smiles'].nunique()}")
    print(f"  Properties: {df['property_value'].describe()}")

    # Test each splitter
    split_configs = [
        ('random', {}, {}),
        ('scaffold', {'use_generic': True}, {}),
        ('scaffold', {'use_generic': False}, {}),
        ('target', {}, {'target_col': 'target_id'}),
        ('stratified', {'n_bins': 5}, {'property_col': 'property_value'}),
        ('temporal', {}, {'time_col': 'timestamp'}),
    ]

    results = []

    for split_type, init_kwargs, split_kwargs in split_configs:
        print("\n" + "-" * 80)
        print(f"Testing: {split_type.upper()} split")
        if init_kwargs or split_kwargs:
            print(f"  Init config: {init_kwargs}")
            print(f"  Split config: {split_kwargs}")
        print("-" * 80)

        try:
            splitter = get_splitter(split_type, **init_kwargs)
            train, val, test = splitter.split(df, **split_kwargs)

            # Analyze splits
            result = {
                'split_type': split_type,
                'config': str({**init_kwargs, **split_kwargs}),
                'train_size': len(train),
                'val_size': len(val),
                'test_size': len(test),
                'train_unique_smiles': train['smiles'].nunique(),
                'test_unique_smiles': test['smiles'].nunique(),
                'property_overlap': (
                    (train['property_value'].min() <= test['property_value'].max()) and
                    (train['property_value'].max() >= test['property_value'].min())
                )
            }

            # Check scaffold overlap if applicable
            if split_type == 'scaffold':
                from rdkit import Chem
                from rdkit.Chem.Scaffolds import MurckoScaffold

                def get_scaffold(smi):
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        if init_kwargs.get('use_generic'):
                            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
                        return Chem.MolToSmiles(scaffold)
                    except:
                        return None

                train_scaffolds = set(train['smiles'].apply(get_scaffold))
                test_scaffolds = set(test['smiles'].apply(get_scaffold))
                scaffold_overlap = len(train_scaffolds & test_scaffolds)

                result['train_scaffolds'] = len(train_scaffolds)
                result['test_scaffolds'] = len(test_scaffolds)
                result['scaffold_overlap'] = scaffold_overlap
                result['scaffold_overlap_pct'] = (
                    scaffold_overlap / len(test_scaffolds) * 100
                    if len(test_scaffolds) > 0 else 0
                )

                print(f"  Train scaffolds: {len(train_scaffolds)}")
                print(f"  Test scaffolds: {len(test_scaffolds)}")
                print(f"  Overlap: {scaffold_overlap} ({result['scaffold_overlap_pct']:.1f}%)")

            results.append(result)

            print(f"\n  ✓ Split successful")
            print(f"    Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
            print(f"    Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
            print(f"    Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    # Summary table
    print("\n" + "=" * 80)
    print("SPLIT COMPARISON SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    return results_df


def demo_metrics():
    """Demonstrate comprehensive metrics."""
    print("\n\n" + "=" * 80)
    print("DEMONSTRATION: COMPREHENSIVE METRICS")
    print("=" * 80)

    # Create synthetic predictions
    np.random.seed(42)
    n_samples = 500
    n_tasks = 3

    # True values
    y_true = np.random.randn(n_samples) * 2 + 5

    # Predictions (with some error)
    y_pred = y_true + np.random.randn(n_samples) * 0.5

    print("\n1. Single-Task Regression Metrics")
    print("-" * 80)

    metrics = RegressionMetrics.compute_all(y_true, y_pred)
    print_metrics_summary(metrics, "Single-Task Metrics")

    # Per-bin analysis
    print("\n2. Performance by Property Value Range")
    print("-" * 80)

    bin_results = RegressionMetrics.compute_per_bin(
        y_true, y_pred, n_bins=5, metric='mae'
    )

    print("\nMAE by property value bins:")
    for i, (center, mae, count) in enumerate(zip(
        bin_results['bin_centers'],
        bin_results['metric_values'],
        bin_results['bin_counts']
    )):
        print(f"  Bin {i+1} (center={center:.2f}): MAE={mae:.4f} (n={count})")

    # Multi-task metrics
    print("\n\n3. Multi-Task Metrics")
    print("-" * 80)

    # Create multi-task data (sparse labels)
    y_true_mt = np.full((n_samples, n_tasks), np.nan)
    y_pred_mt = np.full((n_samples, n_tasks), np.nan)

    for i in range(n_tasks):
        # Each task has measurements for different samples
        mask = np.random.rand(n_samples) > 0.5
        y_true_mt[mask, i] = np.random.randn(mask.sum()) * 2 + 5
        y_pred_mt[mask, i] = y_true_mt[mask, i] + np.random.randn(mask.sum()) * 0.5

    task_names = ['LogP', 'Solubility', 'Clearance']

    mt_metrics = MultiTaskMetrics.compute_all_tasks(
        y_true_mt, y_pred_mt, task_names
    )

    print("\nPer-task metrics:")
    print(mt_metrics[['task', 'n_samples', 'mae', 'rmse', 'r2']].to_string(index=False))

    macro_metrics = MultiTaskMetrics.compute_macro_metrics(mt_metrics)
    print("\nMacro-averaged metrics:")
    for key, val in macro_metrics.items():
        print(f"  {key}: {val:.4f}")

    # Ranking metrics
    print("\n\n4. Ranking Metrics (Drug Discovery)")
    print("-" * 80)

    from src.utils.metrics import RankingMetrics

    # Top-k accuracy
    top_100_acc = RankingMetrics.top_k_accuracy(y_true, y_pred, k=100)
    print(f"\nTop-100 accuracy: {top_100_acc:.4f}")
    print(f"  (Fraction of true top-100 compounds in predicted top-100)")

    # Enrichment factor
    threshold = np.percentile(y_true, 90)  # Top 10% are "active"
    ef = RankingMetrics.enrichment_factor(y_true, y_pred, threshold, top_percent=0.05)
    print(f"\nEnrichment factor (top 5%): {ef:.2f}x")
    print(f"  (>1 means better than random screening)")

    # NDCG
    ndcg = RankingMetrics.ndcg_score(y_true, y_pred, k=100)
    print(f"\nNDCG@100: {ndcg:.4f}")
    print(f"  (Ranking quality, higher is better)")

    print("\n" + "=" * 80)
    print("✓ DEMONSTRATION COMPLETE")
    print("=" * 80)


def demo_split_comparison():
    """Compare model performance across different splits."""
    print("\n\n" + "=" * 80)
    print("DEMONSTRATION: CROSS-SPLIT EVALUATION")
    print("=" * 80)

    print("\nSimulating model evaluation across different splits...")

    # Create dataset
    np.random.seed(42)
    smiles_list = [
        "CCO", "CC(C)O", "CCCO", "c1ccccc1", "c1ccccc1O",
        "c1ccccc1C", "CC(=O)O", "CCC(=O)O", "c1ccccc1N", "c1ccccc1Cl"
    ] * 50

    df = pd.DataFrame({
        'smiles': smiles_list,
        'property_value': np.random.randn(500) * 2 + 5,
        'target_id': np.random.choice(['A', 'B', 'C'], 500)
    })

    # Test different splits
    splits = ['random', 'scaffold', 'target']
    results = []

    for split_type in splits:
        print(f"\n{split_type.upper()} Split")
        print("-" * 80)

        try:
            splitter = get_splitter(split_type)

            if split_type == 'target':
                train, val, test = splitter.split(df, target_col='target_id')
            else:
                train, val, test = splitter.split(df)

            # Simulate predictions (simple baseline: predict mean)
            train_mean = train['property_value'].mean()
            y_true = test['property_value'].values
            y_pred = np.full_like(y_true, train_mean)

            # Add some realistic variance
            y_pred = y_pred + np.random.randn(len(y_pred)) * 0.5

            # Compute metrics
            metrics = RegressionMetrics.compute_all(y_true, y_pred)

            result = {
                'split': split_type,
                'test_size': len(test),
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'pearson_r': metrics['pearson_r']
            }
            results.append(result)

            print(f"  Test size: {len(test)}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  R²: {metrics['r2']:.4f}")

        except Exception as e:
            print(f"  Failed: {e}")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON ACROSS SPLITS")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    print("\n📊 Interpretation:")
    print("  - Random split: Usually best performance (train/test most similar)")
    print("  - Scaffold split: Most challenging (novel scaffolds in test)")
    print("  - Target split: Tests generalization to new biological targets")


if __name__ == '__main__':
    print("\n" + "🧪" * 40)
    print("EDIT-CHEM: EVALUATION FRAMEWORK DEMONSTRATION")
    print("🧪" * 40)

    # Run demonstrations
    demo_splits()
    demo_metrics()
    demo_split_comparison()

    print("\n" + "=" * 80)
    print("ALL DEMONSTRATIONS COMPLETE!")
    print("=" * 80)
    print("\n📚 Next steps:")
    print("  1. Use these splits in your training pipeline")
    print("  2. Report metrics across all splits for robust evaluation")
    print("  3. Focus on scaffold split for challenging generalization test")
    print("\n")

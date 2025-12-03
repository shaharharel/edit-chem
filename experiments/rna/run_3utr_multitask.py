#!/usr/bin/env python3
"""
Multi-task experiment for 3'UTR MPRA data across 6 cell lines.

Compares:
1. Baseline Property Predictor (f(seq) → property)
2. Edit Framework (emb_B - emb_A → delta)
3. Structured Edit Embedding (full edit representation → delta)

All using frozen/non-trainable RNA-FM embeddings for speed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm
import json
import time
from collections import defaultdict


def load_multitask_data(data_path: str, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """
    Load 3'UTR multi-task data and split by variant_id (not by row).

    This ensures the same variant is in train/val/test for all cell lines.
    """
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")

    # Get unique variants
    unique_variants = df['variant_id'].unique()
    n_variants = len(unique_variants)
    print(f"Unique variants: {n_variants}")

    # Split variants
    np.random.seed(seed)
    indices = np.random.permutation(n_variants)

    test_end = int(n_variants * test_size)
    val_end = test_end + int(n_variants * val_size)

    test_variants = set(unique_variants[indices[:test_end]])
    val_variants = set(unique_variants[indices[test_end:val_end]])
    train_variants = set(unique_variants[indices[val_end:]])

    # Split dataframe
    train_df = df[df['variant_id'].isin(train_variants)].copy()
    val_df = df[df['variant_id'].isin(val_variants)].copy()
    test_df = df[df['variant_id'].isin(test_variants)].copy()

    print(f"Train variants: {len(train_variants)}, rows: {len(train_df)}")
    print(f"Val variants: {len(val_variants)}, rows: {len(val_df)}")
    print(f"Test variants: {len(test_variants)}, rows: {len(test_df)}")

    # Organize by property (cell line)
    datasets = {}
    for prop in df['property_name'].unique():
        datasets[prop] = {
            'train': train_df[train_df['property_name'] == prop].copy(),
            'val': val_df[val_df['property_name'] == prop].copy(),
            'test': test_df[test_df['property_name'] == prop].copy()
        }

    return datasets, df


def precompute_embeddings(df, embedder, batch_size=32):
    """Pre-compute embeddings for all unique sequences."""
    all_seqs = list(set(df['seq_a'].tolist() + df['seq_b'].tolist()))
    print(f"Pre-computing embeddings for {len(all_seqs)} unique sequences...")

    embeddings = {}

    for i in tqdm(range(0, len(all_seqs), batch_size), desc="Embedding"):
        batch_seqs = all_seqs[i:i+batch_size]
        # Use encode() for RNAFMEmbedder, embed() for others
        if hasattr(embedder, 'encode'):
            batch_emb = embedder.encode(batch_seqs)
        else:
            batch_emb = embedder.embed(batch_seqs)

        # Convert to numpy if tensor
        if hasattr(batch_emb, 'cpu'):
            batch_emb = batch_emb.cpu().numpy()

        for seq, emb in zip(batch_seqs, batch_emb):
            embeddings[seq] = emb

    return embeddings


def train_baseline_property_predictor(datasets, seq_to_emb, config):
    """
    Train baseline property predictor that predicts property value from sequence.
    At test time, computes delta = f(seq_b) - f(seq_a).
    """
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor

    print("\n=== Training Baseline Property Predictor ===")

    # Collect all unique sequences and their values
    seq_values = defaultdict(list)

    for prop_name, splits in datasets.items():
        for _, row in splits['train'].iterrows():
            seq_values[row['seq_a']].append(row['value_a'])
            seq_values[row['seq_b']].append(row['value_b'])

    # Average values for sequences appearing multiple times
    train_seqs = list(seq_values.keys())
    train_X = np.array([seq_to_emb[s] for s in train_seqs])
    train_y = np.array([np.mean(seq_values[s]) for s in train_seqs])

    print(f"Training on {len(train_y)} unique sequences")

    # Train Ridge regression (fast, non-trainable approach)
    model = Ridge(alpha=1.0)
    model.fit(train_X, train_y)

    return model


def train_edit_framework(datasets, seq_to_emb, config):
    """
    Train edit framework: predict delta from edit embedding (emb_B - emb_A).
    """
    from sklearn.linear_model import Ridge

    print("\n=== Training Edit Framework ===")

    # Collect training data
    train_X = []
    train_y = []

    for prop_name, splits in datasets.items():
        for _, row in splits['train'].iterrows():
            emb_a = seq_to_emb[row['seq_a']]
            emb_b = seq_to_emb[row['seq_b']]
            edit_emb = emb_b - emb_a  # Edit embedding as difference
            train_X.append(np.concatenate([emb_a, edit_emb]))  # [seq_a_emb, edit_emb]
            train_y.append(row['delta'])

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    print(f"Training on {len(train_y)} pairs")
    print(f"Feature dim: {train_X.shape[1]}")

    # Train Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(train_X, train_y)

    return model


def evaluate_baseline(model, datasets, seq_to_emb):
    """Evaluate baseline property predictor."""
    results = {}

    for prop_name, splits in datasets.items():
        test_df = splits['test']

        all_true = []
        all_pred = []

        for _, row in test_df.iterrows():
            true_delta = row['delta']

            emb_a = seq_to_emb[row['seq_a']].reshape(1, -1)
            emb_b = seq_to_emb[row['seq_b']].reshape(1, -1)

            pred_a = model.predict(emb_a)[0]
            pred_b = model.predict(emb_b)[0]
            pred_delta = pred_b - pred_a

            all_true.append(true_delta)
            all_pred.append(pred_delta)

        results[prop_name] = compute_metrics(all_true, all_pred)

    return results


def evaluate_edit_framework(model, datasets, seq_to_emb):
    """Evaluate edit framework."""
    results = {}

    for prop_name, splits in datasets.items():
        test_df = splits['test']

        all_true = []
        all_pred = []

        for _, row in test_df.iterrows():
            true_delta = row['delta']

            emb_a = seq_to_emb[row['seq_a']]
            emb_b = seq_to_emb[row['seq_b']]
            edit_emb = emb_b - emb_a

            X = np.concatenate([emb_a, edit_emb]).reshape(1, -1)
            pred_delta = model.predict(X)[0]

            all_true.append(true_delta)
            all_pred.append(pred_delta)

        results[prop_name] = compute_metrics(all_true, all_pred)

    return results


def train_structured_edit(datasets, config):
    """
    Train structured edit embedding model.
    Uses mutation type + position + context features.
    """
    from sklearn.linear_model import Ridge

    print("\n=== Training Structured Edit Model (Simple) ===")

    # Create mutation type one-hot encoding
    mutation_types = ['A→C', 'A→G', 'A→U', 'C→A', 'C→G', 'C→U',
                      'G→A', 'G→C', 'G→U', 'U→A', 'U→C', 'U→G']
    mut_to_idx = {m: i for i, m in enumerate(mutation_types)}

    # Collect training data with structured features
    train_X = []
    train_y = []

    for prop_name, splits in datasets.items():
        for _, row in splits['train'].iterrows():
            # Mutation type one-hot
            mut_type = f"{row['edit_from']}→{row['edit_to']}"
            mut_vec = np.zeros(len(mutation_types))
            if mut_type in mut_to_idx:
                mut_vec[mut_to_idx[mut_type]] = 1

            # Position features (normalized)
            seq_len = len(row['seq_a'])
            pos = row['edit_position']
            pos_features = [
                pos / seq_len,  # Relative position
                pos / 50,  # Absolute position normalized
                1 if pos < 10 else 0,  # Near start
                1 if pos > seq_len - 10 else 0,  # Near end
            ]

            # Local context (nucleotide one-hot for ±2bp)
            context_features = []
            seq = row['seq_a']
            for offset in [-2, -1, 1, 2]:
                ctx_pos = pos + offset
                if 0 <= ctx_pos < len(seq):
                    nuc = seq[ctx_pos]
                    nuc_vec = [1 if nuc == n else 0 for n in ['A', 'C', 'G', 'U']]
                else:
                    nuc_vec = [0, 0, 0, 0]
                context_features.extend(nuc_vec)

            # Combine features
            features = np.concatenate([
                mut_vec,
                np.array(pos_features),
                np.array(context_features)
            ])

            train_X.append(features)
            train_y.append(row['delta'])

    train_X = np.array(train_X)
    train_y = np.array(train_y)

    print(f"Training on {len(train_y)} pairs")
    print(f"Feature dim: {train_X.shape[1]} (mut_type={len(mutation_types)}, pos=4, context=16)")

    # Train Ridge regression
    model = Ridge(alpha=1.0)
    model.fit(train_X, train_y)

    # Store mutation type mapping
    model.mutation_types = mutation_types
    model.mut_to_idx = mut_to_idx

    return model


def evaluate_structured_edit(model, datasets):
    """Evaluate structured edit model."""
    mutation_types = model.mutation_types
    mut_to_idx = model.mut_to_idx

    results = {}

    for prop_name, splits in datasets.items():
        test_df = splits['test']

        all_true = []
        all_pred = []

        for _, row in test_df.iterrows():
            true_delta = row['delta']

            # Build features (same as training)
            mut_type = f"{row['edit_from']}→{row['edit_to']}"
            mut_vec = np.zeros(len(mutation_types))
            if mut_type in mut_to_idx:
                mut_vec[mut_to_idx[mut_type]] = 1

            seq_len = len(row['seq_a'])
            pos = row['edit_position']
            pos_features = [
                pos / seq_len,
                pos / 50,
                1 if pos < 10 else 0,
                1 if pos > seq_len - 10 else 0,
            ]

            context_features = []
            seq = row['seq_a']
            for offset in [-2, -1, 1, 2]:
                ctx_pos = pos + offset
                if 0 <= ctx_pos < len(seq):
                    nuc = seq[ctx_pos]
                    nuc_vec = [1 if nuc == n else 0 for n in ['A', 'C', 'G', 'U']]
                else:
                    nuc_vec = [0, 0, 0, 0]
                context_features.extend(nuc_vec)

            features = np.concatenate([
                mut_vec,
                np.array(pos_features),
                np.array(context_features)
            ]).reshape(1, -1)

            pred_delta = model.predict(features)[0]

            all_true.append(true_delta)
            all_pred.append(pred_delta)

        results[prop_name] = compute_metrics(all_true, all_pred)

    return results


def compute_metrics(all_true, all_pred):
    """Compute evaluation metrics."""
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    mae = np.mean(np.abs(all_true - all_pred))
    rmse = np.sqrt(np.mean((all_true - all_pred) ** 2))

    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    if len(all_true) > 2 and np.std(all_true) > 0 and np.std(all_pred) > 0:
        pearson_r = stats.pearsonr(all_true, all_pred)[0]
        spearman_r = stats.spearmanr(all_true, all_pred)[0]
    else:
        pearson_r = 0
        spearman_r = 0

    # Direction accuracy
    direction_correct = np.sum(np.sign(all_true) == np.sign(all_pred))
    direction_accuracy = direction_correct / len(all_true)

    return {
        'n_samples': len(all_true),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'direction_accuracy': float(direction_accuracy),
        'true_mean': float(np.mean(all_true)),
        'true_std': float(np.std(all_true)),
        'pred_mean': float(np.mean(all_pred)),
        'pred_std': float(np.std(all_pred))
    }


def print_results_table(all_results, cell_lines):
    """Print results in a nice table format."""

    print("\n" + "="*100)
    print("RESULTS BY METHOD")
    print("="*100)

    for method_name, results in all_results.items():
        print(f"\n### {method_name} ###")
        print("-"*90)
        print(f"{'Cell Line':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Pearson':>10} {'Dir Acc':>10} {'N':>6}")
        print("-"*90)

        all_mae = []
        all_dir_acc = []

        for cell_line in cell_lines:
            prop_name = f"3UTR_skew_{cell_line}"
            if prop_name in results:
                m = results[prop_name]
                print(f"{cell_line:<20} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f} {m['pearson_r']:>10.4f} {m['direction_accuracy']:>10.2%} {m['n_samples']:>6}")
                all_mae.append(m['mae'])
                all_dir_acc.append(m['direction_accuracy'])

        print("-"*90)
        print(f"{'AVERAGE':<20} {np.mean(all_mae):>8.4f} {'':>8} {'':>8} {'':>10} {np.mean(all_dir_acc):>10.2%}")


def main():
    """Run multi-task experiment on 3'UTR data."""

    print("="*80)
    print("3'UTR MULTI-TASK EXPERIMENT")
    print("="*80)

    # Config
    data_path = "data/rna/pairs/mprau_3utr_multitask_with_seq.csv"
    output_dir = Path("experiments/rna/results/3utr_multitask")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n=== Loading Data ===")
    datasets, df = load_multitask_data(data_path)

    cell_lines = ['HEK293FT', 'HEPG2', 'HMEC', 'K562', 'GM12878', 'SKNSH']

    # Print data statistics
    print("\n=== Data Statistics ===")
    print(f"Sequence length: {df['seq_a'].str.len().iloc[0]} bp")
    print(f"Total rows: {len(df)}")
    print(f"Cell lines: {cell_lines}")

    # Analyze correlations between cell lines
    print("\n=== Cell Line Correlations ===")
    pivot = df.pivot_table(index='variant_id', columns='cell_type', values='delta')
    corr = pivot.corr()
    print(corr.round(3))

    # Analyze mutation type statistics
    print("\n=== Mutation Type Distribution ===")
    mut_stats = df[df['cell_type'] == 'HEK293FT'].groupby(['edit_from', 'edit_to'])['delta'].agg(['mean', 'std', 'count'])
    print(mut_stats.sort_values('mean', ascending=False).head(12))

    # Create embedder (using simple one-hot for speed, or RNA-FM)
    print("\n=== Creating Embedder ===")

    try:
        from src.embedding.rna.rnafm import RNAFMEmbedder
        embedder = RNAFMEmbedder(model_name='rna_fm_t12', pooling='mean', device='mps')
        print("Using RNA-FM embedder")
    except Exception as e:
        print(f"RNA-FM not available ({e}), using simple nucleotide embedding")

        class SimpleNucleotideEmbedder:
            """Simple one-hot + k-mer embedding."""
            def __init__(self, kmer_sizes=[3, 4]):
                self.kmer_sizes = kmer_sizes
                self.embedding_dim = None

            def embed(self, sequences):
                embeddings = []
                for seq in sequences:
                    # One-hot for each position (not practical for variable length)
                    # Instead, use k-mer frequencies
                    features = []
                    for k in self.kmer_sizes:
                        kmer_counts = defaultdict(int)
                        for i in range(len(seq) - k + 1):
                            kmer_counts[seq[i:i+k]] += 1
                        # Normalize
                        total = sum(kmer_counts.values())
                        for kmer in sorted(kmer_counts.keys()):
                            features.append(kmer_counts[kmer] / total)
                    embeddings.append(np.array(features))

                # Pad to same length
                max_len = max(len(e) for e in embeddings)
                embeddings = [np.pad(e, (0, max_len - len(e))) for e in embeddings]

                self.embedding_dim = max_len
                return np.array(embeddings)

        embedder = SimpleNucleotideEmbedder()

    # Pre-compute embeddings
    print("\n=== Pre-computing Embeddings ===")
    start_time = time.time()
    seq_to_emb = precompute_embeddings(df, embedder)
    print(f"Embedding time: {time.time() - start_time:.1f}s")
    print(f"Embedding dim: {next(iter(seq_to_emb.values())).shape}")

    # Train and evaluate models
    all_results = {}

    # 1. Baseline Property Predictor
    print("\n" + "="*60)
    print("METHOD 1: Baseline Property Predictor")
    print("="*60)
    baseline_model = train_baseline_property_predictor(datasets, seq_to_emb, {})
    baseline_results = evaluate_baseline(baseline_model, datasets, seq_to_emb)
    all_results['Baseline Property Predictor'] = baseline_results

    # 2. Edit Framework
    print("\n" + "="*60)
    print("METHOD 2: Edit Framework (emb_B - emb_A)")
    print("="*60)
    edit_model = train_edit_framework(datasets, seq_to_emb, {})
    edit_results = evaluate_edit_framework(edit_model, datasets, seq_to_emb)
    all_results['Edit Framework'] = edit_results

    # 3. Structured Edit (simple features)
    print("\n" + "="*60)
    print("METHOD 3: Structured Edit (mutation type + position + context)")
    print("="*60)
    structured_model = train_structured_edit(datasets, {})
    structured_results = evaluate_structured_edit(structured_model, datasets)
    all_results['Structured Edit'] = structured_results

    # Print results
    print_results_table(all_results, cell_lines)

    # Compute overall statistics
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)

    summary = []
    for method_name, results in all_results.items():
        all_mae = [results[f"3UTR_skew_{cl}"]['mae'] for cl in cell_lines]
        all_dir = [results[f"3UTR_skew_{cl}"]['direction_accuracy'] for cl in cell_lines]
        all_r2 = [results[f"3UTR_skew_{cl}"]['r2'] for cl in cell_lines]

        summary.append({
            'method': method_name,
            'avg_mae': np.mean(all_mae),
            'avg_dir_acc': np.mean(all_dir),
            'avg_r2': np.mean(all_r2),
            'std_mae': np.std(all_mae),
            'std_dir_acc': np.std(all_dir)
        })

    print(f"\n{'Method':<35} {'Avg MAE':>10} {'Avg Dir Acc':>12} {'Avg R²':>10}")
    print("-"*70)
    for s in sorted(summary, key=lambda x: -x['avg_dir_acc']):
        print(f"{s['method']:<35} {s['avg_mae']:>10.4f} {s['avg_dir_acc']:>12.2%} {s['avg_r2']:>10.4f}")

    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'summary.csv', index=False)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED!")
    print(f"{'='*80}")

    return all_results


if __name__ == '__main__':
    main()

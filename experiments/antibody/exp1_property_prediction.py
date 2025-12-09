#!/usr/bin/env python
"""
Experiment 1: Antibody Property Prediction

Test embedding quality by predicting absolute property values from single antibody sequences.
This is NOT the edit framework - just a simple MLP on top of embeddings.

Task: Given an antibody (H+L sequences) → predict its property value

For each dataset:
1. Flatten pairs to unique antibodies with their property values
2. Random split with no antibody leakage between train/val/test
3. Embed antibodies with each embedder
4. Train MLP predictor
5. Report regression metrics per dataset and per property

Usage:
    python experiments/antibody/exp1_property_prediction.py --dataset abbibench --embedder igbert
    python experiments/antibody/exp1_property_prediction.py --all
"""

import argparse
import json
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_unified_data(source_dataset: str) -> pd.DataFrame:
    """Load unified antibody data for a specific dataset."""
    with open('data/antibody/unified/unified_antibody_data.json') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df = df[df['source_dataset'] == source_dataset]

    return df


def load_abbibench_raw() -> pd.DataFrame:
    """
    Load raw AbBiBench data directly (185k samples).

    AbBiBench contains individual antibody variants with binding scores,
    not pairs. Each row is: heavy_chain_seq, light_chain_seq, binding_score.
    """
    data_file = Path('data/antibody/abbibench/train.csv')
    if not data_file.exists():
        raise FileNotFoundError(f"AbBiBench data not found at {data_file}")

    df = pd.read_csv(data_file)

    # Rename columns to match expected format
    df = df.rename(columns={
        'heavy_chain_seq': 'heavy_seq',
        'light_chain_seq': 'light_seq',
        'binding_score': 'property_value'
    })

    # Add metadata columns
    df['source_dataset'] = 'abbibench_raw'
    df['antibody_id'] = [f'abbibench_{i}' for i in range(len(df))]
    df['antigen_id'] = None

    return df


def load_magma_seq_flat() -> pd.DataFrame:
    """
    Load MAGMA-seq data as flat dataframe for property prediction.

    Uses the existing loader and extracts mutant sequences with ddG values.
    Returns ~2,892 samples.
    """
    from src.data.antibody.loaders import load_magma_seq

    data = load_magma_seq('data/antibody/magma_seq/')

    rows = []
    for pair in data.pairs:
        # Apply mutations to get mutant sequence
        heavy_mut = list(pair.heavy_wt)
        light_mut = list(pair.light_wt) if pair.light_wt else []

        for mut in pair.mutations:
            if mut.chain == 'H' and mut.position < len(heavy_mut):
                heavy_mut[mut.position] = mut.to_aa
            elif mut.chain == 'L' and mut.position < len(light_mut):
                light_mut[mut.position] = mut.to_aa

        rows.append({
            'heavy_seq': ''.join(heavy_mut),
            'light_seq': ''.join(light_mut) if light_mut else '',
            'property_value': pair.delta_value,  # ddG
            'source_dataset': 'magma_seq',
            'antibody_id': pair.antibody_id,
            'antigen_id': pair.antigen_id,
        })

    return pd.DataFrame(rows)


def flatten_to_unique_antibodies(df: pd.DataFrame, use_mutant: bool = True) -> pd.DataFrame:
    """
    Flatten mutation pairs to unique antibodies with property values.

    For property prediction, we want one row per unique antibody sequence.
    We can use either:
    - Mutant sequences with raw_mut_value (if available) or delta_value
    - WT sequences with raw_wt_value (if available)

    Args:
        df: DataFrame with mutation data
        use_mutant: If True, use mutant sequences; if False, use WT sequences

    Returns:
        DataFrame with unique antibodies and their property values
    """
    if use_mutant:
        # Use mutant sequences
        seq_cols = ['heavy_mut', 'light_mut']
        value_col = 'raw_mut_value' if 'raw_mut_value' in df.columns and df['raw_mut_value'].notna().any() else 'delta_value'
    else:
        # Use WT sequences
        seq_cols = ['heavy_wt', 'light_wt']
        value_col = 'raw_wt_value' if 'raw_wt_value' in df.columns and df['raw_wt_value'].notna().any() else None
        if value_col is None:
            raise ValueError("No raw_wt_value available for WT sequences")

    # Filter for valid sequences
    df = df.copy()
    df['heavy_seq'] = df[seq_cols[0]].fillna('')
    df['light_seq'] = df[seq_cols[1]].fillna('')

    # Must have heavy chain
    df = df[df['heavy_seq'].str.len() > 0]

    # Get property value
    df['property_value'] = df[value_col]

    # Remove rows with NaN property values
    df = df[df['property_value'].notna()]

    # Group by unique antibody sequence and aggregate
    # For duplicates, take mean property value
    grouped = df.groupby(['heavy_seq', 'light_seq']).agg({
        'property_value': 'mean',
        'antibody_id': 'first',
        'antigen_id': 'first',
        'source_dataset': 'first',
    }).reset_index()

    return grouped


def create_embedder(embedder_type: str, device: str, trainable: bool = False):
    """Create antibody embedder."""
    if embedder_type == 'igbert':
        from src.embedding.antibody import IgBertEmbedder
        return IgBertEmbedder(device=device, trainable=trainable)
    elif embedder_type == 'igt5':
        from src.embedding.antibody import IgT5Embedder
        return IgT5Embedder(device=device, trainable=trainable)
    elif embedder_type == 'ablang2':
        from src.embedding.antibody import AbLang2Embedder
        return AbLang2Embedder(device=device, trainable=trainable)
    elif embedder_type == 'antiberta2':
        from src.embedding.antibody import AntiBERTa2Embedder
        return AntiBERTa2Embedder(device=device, trainable=trainable)
    elif embedder_type == 'balm':
        from src.embedding.antibody import BALMEmbedder
        return BALMEmbedder(device=device, trainable=trainable)
    elif embedder_type == 'balm_paired':
        from src.embedding.antibody import BALMPairedEmbedder
        return BALMPairedEmbedder(device=device, trainable=trainable)
    else:
        raise ValueError(f"Unknown embedder: {embedder_type}")


def get_embedding_cache_path(dataset: str, embedder_type: str) -> Path:
    """Get path for embedding cache file."""
    cache_dir = Path('experiments/antibody/embedding_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f'{dataset}_{embedder_type}.npy'


def load_embedding_cache(dataset: str, embedder_type: str) -> Optional[np.ndarray]:
    """Load embeddings from cache if available."""
    cache_path = get_embedding_cache_path(dataset, embedder_type)
    if cache_path.exists():
        return np.load(cache_path)
    return None


def save_embedding_cache(embeddings: np.ndarray, dataset: str, embedder_type: str):
    """Save embeddings to cache."""
    cache_path = get_embedding_cache_path(dataset, embedder_type)
    np.save(cache_path, embeddings)
    print(f"   Saved embedding cache to {cache_path}")


def embed_antibodies(
    df: pd.DataFrame,
    embedder,
    device: str,
    batch_size: int = 32,
    dataset: str = None,
    embedder_type: str = None,
    use_cache: bool = True,
    cache_threshold: int = 10000,
) -> np.ndarray:
    """
    Embed antibodies using the given embedder.

    If use_cache=True and dataset size > cache_threshold, will save/load
    embeddings to/from disk cache. This is critical for large datasets
    like abbibench_raw (185k samples) since embedders are frozen.
    """
    n_samples = len(df)

    # Check cache for large datasets
    should_cache = use_cache and dataset and embedder_type and n_samples > cache_threshold
    if should_cache:
        cached = load_embedding_cache(dataset, embedder_type)
        if cached is not None and len(cached) == n_samples:
            print(f"   Loaded embeddings from cache ({len(cached)} samples)")
            return cached

    embeddings = []

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]

            for _, row in batch.iterrows():
                try:
                    heavy = row['heavy_seq']
                    light = row['light_seq'] if row['light_seq'] else ''

                    output = embedder.encode(heavy, light)
                    embeddings.append(output.global_embedding.cpu().numpy())
                except Exception as e:
                    # Use zeros for failed embeddings
                    embeddings.append(np.zeros(embedder.embedding_dim))

            # Progress indicator for large datasets
            if n_samples > 1000 and (i + batch_size) % 5000 == 0:
                print(f"      Embedded {i + batch_size}/{n_samples} antibodies...")

    embeddings_arr = np.array(embeddings)

    # Save to cache for large datasets
    if should_cache:
        save_embedding_cache(embeddings_arr, dataset, embedder_type)

    return embeddings_arr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Correlation coefficients
    if np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
        pearson = 0.0
        spearman = 0.0
    else:
        pearson, _ = scipy_stats.pearsonr(y_pred, y_true)
        spearman, _ = scipy_stats.spearmanr(y_pred, y_true)

    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson': float(pearson),
        'spearman': float(spearman),
        'n_samples': len(y_true),
    }


def run_experiment(
    dataset: str,
    embedder_type: str,
    device: str = 'auto',
    hidden_dims: List[int] = [256, 128],
    dropout: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 15,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run property prediction experiment for one dataset + embedder combination.

    Returns:
        Dictionary with results and metrics
    """
    if device == 'auto':
        device = get_device()

    results = {
        'dataset': dataset,
        'embedder': embedder_type,
        'device': device,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': {
            'hidden_dims': hidden_dims,
            'dropout': dropout,
            'lr': lr,
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'patience': patience,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'random_seed': random_seed,
        },
    }

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset} | Embedder: {embedder_type}")
    print(f"{'='*70}")

    # Load and prepare data
    print("\n1. Loading data...")
    try:
        if dataset == 'abbibench_raw':
            # Load raw abbibench directly (185k samples)
            df_flat = load_abbibench_raw()
            print(f"   Loaded {len(df_flat)} antibodies (raw format)")
        elif dataset == 'magma_seq':
            # Load magma_seq directly (~2.9k samples)
            df_flat = load_magma_seq_flat()
            print(f"   Loaded {len(df_flat)} antibodies (magma_seq)")
        else:
            df = load_unified_data(dataset)
            print(f"   Loaded {len(df)} entries")

            # Flatten to unique antibodies
            print("\n2. Flattening to unique antibodies...")
            df_flat = flatten_to_unique_antibodies(df, use_mutant=True)
            print(f"   {len(df_flat)} unique antibodies")

        print(f"   Property value range: [{df_flat['property_value'].min():.3f}, {df_flat['property_value'].max():.3f}]")
    except Exception as e:
        results['error'] = f"Failed to load/flatten data: {e}"
        return results

    if len(df_flat) < 100:
        results['error'] = f"Not enough data: {len(df_flat)} antibodies (need >= 100)"
        return results

    results['n_antibodies'] = len(df_flat)
    results['property_range'] = [float(df_flat['property_value'].min()), float(df_flat['property_value'].max())]

    # Create embedder
    print(f"\n3. Creating {embedder_type} embedder...")
    try:
        embedder = create_embedder(embedder_type, device)
        print(f"   Embedding dimension: {embedder.embedding_dim}")
        results['embedding_dim'] = embedder.embedding_dim
    except Exception as e:
        print(f"   ERROR: Failed to create embedder: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = f"Failed to create embedder: {e}"
        return results

    # Embed ALL data BEFORE shuffling (for consistent caching)
    print("\n4. Embedding antibodies...")
    start_time = time.time()

    try:
        X_all = embed_antibodies(
            df_flat, embedder, device,
            dataset=dataset,
            embedder_type=embedder_type,
            use_cache=True,
            cache_threshold=10000,  # Cache if > 10k samples
        )
        embed_time = time.time() - start_time
        print(f"   Embedding time: {embed_time:.1f}s")
        results['embedding_time_s'] = embed_time
    except Exception as e:
        print(f"   ERROR: Failed to embed: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = f"Failed to embed: {e}"
        del embedder
        gc.collect()
        return results

    # Get all labels
    y_all = df_flat['property_value'].values.astype(np.float32)

    # Split data (random, but ensuring no leakage)
    print("\n5. Splitting data...")
    np.random.seed(random_seed)

    # Create shuffled indices
    indices = np.random.permutation(len(df_flat))

    n_train = int(len(df_flat) * train_ratio)
    n_val = int(len(df_flat) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    X_train = X_all[train_idx]
    X_val = X_all[val_idx]
    X_test = X_all[test_idx]

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    results['splits'] = {
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test),
    }

    # Train model using PropertyPredictorMLP
    print("\n6. Training MLP predictor...")
    from src.models.predictors import PropertyPredictorMLP
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = PropertyPredictorMLP(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=lr,
        n_tasks=1,
    )

    # Early stopping callback
    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if device == 'cuda' else ('mps' if device == 'mps' else 'cpu'),
        devices=1,
        callbacks=[early_stop],
        enable_progress_bar=verbose,
        enable_model_summary=False,
        logger=False,
    )

    # Train
    start_time = time.time()
    trainer.fit(model, train_loader, val_loader)
    train_time = time.time() - start_time

    print(f"   Training time: {train_time:.1f}s")
    print(f"   Epochs trained: {trainer.current_epoch + 1}")

    results['training_time_s'] = train_time
    results['epochs_trained'] = trainer.current_epoch + 1

    # Evaluate
    print("\n7. Evaluating...")
    model.eval()
    model.to(device)

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    results['metrics'] = metrics

    print(f"\n   Results:")
    print(f"   MAE:      {metrics['mae']:.4f}")
    print(f"   RMSE:     {metrics['rmse']:.4f}")
    print(f"   R²:       {metrics['r2']:.4f}")
    print(f"   Pearson:  {metrics['pearson']:.4f}")
    print(f"   Spearman: {metrics['spearman']:.4f}")

    # Cleanup
    del embedder, model, trainer
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()

    return results


def run_all_experiments(
    datasets: List[str] = None,
    embedders: List[str] = None,
    output_dir: str = 'experiments/antibody/results/exp1_property_prediction',
    **kwargs
) -> Dict[str, Any]:
    """Run experiments for all dataset-embedder combinations."""

    if datasets is None:
        # Default datasets (NOT abagym):
        # - skempi2: ~600 mutation pairs
        # - magma_seq: ~2.9k samples
        # - trastuzumab_dms: ~36k DMS data
        # - abbibench_raw: 185k samples (run last - largest)
        datasets = ['skempi2', 'magma_seq', 'trastuzumab_dms', 'abbibench_raw']

    if embedders is None:
        embedders = ['igbert', 'igt5', 'ablang2', 'antiberta2', 'balm', 'balm_paired']

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        'experiment': 'property_prediction',
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'datasets': datasets,
        'embedders': embedders,
        'results': [],
    }

    print("="*70)
    print("EXPERIMENT 1: ANTIBODY PROPERTY PREDICTION")
    print("="*70)
    print(f"\nDatasets: {datasets}")
    print(f"Embedders: {embedders}")
    print(f"Total combinations: {len(datasets) * len(embedders)}")

    for dataset in datasets:
        for embedder in embedders:
            try:
                result = run_experiment(
                    dataset=dataset,
                    embedder_type=embedder,
                    **kwargs
                )
                all_results['results'].append(result)
            except Exception as e:
                print(f"\nERROR in {dataset}/{embedder}: {e}")
                all_results['results'].append({
                    'dataset': dataset,
                    'embedder': embedder,
                    'error': str(e),
                })

            # Save individual result for crash resilience
            individual_file = output_path / f"{dataset}_{embedder}_{all_results['timestamp']}.json"
            with open(individual_file, 'w') as f:
                json.dump(all_results['results'][-1], f, indent=2)

            # Also save intermediate full results
            intermediate_file = output_path / f"results_{all_results['timestamp']}_partial.json"
            with open(intermediate_file, 'w') as f:
                json.dump(all_results, f, indent=2)

            # Flush memory between embedders
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # Save results
    results_file = output_path / f"results_{all_results['timestamp']}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Print summary table
    print(f"\n{'Dataset':<20} {'Embedder':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'Pearson':<10}")
    print("-" * 75)

    for r in all_results['results']:
        if 'error' in r:
            print(f"{r['dataset']:<20} {r['embedder']:<15} ERROR: {r['error'][:30]}")
        else:
            m = r['metrics']
            print(f"{r['dataset']:<20} {r['embedder']:<15} {m['mae']:<10.4f} {m['rmse']:<10.4f} {m['r2']:<10.4f} {m['pearson']:<10.4f}")

    print(f"\nResults saved to: {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Property Prediction')

    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to run')
    parser.add_argument('--embedder', type=str, default=None,
                        help='Specific embedder to use')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of datasets to run (for --all mode)')
    parser.add_argument('--embedders', type=str, nargs='+', default=None,
                        help='List of embedders to use (for --all mode)')
    parser.add_argument('--all', action='store_true',
                        help='Run all dataset-embedder combinations')

    # Training config
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--output_dir', type=str,
                        default='experiments/antibody/results/exp1_property_prediction')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    kwargs = {
        'hidden_dims': args.hidden_dims,
        'dropout': args.dropout,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'random_seed': args.seed,
        'verbose': not args.quiet,
    }

    if args.all:
        run_all_experiments(
            datasets=args.datasets,
            embedders=args.embedders,
            output_dir=args.output_dir,
            **kwargs
        )
    elif args.dataset and args.embedder:
        result = run_experiment(
            dataset=args.dataset,
            embedder_type=args.embedder,
            **kwargs
        )

        # Save single result
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / f"{args.dataset}_{args.embedder}_{result['timestamp']}.json"
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    else:
        parser.error("Either --all or both --dataset and --embedder must be specified")


if __name__ == '__main__':
    main()

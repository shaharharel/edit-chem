#!/usr/bin/env python3
"""
Comprehensive comparison of RNA embedders on 3'UTR and 5'UTR datasets.

This experiment compares:
- Embedders: NucleotideEmbedder, RNAFMEmbedder, RNABERTEmbedder, UTRLMEmbedder
- Tasks: Regression (MSE), Binary Classification (direction prediction)
- Modes: Frozen embedder, Trainable embedder (where applicable)
- Datasets: 5'UTR (MRL prediction), 3'UTR (skew prediction)

All experiments use fixed seeds for reproducibility.

Usage:
    # Full comparison (all embedders, both datasets, both tasks)
    python compare_rna_embedders.py

    # Quick sanity check (minimal epochs, small samples)
    python compare_rna_embedders.py --sanity-check

    # Specific configuration
    python compare_rna_embedders.py --embedders nucleotide rnafm --datasets 5utr --tasks regression
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Fixed seed for reproducibility
GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASETS = {
    '5utr': {
        'path': 'data/rna/pairs/mpra_5utr_pairs_long.csv',
        'property_col': 'property_name',
        'property_filter': ['MRL_5UTR'],
        'description': '5\'UTR Mean Ribosome Loading (Griesemer et al.)'
    },
    '3utr': {
        'path': 'data/rna/pairs/mprau_3utr_multitask_with_seq.csv',
        'property_col': 'property_name',
        'property_filter': ['3UTR_skew_HEK293FT'],
        'description': '3\'UTR Allelic Skew HEK293FT (Griesemer/Ulirsch et al.)'
    }
}

# =============================================================================
# Embedder Configuration
# =============================================================================

def get_embedder_configs() -> Dict:
    """Return embedder configurations."""
    return {
        'nucleotide': {
            'class': 'NucleotideEmbedder',
            'supports_trainable': False,  # Feature-based, not trainable
            'init_kwargs': {
                'include_onehot': True,
                'include_kmers': True,
                'kmer_sizes': [3, 4],
                'include_stats': True,
                'include_positional': True
            }
        },
        'rnafm': {
            'class': 'RNAFMEmbedder',
            'supports_trainable': True,
            'init_kwargs': {
                'model_name': 'rna_fm_t12',
                'pooling': 'mean',
                'batch_size': 32
            }
        },
        'rnabert': {
            'class': 'RNABERTEmbedder',
            'supports_trainable': True,
            'init_kwargs': {
                'model_name': 'multimolecule/rnabert',
                'pooling': 'mean',
                'batch_size': 32,
                'max_length': 440
            }
        },
        'utrlm': {
            'class': 'UTRLMEmbedder',
            'supports_trainable': True,
            'init_kwargs': {
                'model_path': 'multimolecule/utrlm-te_el',
                'pooling': 'mean',
                'max_length': 256
            }
        }
    }


def create_embedder(embedder_name: str, trainable: bool = False, device: str = 'auto'):
    """Create an embedder instance."""
    config = get_embedder_configs()[embedder_name]

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if embedder_name == 'nucleotide':
        from src.embedding.rna import NucleotideEmbedder
        return NucleotideEmbedder(**config['init_kwargs'])

    elif embedder_name == 'rnafm':
        from src.embedding.rna.rnafm import RNAFMEmbedder
        return RNAFMEmbedder(
            device=device,
            trainable=trainable,
            **config['init_kwargs']
        )

    elif embedder_name == 'rnabert':
        from src.embedding.rna.rnabert import RNABERTEmbedder
        return RNABERTEmbedder(
            device=device,
            trainable=trainable,
            **config['init_kwargs']
        )

    elif embedder_name == 'utrlm':
        from src.embedding.rna.utrlm import UTRLMEmbedder
        embedder = UTRLMEmbedder(
            trainable=trainable,
            **config['init_kwargs']
        )
        # Move to device
        if hasattr(embedder, '_hf_model'):
            embedder._hf_model = embedder._hf_model.to(device)
        return embedder

    else:
        raise ValueError(f"Unknown embedder: {embedder_name}")


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(dataset_name: str, max_samples: Optional[int] = None) -> pd.DataFrame:
    """Load and preprocess dataset."""
    config = DATASETS[dataset_name]

    # Find project root
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / config['path']

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # Filter by property
    if config['property_filter']:
        df = df[df[config['property_col']].isin(config['property_filter'])]

    # Require sequences
    df = df.dropna(subset=['seq_a', 'seq_b', 'delta'])

    # Subsample if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=GLOBAL_SEED)

    print(f"Loaded {dataset_name}: {len(df)} pairs")
    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15,
               seed: int = GLOBAL_SEED) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test."""
    np.random.seed(seed)

    n = len(df)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


# =============================================================================
# Embedding Computation
# =============================================================================

def compute_embeddings(embedder, sequences: List[str], batch_size: int = 32,
                       show_progress: bool = True) -> np.ndarray:
    """Compute embeddings for sequences."""
    all_embeddings = []

    iterator = range(0, len(sequences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding", leave=False)

    for i in iterator:
        batch = sequences[i:i+batch_size]
        emb = embedder.encode(batch)
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        all_embeddings.append(emb)

    return np.vstack(all_embeddings)


# =============================================================================
# Models
# =============================================================================

class RegressionHead(nn.Module):
    """Simple MLP head for regression."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.1):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ClassificationHead(nn.Module):
    """MLP head for binary classification (direction prediction)."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.1):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# Training
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task: str,  # 'regression' or 'classification'
    max_epochs: int = 30,
    lr: float = 0.001,
    patience: int = 5,
    device: str = 'cpu'
) -> Dict:
    """Train model with early stopping."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1,
        'history': history
    }


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    task: str,
    device: str = 'cpu'
) -> Dict:
    """Evaluate model on test set."""

    model = model.to(device)
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)

            if task == 'classification':
                outputs = torch.sigmoid(outputs)

            all_preds.extend(outputs.cpu().numpy())
            all_true.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    metrics = {}

    if task == 'regression':
        metrics['mae'] = np.mean(np.abs(all_true - all_preds))
        metrics['rmse'] = np.sqrt(np.mean((all_true - all_preds) ** 2))
        metrics['r2'] = 1 - np.sum((all_true - all_preds) ** 2) / np.sum((all_true - np.mean(all_true)) ** 2)
        metrics['pearson_r'] = stats.pearsonr(all_true, all_preds)[0]
        metrics['spearman_r'] = stats.spearmanr(all_true, all_preds)[0]

        # Direction accuracy (for regression, based on sign)
        metrics['direction_accuracy'] = np.mean(np.sign(all_preds) == np.sign(all_true))

    else:  # classification
        pred_labels = (all_preds > 0.5).astype(int)
        true_labels = all_true.astype(int)

        metrics['accuracy'] = np.mean(pred_labels == true_labels)
        metrics['precision'] = np.sum((pred_labels == 1) & (true_labels == 1)) / max(np.sum(pred_labels == 1), 1)
        metrics['recall'] = np.sum((pred_labels == 1) & (true_labels == 1)) / max(np.sum(true_labels == 1), 1)
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / max(metrics['precision'] + metrics['recall'], 1e-8)

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            metrics['auc'] = roc_auc_score(true_labels, all_preds)
        except:
            metrics['auc'] = 0.5

    return metrics


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_single_experiment(
    embedder_name: str,
    dataset_name: str,
    task: str,
    trainable: bool,
    max_epochs: int = 30,
    batch_size: int = 64,
    hidden_dims: List[int] = None,
    lr: float = 0.001,
    max_samples: Optional[int] = None,
    device: str = 'auto'
) -> Dict:
    """Run a single experiment configuration."""

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(GLOBAL_SEED)

    # Check if embedder supports trainable mode
    config = get_embedder_configs()[embedder_name]
    if trainable and not config['supports_trainable']:
        return {'skipped': True, 'reason': f'{embedder_name} does not support trainable mode'}

    result = {
        'embedder': embedder_name,
        'dataset': dataset_name,
        'task': task,
        'trainable': trainable,
        'device': device
    }

    try:
        # Load data
        df = load_dataset(dataset_name, max_samples=max_samples)
        train_df, val_df, test_df = split_data(df)

        result['train_size'] = len(train_df)
        result['val_size'] = len(val_df)
        result['test_size'] = len(test_df)

        # Create embedder
        print(f"  Creating {embedder_name} embedder (trainable={trainable})...")
        embedder = create_embedder(embedder_name, trainable=trainable, device=device)
        result['embedding_dim'] = embedder.embedding_dim if hasattr(embedder, 'embedding_dim') else embedder.output_dim

        # Compute embeddings
        print(f"  Computing embeddings...")
        all_seqs = list(set(
            train_df['seq_a'].tolist() + train_df['seq_b'].tolist() +
            val_df['seq_a'].tolist() + val_df['seq_b'].tolist() +
            test_df['seq_a'].tolist() + test_df['seq_b'].tolist()
        ))

        embeddings = compute_embeddings(embedder, all_seqs, show_progress=True)
        seq_to_idx = {seq: i for i, seq in enumerate(all_seqs)}

        # Prepare data
        def prepare_data(df_split):
            X = []
            y = []
            for _, row in df_split.iterrows():
                emb_a = embeddings[seq_to_idx[row['seq_a']]]
                emb_b = embeddings[seq_to_idx[row['seq_b']]]
                # Edit embedding = difference
                X.append(emb_b - emb_a)

                if task == 'regression':
                    y.append(row['delta'])
                else:  # classification
                    y.append(1.0 if row['delta'] > 0 else 0.0)

            return np.array(X), np.array(y)

        X_train, y_train = prepare_data(train_df)
        X_val, y_val = prepare_data(val_df)
        X_test, y_test = prepare_data(test_df)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Create model
        input_dim = X_train.shape[1]
        if hidden_dims is None:
            hidden_dims = [256, 128]

        if task == 'regression':
            model = RegressionHead(input_dim, hidden_dims)
        else:
            model = ClassificationHead(input_dim, hidden_dims)

        # Train
        print(f"  Training...")
        train_result = train_model(
            model, train_loader, val_loader,
            task=task, max_epochs=max_epochs, lr=lr,
            patience=5, device=device
        )
        result['epochs_trained'] = train_result['epochs_trained']
        result['best_val_loss'] = train_result['best_val_loss']

        # Evaluate
        print(f"  Evaluating...")
        metrics = evaluate_model(model, test_loader, task=task, device=device)
        result['metrics'] = metrics

        result['success'] = True

        # Clean up GPU memory
        del embedder, model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()

    return result


def run_comparison(
    embedders: List[str] = None,
    datasets: List[str] = None,
    tasks: List[str] = None,
    test_trainable: bool = True,
    max_epochs: int = 30,
    max_samples: Optional[int] = None,
    output_dir: str = 'results/embedder_comparison'
) -> pd.DataFrame:
    """Run full comparison across all configurations."""

    if embedders is None:
        embedders = ['nucleotide', 'rnafm', 'rnabert', 'utrlm']
    if datasets is None:
        datasets = ['5utr', '3utr']
    if tasks is None:
        tasks = ['regression', 'classification']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"RNA EMBEDDER COMPARISON EXPERIMENT")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Embedders: {embedders}")
    print(f"Datasets: {datasets}")
    print(f"Tasks: {tasks}")
    print(f"Test trainable: {test_trainable}")
    print(f"Max epochs: {max_epochs}")
    print(f"Max samples: {max_samples or 'all'}")
    print(f"{'='*80}\n")

    results = []
    total_experiments = len(embedders) * len(datasets) * len(tasks) * (2 if test_trainable else 1)
    exp_num = 0

    for dataset in datasets:
        for task in tasks:
            for embedder in embedders:
                trainable_modes = [False]
                if test_trainable and get_embedder_configs()[embedder]['supports_trainable']:
                    trainable_modes.append(True)

                for trainable in trainable_modes:
                    exp_num += 1
                    mode = "trainable" if trainable else "frozen"

                    print(f"\n[{exp_num}/{total_experiments}] {embedder} | {dataset} | {task} | {mode}")
                    print("-" * 60)

                    start_time = time.time()
                    result = run_single_experiment(
                        embedder_name=embedder,
                        dataset_name=dataset,
                        task=task,
                        trainable=trainable,
                        max_epochs=max_epochs,
                        max_samples=max_samples,
                        device=device
                    )
                    result['runtime_seconds'] = time.time() - start_time

                    if result.get('success'):
                        metrics = result.get('metrics', {})
                        if task == 'regression':
                            print(f"  Results: MAE={metrics.get('mae', 0):.4f}, "
                                  f"R²={metrics.get('r2', 0):.4f}, "
                                  f"Pearson={metrics.get('pearson_r', 0):.4f}, "
                                  f"Dir.Acc={metrics.get('direction_accuracy', 0):.2%}")
                        else:
                            print(f"  Results: Accuracy={metrics.get('accuracy', 0):.2%}, "
                                  f"F1={metrics.get('f1', 0):.4f}, "
                                  f"AUC={metrics.get('auc', 0):.4f}")
                    elif result.get('skipped'):
                        print(f"  SKIPPED: {result.get('reason')}")
                    else:
                        print(f"  FAILED: {result.get('error')}")

                    results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results
    results_df.to_csv(output_path / f'results_{timestamp}.csv', index=False)

    # Save summary
    summary = generate_summary(results_df)
    with open(output_path / f'summary_{timestamp}.txt', 'w') as f:
        f.write(summary)

    # Save JSON for programmatic access
    with open(output_path / f'results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"\n{summary}")

    return results_df


def generate_summary(df: pd.DataFrame) -> str:
    """Generate summary text from results."""
    lines = []
    lines.append("=" * 80)
    lines.append("SUMMARY: RNA EMBEDDER COMPARISON")
    lines.append("=" * 80)

    # Successful experiments
    if 'success' in df.columns:
        successful = df[df['success'] == True]
    else:
        successful = df

    if 'skipped' in df.columns:
        skipped = df[df['skipped'] == True]
    else:
        skipped = pd.DataFrame()

    if 'success' in df.columns and 'skipped' in df.columns:
        failed = df[(df['success'] == False) & (df['skipped'] == False)]
    elif 'success' in df.columns:
        failed = df[df['success'] == False]
    else:
        failed = pd.DataFrame()

    lines.append(f"\nTotal experiments: {len(df)}")
    lines.append(f"  Successful: {len(successful)}")
    lines.append(f"  Skipped: {len(skipped)}")
    lines.append(f"  Failed: {len(failed)}")

    if len(successful) == 0:
        return "\n".join(lines)

    # Extract metrics
    metrics_data = []
    for _, row in successful.iterrows():
        if 'metrics' in row and isinstance(row['metrics'], dict):
            metrics_data.append({
                'embedder': row['embedder'],
                'dataset': row['dataset'],
                'task': row['task'],
                'trainable': row['trainable'],
                **row['metrics']
            })

    if not metrics_data:
        return "\n".join(lines)

    metrics_df = pd.DataFrame(metrics_data)

    # Best per task
    lines.append("\n" + "=" * 80)
    lines.append("BEST RESULTS BY TASK")
    lines.append("=" * 80)

    for task in metrics_df['task'].unique():
        task_df = metrics_df[metrics_df['task'] == task]

        if task == 'regression':
            best_idx = task_df['pearson_r'].idxmax()
            best = task_df.loc[best_idx]
            lines.append(f"\n{task.upper()}:")
            lines.append(f"  Best: {best['embedder']} ({'trainable' if best['trainable'] else 'frozen'})")
            lines.append(f"  Dataset: {best['dataset']}")
            lines.append(f"  Pearson r: {best['pearson_r']:.4f}")
            lines.append(f"  R²: {best['r2']:.4f}")
            lines.append(f"  MAE: {best['mae']:.4f}")
        else:
            best_idx = task_df['accuracy'].idxmax()
            best = task_df.loc[best_idx]
            lines.append(f"\n{task.upper()}:")
            lines.append(f"  Best: {best['embedder']} ({'trainable' if best['trainable'] else 'frozen'})")
            lines.append(f"  Dataset: {best['dataset']}")
            lines.append(f"  Accuracy: {best['accuracy']:.2%}")
            lines.append(f"  F1: {best['f1']:.4f}")
            lines.append(f"  AUC: {best['auc']:.4f}")

    # Per-embedder summary
    lines.append("\n" + "=" * 80)
    lines.append("RESULTS BY EMBEDDER")
    lines.append("=" * 80)

    for embedder in metrics_df['embedder'].unique():
        emb_df = metrics_df[metrics_df['embedder'] == embedder]
        lines.append(f"\n{embedder.upper()}:")

        for dataset in emb_df['dataset'].unique():
            ds_df = emb_df[emb_df['dataset'] == dataset]
            for _, row in ds_df.iterrows():
                mode = "trainable" if row['trainable'] else "frozen"
                if row['task'] == 'regression':
                    lines.append(f"  {dataset} | {row['task']} | {mode}: "
                               f"R²={row['r2']:.3f}, Pearson={row['pearson_r']:.3f}")
                else:
                    lines.append(f"  {dataset} | {row['task']} | {mode}: "
                               f"Acc={row['accuracy']:.1%}, AUC={row['auc']:.3f}")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare RNA embedders on UTR datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full comparison
    python compare_rna_embedders.py

    # Quick sanity check
    python compare_rna_embedders.py --sanity-check

    # Specific configuration
    python compare_rna_embedders.py --embedders nucleotide rnafm --datasets 5utr --tasks regression
        """
    )

    parser.add_argument('--embedders', nargs='+',
                       choices=['nucleotide', 'rnafm', 'rnabert', 'utrlm'],
                       default=['nucleotide', 'rnafm', 'rnabert', 'utrlm'],
                       help='Embedders to test')
    parser.add_argument('--datasets', nargs='+',
                       choices=['5utr', '3utr'],
                       default=['5utr', '3utr'],
                       help='Datasets to use')
    parser.add_argument('--tasks', nargs='+',
                       choices=['regression', 'classification'],
                       default=['regression', 'classification'],
                       help='Tasks to run')
    parser.add_argument('--no-trainable', action='store_true',
                       help='Skip trainable embedder experiments')
    parser.add_argument('--max-epochs', type=int, default=30,
                       help='Maximum training epochs')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit samples per dataset (for testing)')
    parser.add_argument('--output-dir', type=str,
                       default='results/embedder_comparison',
                       help='Output directory')
    parser.add_argument('--sanity-check', action='store_true',
                       help='Quick sanity check with minimal configuration')

    args = parser.parse_args()

    # Sanity check mode
    if args.sanity_check:
        print("\n" + "=" * 80)
        print("SANITY CHECK MODE - Minimal configuration")
        print("=" * 80)

        args.embedders = ['nucleotide']  # Fastest embedder
        args.datasets = ['5utr']  # Smaller dataset
        args.tasks = ['regression']
        args.max_epochs = 3
        args.max_samples = 500
        args.no_trainable = True

    # Run comparison
    results_df = run_comparison(
        embedders=args.embedders,
        datasets=args.datasets,
        tasks=args.tasks,
        test_trainable=not args.no_trainable,
        max_epochs=args.max_epochs,
        max_samples=args.max_samples,
        output_dir=args.output_dir
    )

    return results_df


if __name__ == '__main__':
    main()

"""
Main entry point for antibody mutation effect prediction experiments.

This module runs comprehensive experiments comparing different methods for
predicting mutation effects on antibodies. It supports:
- Multiple antibody language models (IgBert, IgT5, AntiBERTa2, etc.)
- Simple and structured edit embeddings
- Multiple datasets (AbAgym, SKEMPI2, AbBiBench, Trastuzumab)
- HTML and DOCX report generation with visualizations

Usage:
    # Run with unified dataset
    python -m experiments.antibody.main

    # Run quick test with sample data
    python -m experiments.antibody.main --quick_test

    # Run with specific configuration
    python -m experiments.antibody.main --embedder igbert --max_samples 1000
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_device() -> str:
    """Get best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_unified_dataset(
    data_file: str,
    source_datasets: Optional[List[str]] = None,
    assay_types: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    require_sequences: bool = True,
) -> pd.DataFrame:
    """
    Load the unified antibody dataset.

    Args:
        data_file: Path to unified JSON or CSV file
        source_datasets: Filter by source dataset names
        assay_types: Filter by assay type
        max_samples: Maximum samples to load
        require_sequences: Only include entries with sequences

    Returns:
        DataFrame with loaded data
    """
    data_path = Path(data_file)

    if data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} entries from {data_file}")

    # Filter by source
    if source_datasets:
        df = df[df['source_dataset'].isin(source_datasets)]
        print(f"  Filtered to {len(df)} entries from {source_datasets}")

    # Filter by assay type
    if assay_types:
        df = df[df['assay_type'].isin(assay_types)]
        print(f"  Filtered to {len(df)} entries with assay types {assay_types}")

    # Filter entries without sequences
    if require_sequences:
        df = df[df['heavy_wt'].str.len() > 0]
        print(f"  Filtered to {len(df)} entries with sequences")

    # Sample if needed
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"  Sampled {max_samples} entries")

    return df


def create_embedder(embedder_type: str, device: str = 'auto'):
    """Create antibody embedder."""
    if device == 'auto':
        device = get_device()

    print(f"Creating {embedder_type} embedder on {device}...")

    try:
        if embedder_type == 'igbert':
            from src.embedding.antibody import IgBertEmbedder
            return IgBertEmbedder(device=device)
        elif embedder_type == 'igt5':
            from src.embedding.antibody import IgT5Embedder
            return IgT5Embedder(device=device)
        elif embedder_type == 'ablang2':
            from src.embedding.antibody import AbLang2Embedder
            return AbLang2Embedder(device=device)
        elif embedder_type == 'antiberta2':
            from src.embedding.antibody import AntiBERTa2Embedder
            return AntiBERTa2Embedder(device=device)
        else:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
    except Exception as e:
        print(f"  Warning: Could not load {embedder_type}: {e}")
        print(f"  Falling back to IgBert...")
        from src.embedding.antibody import IgBertEmbedder
        return IgBertEmbedder(device=device)


def create_edit_embedder(base_embedder, embedder_type: str = 'simple', **kwargs):
    """Create edit embedder."""
    if embedder_type == 'simple':
        from src.embedding.antibody import AntibodyEditEmbedder
        return AntibodyEditEmbedder(base_embedder=base_embedder)
    elif embedder_type == 'structured':
        from src.embedding.antibody import StructuredAntibodyEditEmbedder
        return StructuredAntibodyEditEmbedder(
            base_embedder=base_embedder,
            output_dim=kwargs.get('output_dim', 320),
        )
    else:
        raise ValueError(f"Unknown edit embedder type: {embedder_type}")


def embed_samples(
    df: pd.DataFrame,
    edit_embedder,
    batch_size: int = 32,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Embed samples using the edit embedder.

    Returns:
        Tuple of (context_embeddings, edit_embeddings, delta_values)
    """
    context_embeddings = []
    edit_embeddings = []
    delta_values = []

    print(f"Embedding {len(df)} samples...")

    with torch.no_grad():
        for idx, row in df.iterrows():
            try:
                # Get sequences
                heavy_wt = row['heavy_wt']
                light_wt = row.get('light_wt', '')
                heavy_mut = row.get('heavy_mut', heavy_wt)
                light_mut = row.get('light_mut', light_wt)

                # Handle NaN
                if pd.isna(light_wt):
                    light_wt = ''
                if pd.isna(light_mut):
                    light_mut = ''

                # Encode - handle both simple and structured embedders
                if hasattr(edit_embedder, 'encode_from_sequences'):
                    # Structured embedder
                    output = edit_embedder.encode_from_sequences(
                        wt_heavy=heavy_wt,
                        wt_light=light_wt,
                        mut_heavy=heavy_mut,
                        mut_light=light_mut,
                    )
                    context_embeddings.append(output.h_context.detach().clone())
                    edit_embeddings.append(output.z_edit.detach().clone())
                else:
                    # Simple edit embedder
                    output = edit_embedder.encode(
                        wt_heavy=heavy_wt,
                        wt_light=light_wt,
                        mut_heavy=heavy_mut,
                        mut_light=light_mut,
                    )
                    context_embeddings.append(output.context_embedding.detach().clone())
                    edit_embeddings.append(output.edit_embedding.detach().clone())

                delta_values.append(row['delta_value'])

            except Exception as e:
                # Skip problematic samples
                continue

    print(f"  Successfully embedded {len(context_embeddings)} samples")

    return (
        torch.stack(context_embeddings),
        torch.stack(edit_embeddings),
        torch.tensor(delta_values, dtype=torch.float32),
    )


def embed_property_samples(
    df: pd.DataFrame,
    base_embedder,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Embed samples for baseline property prediction.

    Returns:
        Tuple of (wt_embeddings, mut_embeddings, delta_values)
    """
    wt_embeddings = []
    mut_embeddings = []
    delta_values = []

    print(f"Embedding {len(df)} samples for property prediction...")

    with torch.no_grad():
        for idx, row in df.iterrows():
            try:
                heavy_wt = row['heavy_wt']
                light_wt = row.get('light_wt', '')
                heavy_mut = row.get('heavy_mut', heavy_wt)
                light_mut = row.get('light_mut', light_wt)

                if pd.isna(light_wt):
                    light_wt = ''
                if pd.isna(light_mut):
                    light_mut = ''

                # Encode wild-type
                wt_output = base_embedder.encode(heavy_wt, light_wt)
                wt_embeddings.append(wt_output.global_embedding.cpu().detach())

                # Encode mutant
                mut_output = base_embedder.encode(heavy_mut, light_mut)
                mut_embeddings.append(mut_output.global_embedding.cpu().detach())

                delta_values.append(row['delta_value'])

            except Exception as e:
                continue

    print(f"  Successfully embedded {len(wt_embeddings)} samples")

    return (
        torch.stack(wt_embeddings),
        torch.stack(mut_embeddings),
        torch.tensor(delta_values, dtype=torch.float32),
    )


def create_baseline_predictor(
    hidden_dims: List[int] = [256, 128],
    dropout: float = 0.2,
    lr: float = 1e-3,
    max_epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    device: str = 'cpu',
):
    """Create a baseline property predictor using the generalized class."""
    from src.models.predictors import BaselinePropertyPredictor
    return BaselinePropertyPredictor(
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )


def train_predictor(
    train_context: torch.Tensor,
    train_edit: torch.Tensor,
    train_y: torch.Tensor,
    val_context: torch.Tensor,
    val_edit: torch.Tensor,
    val_y: torch.Tensor,
    hidden_dims: List[int] = [256, 128],
    dropout: float = 0.2,
    lr: float = 1e-3,
    max_epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    device: str = 'cpu',
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Train a simple MLP predictor.

    Returns:
        Tuple of (model, training_history)
    """
    from src.models.predictors import EditEffectMLP

    context_dim = train_context.shape[1]
    edit_dim = train_edit.shape[1]

    model = EditEffectMLP(
        mol_dim=context_dim,
        edit_dim=edit_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        learning_rate=lr,
    )
    model = model.to(device)

    # Create dataloaders
    train_dataset = TensorDataset(
        train_context.to(device),
        train_edit.to(device),
        train_y.to(device),
    )
    val_dataset = TensorDataset(
        val_context.to(device),
        val_edit.to(device),
        val_y.to(device),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_losses = []
        for context, edit, y in train_loader:
            optimizer.zero_grad()
            pred = model(context, edit)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for context, edit, y in val_loader:
                pred = model(context, edit)
                loss = criterion(pred.squeeze(), y)
                val_losses.append(loss.item())
                val_preds.extend(pred.squeeze().cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_mae = np.mean(np.abs(np.array(val_preds) - np.array(val_targets)))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{max_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mae={val_mae:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return model, history


def evaluate_model(
    model: nn.Module,
    test_context: torch.Tensor,
    test_edit: torch.Tensor,
    test_y: torch.Tensor,
    device: str = 'cpu',
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model on test set.

    Returns:
        Tuple of (metrics_dict, y_true, y_pred)
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        pred = model(test_context.to(device), test_edit.to(device))
        pred = pred.squeeze().cpu().numpy()
        true = test_y.numpy()

    # Calculate metrics
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)

    # Correlations
    if len(pred) > 2:
        pearson_r, _ = stats.pearsonr(pred, true)
        spearman_r, _ = stats.spearmanr(pred, true)
    else:
        pearson_r = spearman_r = 0.0

    # R2
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'pearson': pearson_r,
        'spearman': spearman_r,
        'r2': r2,
    }

    return metrics, true, pred


def run_experiment(
    data_file: str,
    methods: List[Dict[str, Any]],
    output_dir: str,
    experiment_name: str = 'antibody_experiment',
    source_datasets: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    device: str = 'auto',
    generate_html: bool = True,
    generate_docx: bool = True,
) -> Dict[str, Any]:
    """
    Run a complete antibody mutation effect prediction experiment.
    """
    if device == 'auto':
        device = get_device()

    print(f"\n{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")

    # Load data
    df = load_unified_dataset(
        data_file=data_file,
        source_datasets=source_datasets,
        max_samples=max_samples,
        require_sequences=True,
    )

    if len(df) == 0:
        raise ValueError("No valid samples found in dataset")

    # Split data
    np.random.seed(random_seed)
    indices = np.random.permutation(len(df))
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    train_df = df.iloc[indices[:n_train]]
    val_df = df.iloc[indices[n_train:n_train + n_val]]
    test_df = df.iloc[indices[n_train + n_val:]]

    print(f"\nData split:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")

    # Initialize report generator
    from experiments.antibody.report_generator import AntibodyReportGenerator
    report_gen = AntibodyReportGenerator(
        output_dir=output_dir,
        experiment_name=experiment_name,
    )
    report_gen.set_config({
        'data_file': data_file,
        'source_datasets': source_datasets,
        'max_samples': max_samples,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_seed': random_seed,
        'device': device,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
    })

    # Run each method
    results = {}

    for method_config in methods:
        method_name = method_config['name']
        method_type = method_config.get('type', 'edit_framework')
        embedder_type = method_config.get('embedder_type', 'igbert')
        edit_embedder_type = method_config.get('edit_embedder_type', 'simple')

        print(f"\n{'='*70}")
        print(f"Running: {method_name}")
        print(f"  Method Type: {method_type}")
        print(f"  Embedder: {embedder_type}")
        if method_type != 'baseline_property':
            print(f"  Edit Embedder: {edit_embedder_type}")
        print(f"{'='*70}")

        try:
            # Create base embedder
            base_embedder = create_embedder(embedder_type, device)

            if method_type == 'baseline_property':
                # Baseline: predict absolute properties, compute delta
                print("\nEmbedding training data...")
                train_wt, train_mut, train_y = embed_property_samples(
                    train_df, base_embedder, device=device
                )

                print("Embedding validation data...")
                val_wt, val_mut, val_y = embed_property_samples(
                    val_df, base_embedder, device=device
                )

                print("Embedding test data...")
                test_wt, test_mut, test_y = embed_property_samples(
                    test_df, base_embedder, device=device
                )

                # Create and train baseline predictor using generalized class
                print("\nTraining baseline predictor...")
                baseline_predictor = create_baseline_predictor(
                    hidden_dims=method_config.get('hidden_dims', [256, 128]),
                    dropout=method_config.get('dropout', 0.2),
                    lr=method_config.get('lr', 1e-3),
                    max_epochs=method_config.get('max_epochs', 50),
                    batch_size=method_config.get('batch_size', 32),
                    patience=method_config.get('patience', 10),
                    device=device,
                )
                history = baseline_predictor.fit(
                    train_wt, train_mut, train_y,
                    val_wt, val_mut, val_y,
                    verbose=True,
                )

                # Evaluate
                print("\nEvaluating...")
                metrics, y_true, y_pred = baseline_predictor.evaluate(
                    test_wt, test_mut, test_y
                )

                # Use mut embeddings for visualization
                test_edit = test_mut
                model = baseline_predictor  # For cleanup

            else:
                # Edit framework methods (simple or structured)
                edit_embedder = create_edit_embedder(
                    base_embedder,
                    edit_embedder_type,
                    output_dim=method_config.get('output_dim', 320),
                )

                # Embed all data
                print("\nEmbedding training data...")
                train_context, train_edit, train_y = embed_samples(
                    train_df, edit_embedder, device=device
                )

                print("Embedding validation data...")
                val_context, val_edit, val_y = embed_samples(
                    val_df, edit_embedder, device=device
                )

                print("Embedding test data...")
                test_context, test_edit, test_y = embed_samples(
                    test_df, edit_embedder, device=device
                )

                # Train model
                print("\nTraining predictor...")
                model, history = train_predictor(
                    train_context, train_edit, train_y,
                    val_context, val_edit, val_y,
                    hidden_dims=method_config.get('hidden_dims', [256, 128]),
                    dropout=method_config.get('dropout', 0.2),
                    lr=method_config.get('lr', 1e-3),
                    max_epochs=method_config.get('max_epochs', 50),
                    batch_size=method_config.get('batch_size', 32),
                    patience=method_config.get('patience', 10),
                    device=device,
                )

                # Evaluate
                print("\nEvaluating...")
                metrics, y_true, y_pred = evaluate_model(
                    model, test_context, test_edit, test_y, device
                )

                del edit_embedder

            print(f"\nTest Results for {method_name}:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            results[method_name] = {
                'metrics': metrics,
                'config': method_config,
            }

            # Add to report
            report_gen.add_method_results(
                method_name=method_name,
                metrics=metrics,
                predictions=(y_true, y_pred),
                embeddings=test_edit.cpu().numpy(),
                training_history=history,
                metadata=method_config,
            )

            # Cleanup
            del model, base_embedder
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error running {method_name}: {e}")
            import traceback
            traceback.print_exc()
            results[method_name] = {'error': str(e)}

    # Generate reports
    print(f"\n{'='*70}")
    print("Generating reports...")
    print(f"{'='*70}")

    report_paths = report_gen.generate_reports(
        generate_html=generate_html,
        generate_docx=generate_docx,
    )

    print(f"\nReports generated:")
    for report_type, path in report_paths.items():
        print(f"  {report_type}: {path}")

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    summary_rows = []
    for method_name, res in results.items():
        if 'error' not in res:
            row = {'Method': method_name}
            row.update(res['metrics'])
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run antibody mutation effect prediction experiments'
    )
    parser.add_argument(
        '--data_file',
        type=str,
        default='data/antibody/unified/unified_antibody_data.json',
        help='Path to unified dataset file'
    )
    parser.add_argument(
        '--source',
        type=str,
        nargs='+',
        default=None,
        help='Filter by source datasets (e.g., skempi2 abagym)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples to use'
    )
    parser.add_argument(
        '--embedder',
        type=str,
        default='igbert',
        choices=['igbert', 'igt5', 'ablang2', 'antiberta2'],
        help='Antibody embedder to use'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/antibody/results',
        help='Output directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Run quick test with 500 samples and 5 epochs'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no_html',
        action='store_true',
        help='Skip HTML report generation'
    )
    parser.add_argument(
        '--no_docx',
        action='store_true',
        help='Skip DOCX report generation'
    )

    args = parser.parse_args()

    # Quick test settings
    if args.quick_test:
        args.max_samples = 500
        args.epochs = 5
        print("Running quick test with 500 samples and 5 epochs")

    # Define methods to compare
    methods = [
        {
            'name': f'Baseline Property - {args.embedder.upper()}',
            'type': 'baseline_property',
            'embedder_type': args.embedder,
            'hidden_dims': [256, 128],
            'dropout': 0.2,
            'lr': 1e-3,
            'max_epochs': args.epochs,
            'batch_size': 32,
            'patience': 10,
        },
        {
            'name': f'Edit Framework - {args.embedder.upper()}',
            'type': 'edit_framework',
            'embedder_type': args.embedder,
            'edit_embedder_type': 'simple',
            'hidden_dims': [256, 128],
            'dropout': 0.2,
            'lr': 1e-3,
            'max_epochs': args.epochs,
            'batch_size': 32,
            'patience': 10,
        },
        {
            'name': f'Structured Edit - {args.embedder.upper()}',
            'type': 'edit_framework',
            'embedder_type': args.embedder,
            'edit_embedder_type': 'structured',
            'output_dim': 320,
            'hidden_dims': [256, 128],
            'dropout': 0.2,
            'lr': 1e-3,
            'max_epochs': args.epochs,
            'batch_size': 32,
            'patience': 10,
        },
    ]

    # Run experiment
    run_experiment(
        data_file=args.data_file,
        methods=methods,
        output_dir=args.output_dir,
        experiment_name=f'antibody_{args.embedder}',
        source_datasets=args.source,
        max_samples=args.max_samples,
        random_seed=args.seed,
        generate_html=not args.no_html,
        generate_docx=not args.no_docx,
    )


if __name__ == '__main__':
    main()

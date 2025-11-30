#!/usr/bin/env python3
"""
Main entry point for RNA MPRA experiments.

Usage:
    python experiments/rna/main.py

This runs the full experiment pipeline:
1. Load MPRA pairs data
2. Split into train/val/test
3. Create RNA embedder
4. Train edit effect predictor and baseline
5. Evaluate and generate reports
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gc
from experiment_config import RNAExperimentConfig
from data_loader import load_datasets
from model_factory import create_embedder_from_config, create_models


def run_experiment(config: RNAExperimentConfig):
    """
    Run a complete RNA experiment.

    Args:
        config: RNAExperimentConfig

    Returns:
        Tuple of (results_dict, report_path)
    """
    print(f"\n{'='*80}")
    print(f"RNA EXPERIMENT: {config.experiment_name}")
    print(f"{'='*80}")

    # Step 1: Load data
    print("\n" + "="*40)
    print("STEP 1: Loading Data")
    print("="*40)

    datasets = load_datasets(config)

    # Combine all properties for training
    train_data = {}
    val_data = {}
    test_data = {}

    for prop_name, splits in datasets.items():
        train_data[prop_name] = splits['train']
        val_data[prop_name] = splits['val']
        test_data[prop_name] = splits['test']

    print(f"\nLoaded {len(train_data)} properties")
    total_train = sum(len(df) for df in train_data.values())
    total_val = sum(len(df) for df in val_data.values())
    total_test = sum(len(df) for df in test_data.values())
    print(f"Total pairs - Train: {total_train:,}, Val: {total_val:,}, Test: {total_test:,}")

    # Step 2: Create embedder
    print("\n" + "="*40)
    print("STEP 2: Creating RNA Embedder")
    print("="*40)

    embedder = create_embedder_from_config(config)
    print(f"Embedder: {embedder.name}")
    print(f"Embedding dim: {embedder.embedding_dim}")

    # Step 3: Create models
    print("\n" + "="*40)
    print("STEP 3: Creating Models")
    print("="*40)

    models = create_models(config, train_data, embedder)

    for name, model_info in models.items():
        print(f"  - {name} ({model_info['type']})")

    # Step 4: Pre-compute embeddings
    print("\n" + "="*40)
    print("STEP 4: Pre-computing Embeddings")
    print("="*40)

    from src.data.rna.dataset import EmbeddingCache

    cache = EmbeddingCache(embedder)

    # Get all unique sequences
    all_seqs = set()
    for prop_data in [train_data, val_data, test_data]:
        for df in prop_data.values():
            all_seqs.update(df['seq_a'].tolist())
            all_seqs.update(df['seq_b'].tolist())

    all_seqs = list(all_seqs)
    print(f"Unique sequences to embed: {len(all_seqs):,}")

    embeddings = cache.get_embeddings(all_seqs, show_progress=True)
    print(f"Embedding shape: {embeddings.shape}")

    # Create lookup
    seq_to_emb = {seq: embeddings[i] for i, seq in enumerate(all_seqs)}

    # Step 5: Train models
    print("\n" + "="*40)
    print("STEP 5: Training Models")
    print("="*40)

    results = {}

    for method_name, model_info in models.items():
        print(f"\nTraining: {method_name}")
        print("-" * 40)

        model = model_info['model']
        method_type = model_info['type']

        # Prepare data based on model type
        if method_type == 'edit_framework':
            # Prepare paired data with embeddings
            train_results = _train_edit_model(
                model, train_data, val_data, seq_to_emb, config
            )
        elif method_type == 'structured_edit_framework':
            # Structured edit framework - train end-to-end
            train_results = _train_structured_edit_model(
                model, model_info, train_data, val_data, config
            )
        else:
            # Baseline property prediction
            train_results = _train_baseline_model(
                model, train_data, val_data, seq_to_emb, config
            )

        results[method_name] = train_results

    # Step 6: Evaluate
    print("\n" + "="*40)
    print("STEP 6: Evaluating Models")
    print("="*40)

    for method_name, model_info in models.items():
        print(f"\nEvaluating: {method_name}")

        eval_results = _evaluate_model(
            model_info, test_data, seq_to_emb, config
        )

        results[method_name].update(eval_results)

    # Step 7: Generate report
    print("\n" + "="*40)
    print("STEP 7: Generating Report")
    print("="*40)

    report_path = _generate_report(results, config, test_data=test_data, models=models)

    return results, report_path


def _train_edit_model(model, train_data, val_data, seq_to_emb, config):
    """Train edit effect predictor."""
    import numpy as np

    # Combine all training data
    all_train_a = []
    all_train_b = []
    all_train_delta = []

    for prop_name, df in train_data.items():
        for _, row in df.iterrows():
            all_train_a.append(seq_to_emb[row['seq_a']])
            all_train_b.append(seq_to_emb[row['seq_b']])
            all_train_delta.append(row['delta'])

    # Combine validation data
    all_val_a = []
    all_val_b = []
    all_val_delta = []

    for prop_name, df in val_data.items():
        for _, row in df.iterrows():
            all_val_a.append(seq_to_emb[row['seq_a']])
            all_val_b.append(seq_to_emb[row['seq_b']])
            all_val_delta.append(row['delta'])

    # Convert to arrays
    train_emb_a = np.array(all_train_a)
    train_emb_b = np.array(all_train_b)
    train_delta = np.array(all_train_delta)

    val_emb_a = np.array(all_val_a)
    val_emb_b = np.array(all_val_b)
    val_delta = np.array(all_val_delta)

    print(f"  Training on {len(train_delta):,} pairs")

    # Train
    model.fit(
        mol_emb_a=train_emb_a,
        mol_emb_b=train_emb_b,
        delta_y=train_delta,
        mol_emb_a_val=val_emb_a,
        mol_emb_b_val=val_emb_b,
        delta_y_val=val_delta,
        verbose=True
    )

    return {'train_size': len(train_delta), 'val_size': len(val_delta)}


def _train_structured_edit_model(model, model_info, train_data, val_data, config):
    """Train structured edit effect predictor end-to-end."""
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Collect all training data
    train_seqs = []
    train_positions = []
    train_from = []
    train_to = []
    train_deltas = []

    for prop_name, df in train_data.items():
        for _, row in df.iterrows():
            train_seqs.append(row['seq_a'])
            train_positions.append(row['edit_position'])
            train_from.append(row['edit_from'])
            train_to.append(row['edit_to'])
            train_deltas.append(row['delta'])

    # Collect validation data
    val_seqs = []
    val_positions = []
    val_from = []
    val_to = []
    val_deltas = []

    for prop_name, df in val_data.items():
        for _, row in df.iterrows():
            val_seqs.append(row['seq_a'])
            val_positions.append(row['edit_position'])
            val_from.append(row['edit_from'])
            val_to.append(row['edit_to'])
            val_deltas.append(row['delta'])

    train_deltas = np.array(train_deltas)
    val_deltas = np.array(val_deltas)

    print(f"  Training on {len(train_deltas):,} pairs")
    print(f"  Validation on {len(val_deltas):,} pairs")

    # Get training config
    method_config = model_info['config']
    lr = method_config.get('lr', 0.001)
    batch_size = method_config.get('batch_size', 32)
    max_epochs = method_config.get('max_epochs', 50)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(max_epochs):
        # Shuffle training data
        indices = np.random.permutation(len(train_seqs))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]

            # Get batch data
            batch_seqs = [train_seqs[j] for j in batch_idx]
            batch_pos = [train_positions[j] for j in batch_idx]
            batch_from = [train_from[j] for j in batch_idx]
            batch_to = [train_to[j] for j in batch_idx]
            batch_targets = torch.FloatTensor(train_deltas[batch_idx])

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_seqs, batch_pos, batch_from, batch_to)

            # Handle device
            if predictions.device != batch_targets.device:
                batch_targets = batch_targets.to(predictions.device)

            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_seqs, val_positions, val_from, val_to)
            val_targets = torch.FloatTensor(val_deltas)
            if val_predictions.device != val_targets.device:
                val_targets = val_targets.to(val_predictions.device)
            val_loss = criterion(val_predictions, val_targets).item()
        model.train()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{max_epochs}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"  Best validation loss: {best_val_loss:.4f}")

    return {'train_size': len(train_deltas), 'val_size': len(val_deltas)}


def _train_baseline_model(model, train_data, val_data, seq_to_emb, config):
    """Train baseline property predictor."""
    import numpy as np

    # Combine all unique sequences with their values
    train_seqs = {}
    val_seqs = {}

    for prop_name, df in train_data.items():
        for _, row in df.iterrows():
            train_seqs[row['seq_a']] = row['value_a']
            train_seqs[row['seq_b']] = row['value_b']

    for prop_name, df in val_data.items():
        for _, row in df.iterrows():
            val_seqs[row['seq_a']] = row['value_a']
            val_seqs[row['seq_b']] = row['value_b']

    # Convert to arrays
    train_emb = np.array([seq_to_emb[s] for s in train_seqs.keys()])
    train_y = np.array(list(train_seqs.values()))

    val_emb = np.array([seq_to_emb[s] for s in val_seqs.keys()])
    val_y = np.array(list(val_seqs.values()))

    print(f"  Training on {len(train_y):,} sequences")

    # Train
    model.fit(
        smiles_train=list(train_seqs.keys()),  # Not used if embeddings provided
        y_train=train_y,
        smiles_val=list(val_seqs.keys()),
        y_val=val_y,
        mol_emb_train=train_emb,
        mol_emb_val=val_emb,
        verbose=True
    )

    return {'train_size': len(train_y), 'val_size': len(val_y)}


def _evaluate_model(model_info, test_data, seq_to_emb, config):
    """Evaluate model on test data."""
    import numpy as np
    from scipy import stats

    model = model_info['model']
    method_type = model_info['type']

    all_true = []
    all_pred = []

    for prop_name, df in test_data.items():
        for _, row in df.iterrows():
            true_delta = row['delta']

            if method_type == 'edit_framework':
                # Predict Δ directly
                pred_delta = model.predict(
                    smiles_a=row['seq_a'],
                    smiles_b=row['seq_b'],
                    mol_emb_a=seq_to_emb[row['seq_a']].reshape(1, -1),
                    mol_emb_b=seq_to_emb[row['seq_b']].reshape(1, -1)
                )
            elif method_type == 'structured_edit_framework':
                # Structured edit model - takes sequence + edit info
                import torch
                model.eval()
                with torch.no_grad():
                    pred_delta = model(
                        row['seq_a'],
                        row['edit_position'],
                        row['edit_from'],
                        row['edit_to']
                    )
                    if hasattr(pred_delta, 'cpu'):
                        pred_delta = pred_delta.cpu().numpy()
                    if hasattr(pred_delta, '__len__') and len(pred_delta) == 1:
                        pred_delta = pred_delta[0]
            else:
                # Predict values and compute Δ
                pred_a = model.predict(
                    smiles=row['seq_a'],
                    mol_emb=seq_to_emb[row['seq_a']].reshape(1, -1)
                )
                pred_b = model.predict(
                    smiles=row['seq_b'],
                    mol_emb=seq_to_emb[row['seq_b']].reshape(1, -1)
                )
                pred_delta = float(pred_b) - float(pred_a)

            all_true.append(true_delta)
            all_pred.append(float(pred_delta) if hasattr(pred_delta, '__float__') else pred_delta[0] if hasattr(pred_delta, '__getitem__') else pred_delta)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Compute metrics
    mae = np.mean(np.abs(all_true - all_pred))
    rmse = np.sqrt(np.mean((all_true - all_pred) ** 2))
    r2 = 1 - np.sum((all_true - all_pred) ** 2) / np.sum((all_true - np.mean(all_true)) ** 2)
    pearson_r = stats.pearsonr(all_true, all_pred)[0]
    spearman_r = stats.spearmanr(all_true, all_pred)[0]

    # Direction accuracy
    direction_correct = np.sum(np.sign(all_true) == np.sign(all_pred))
    direction_accuracy = direction_correct / len(all_true)

    results = {
        'test_size': len(all_true),
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
        'direction_accuracy': direction_accuracy,
        'predictions': (all_true, all_pred)  # Store for scatter plots
    }

    print(f"  Test MAE: {mae:.4f}")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  Test R²: {r2:.4f}")
    print(f"  Test Pearson r: {pearson_r:.4f}")
    print(f"  Direction accuracy: {direction_accuracy:.2%}")

    return results


def _generate_report(results, config, test_data=None, models=None):
    """Generate experiment report with visualizations."""
    from pathlib import Path
    import json
    import pandas as pd

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract predictions for scatter plots
    predictions = {}
    for method, metrics in results.items():
        if 'predictions' in metrics:
            predictions[method] = metrics['predictions']

    # Create serializable results (without predictions array)
    serializable_results = {}
    for k, v in results.items():
        serializable_results[k] = {}
        for k2, v2 in v.items():
            if k2 != 'predictions':
                if hasattr(v2, 'item'):
                    serializable_results[k][k2] = v2.item()
                else:
                    serializable_results[k][k2] = v2

    # Save results as JSON
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to: {results_file}")

    # Combine test data into single DataFrame
    if test_data is not None:
        all_test = []
        for prop_name, df in test_data.items():
            all_test.append(df)
        test_df = pd.concat(all_test, ignore_index=True) if all_test else None
    else:
        # Try to load from file
        data_path = Path(config.output_dir).parent.parent.parent / config.data_file
        if data_path.exists():
            test_df = pd.read_csv(data_path)
        else:
            test_df = None

    # Use the full report generator
    try:
        from report_generator import generate_rna_report
        report_path = generate_rna_report(
            results=serializable_results,
            config=config,
            data=test_df,
            predictions=predictions,
            trained_models=models
        )
        print(f"Full report saved to: {report_path}")
        return report_path
    except Exception as e:
        print(f"Warning: Could not generate full report: {e}")
        print("Falling back to simple summary...")

    # Fallback: Generate simple markdown summary
    summary_file = output_dir / 'summary.md'

    with open(summary_file, 'w') as f:
        f.write(f"# RNA Experiment: {config.experiment_name}\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- Embedder: {config.embedder_type}\n")
        f.write(f"- Splitter: {config.splitter_type}\n")
        f.write(f"- Data file: {config.data_file}\n\n")

        f.write(f"## Results\n\n")
        f.write("| Method | MAE | RMSE | R² | Pearson r | Direction Acc |\n")
        f.write("|--------|-----|------|----|-----------|--------------|\n")

        for method, metrics in serializable_results.items():
            f.write(f"| {method} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | ")
            f.write(f"{metrics['r2']:.4f} | {metrics['pearson_r']:.4f} | ")
            f.write(f"{metrics['direction_accuracy']:.2%} |\n")

    print(f"Summary saved to: {summary_file}")

    return summary_file


def main():
    """
    Main entry point for RNA MPRA experiments.

    Methods are configured inline below - edit the 'methods' list to add/remove
    methods or change their hyperparameters.
    """
    # Splitters to run
    splitters = ['random']  # Can add: 'sequence_similarity', 'edit_type'

    for splitter_type in splitters:
        print(f"\n{'='*80}")
        print(f"Running experiment with {splitter_type.upper()} split")
        print(f"{'='*80}\n")

        # Configure splitter-specific parameters
        splitter_params = {}
        if splitter_type == 'sequence_similarity':
            splitter_params[splitter_type] = {'similarity_threshold': 0.8}
        elif splitter_type == 'edit_type':
            splitter_params[splitter_type] = {'edit_col': 'edit_type'}

        config = RNAExperimentConfig(
            experiment_name=f"rna_mpra_edit_prediction_{splitter_type}",

            # Data file - update this path for your dataset
            data_file="../../data/rna/pairs/mpra_5utr_pairs_long.csv",

            # Splitting
            splitter_type=splitter_type,
            splitter_params=splitter_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,

            # Number of tasks (properties to predict)
            num_tasks=1,
            min_pairs_per_property=0,

            # Methods to run - configure which methods and their hyperparameters
            methods=[
                # Method 1: Structured Edit Framework (uses RNA-FM internally)
                # This is the richest representation of edits
                {
                    'name': 'Structured Edit Framework',
                    'type': 'structured_edit_framework',
                    # RNA-FM settings (used internally)
                    'rnafm_model': 'rna_fm_t12',
                    'rnafm_pooling': 'mean',
                    'device': 'auto',
                    # StructuredRNAEditEmbedder dimensions
                    'mutation_type_dim': 64,
                    'mutation_effect_dim': 256,
                    'position_dim': 64,
                    'local_context_dim': 256,
                    'attention_context_dim': 128,
                    'fusion_hidden_dims': [512, 384],
                    'output_dim': 256,
                    'window_size': 10,
                    'embedder_dropout': 0.1,
                    # Prediction head
                    'head_hidden_dims': [128, 64],
                    'dropout': 0.1,
                    # Training
                    'lr': 0.001,
                    'batch_size': 32,
                    'max_epochs': 50
                },
                # Method 2: Edit Framework with RNA-FM embeddings
                # Uses pre-computed embeddings for sequence pairs
                {
                    'name': 'Edit Framework - RNA-FM',
                    'type': 'edit_framework',
                    'use_edit_fragments': False,
                    'hidden_dims': [512, 256, 128],
                    'head_hidden_dims': [128, 64],
                    'dropout': 0.1,
                    'lr': 0.001,
                    'batch_size': 32,
                    'max_epochs': 50
                },
                # Method 3: Baseline Property Predictor
                # Predicts property from single sequence, computes Δ at test time
                {
                    'name': 'Baseline Property Predictor',
                    'type': 'baseline_property',
                    'hidden_dims': [512, 256, 128],
                    'head_hidden_dims': [128, 64],
                    'dropout': 0.1,
                    'lr': 0.001,
                    'batch_size': 32,
                    'max_epochs': 50
                }
            ],

            # Base embedder for edit_framework and baseline methods
            # (structured_edit_framework uses RNA-FM internally)
            embedder_type='rnafm',
            trainable_embedder=False,
            embedder_device='auto',
            rnafm_model='rna_fm_t12',
            rnafm_pooling='mean',

            # Evaluation metrics
            metrics=['mae', 'rmse', 'r2', 'pearson_r', 'spearman_r', 'direction_accuracy'],

            # Output
            output_dir=f'experiments/rna/results/{splitter_type}',

            # Analysis
            include_motif_analysis=True,
            include_edit_embedding_comparison=True
        )

        results, report_path = run_experiment(config)

        print(f"\n{'='*80}")
        print(f"Experiment '{config.experiment_name}' completed successfully!")
        print(f"Report saved to: {report_path}")
        print(f"{'='*80}\n")

        # Free memory after each splitter
        del results, report_path, config
        gc.collect()
        print(f"Memory freed after {splitter_type} splitter\n")

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print("\nResults directories:")
    for splitter_type in splitters:
        print(f"  - experiments/rna/results/{splitter_type}/")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

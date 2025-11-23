"""
Helper functions for model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_multi_task_model(
    model,
    test_data,
    task_names: List[str],
    model_name: str = "Model",
    mol_emb_a: Optional[np.ndarray] = None,
    mol_emb_b: Optional[np.ndarray] = None,
    mol_emb: Optional[np.ndarray] = None,
    edit_frag_a: Optional[np.ndarray] = None,
    edit_frag_b: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate multi-task model on test data.

    Args:
        model: Trained model (EditEffectPredictor or PropertyPredictor)
        test_data: Test dataset (pandas DataFrame)
        task_names: List of task names
        model_name: Name for display
        mol_emb_a: Pre-computed embeddings for mol_a (for EditEffectPredictor)
        mol_emb_b: Pre-computed embeddings for mol_b (for EditEffectPredictor)
        mol_emb: Pre-computed embeddings for molecules (for PropertyPredictor)
        edit_frag_a: Pre-computed edit fragment A embeddings (for Mode 2)
        edit_frag_b: Pre-computed edit fragment B embeddings (for Mode 2)

    Returns:
        Tuple of (metrics_df, predictions_dict)
        - metrics_df: DataFrame with MSE, MAE, R² per task
        - predictions_dict: Dict with {task_name: (y_true, y_pred)}
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name}")
    print(f"{'='*70}")

    # Check if this is EditEffectPredictor (needs mol_a and mol_b) or PropertyPredictor (needs smiles only)
    is_edit_predictor = hasattr(model, 'edit_embedder') or hasattr(model, 'trainable_edit_embeddings')

    if is_edit_predictor:
        # EditEffectPredictor: predict with pre-computed embeddings or SMILES
        if mol_emb_a is not None and mol_emb_b is not None:
            print("Using pre-computed embeddings for evaluation")
            # Check if Mode 2 (edit fragments) is being used
            if edit_frag_a is not None and edit_frag_b is not None:
                print("  Mode 2: Using edit fragment embeddings")
                predictions = model.predict(
                    smiles_a=test_data['mol_a'].values,
                    smiles_b=test_data['mol_b'].values,
                    mol_emb_a=mol_emb_a,
                    mol_emb_b=mol_emb_b,
                    edit_frag_a_emb=edit_frag_a,
                    edit_frag_b_emb=edit_frag_b
                )
            else:
                predictions = model.predict(
                    smiles_a=test_data['mol_a'].values,
                    smiles_b=test_data['mol_b'].values,
                    mol_emb_a=mol_emb_a,
                    mol_emb_b=mol_emb_b
                )
        else:
            smiles_a = test_data['mol_a'].values
            smiles_b = test_data['mol_b'].values
            predictions = model.predict(smiles_a, smiles_b)

        # Extract true labels - build multi-task array from property_idx
        n_samples = len(test_data)
        n_tasks = len(task_names)
        y_test = np.full((n_samples, n_tasks), np.nan)

        for idx, row in test_data.iterrows():
            prop_idx = int(row['property_idx'])
            y_test[idx, prop_idx] = row['delta']
    else:
        # PropertyPredictor: predict with pre-computed embeddings or SMILES
        if mol_emb is not None:
            print("Using pre-computed embeddings for evaluation")
            predictions = model.predict(
                smiles=test_data['smiles'].values,
                mol_emb=mol_emb
            )
        else:
            smiles = test_data['smiles'].values
            predictions = model.predict(smiles)

        # Extract true labels
        n_samples = len(test_data)
        n_tasks = len(task_names)
        y_test = np.full((n_samples, n_tasks), np.nan)

        for idx, row in test_data.iterrows():
            prop_idx = int(row['property_idx'])
            y_test[idx, prop_idx] = row['property_value']

    # Convert predictions dict to array if needed
    if isinstance(predictions, dict):
        # Multi-task model returns {task_name: predictions}
        pred_array = np.column_stack([predictions[task] for task in task_names])
    else:
        # Single task or already array
        pred_array = predictions
        if pred_array.ndim == 1:
            pred_array = pred_array.reshape(-1, 1)

    # Calculate metrics per task
    metrics = []
    predictions_dict = {}

    for i, task_name in enumerate(task_names):
        y_true = y_test[:, i]
        y_pred = pred_array[:, i]

        # Filter out NaN values (unmeasured labels)
        mask = ~np.isnan(y_true)
        y_true_valid = y_true[mask]
        y_pred_valid = y_pred[mask]

        if len(y_true_valid) == 0:
            print(f"  ⚠️  {task_name}: No valid test samples")
            continue

        # Calculate metrics
        mse = mean_squared_error(y_true_valid, y_pred_valid)
        mae = mean_absolute_error(y_true_valid, y_pred_valid)
        r2 = r2_score(y_true_valid, y_pred_valid)

        metrics.append({
            'Task': task_name[:40],  # Truncate long names
            'N': len(y_true_valid),
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'R²': r2
        })

        predictions_dict[task_name] = (y_true_valid, y_pred_valid)

        # Print metrics
        print(f"\n  {task_name[:40]}")
        print(f"    N:    {len(y_true_valid):>6}")
        print(f"    MSE:  {mse:>6.4f}")
        print(f"    RMSE: {np.sqrt(mse):>6.4f}")
        print(f"    MAE:  {mae:>6.4f}")
        print(f"    R²:   {r2:>6.4f}")

    metrics_df = pd.DataFrame(metrics)

    # Overall metrics (macro average)
    print(f"\n{'='*70}")
    print(f"Overall (Macro Average)")
    print(f"{'='*70}")
    print(f"  MSE:  {metrics_df['MSE'].mean():.4f}")
    print(f"  RMSE: {metrics_df['RMSE'].mean():.4f}")
    print(f"  MAE:  {metrics_df['MAE'].mean():.4f}")
    print(f"  R²:   {metrics_df['R²'].mean():.4f}")
    print(f"{'='*70}\n")

    return metrics_df, predictions_dict


def plot_predictions_scatter(
    predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_name: str = "Model",
    figsize: Tuple[int, int] = (18, 10)
) -> plt.Figure:
    """
    Plot scatter plots of predictions vs true values for each task.

    Args:
        predictions_dict: Dict with {task_name: (y_true, y_pred)}
        model_name: Name for plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    num_tasks = len(predictions_dict)
    ncols = 3
    nrows = (num_tasks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for i, (task_name, (y_true, y_pred)) in enumerate(predictions_dict.items()):
        ax = axes[i]

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='none')

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val],
               'r--', linewidth=2, alpha=0.7, label='Perfect')

        # Metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        ax.set_xlabel('True Δ Value', fontsize=10)
        ax.set_ylabel('Predicted Δ Value', fontsize=10)

        # Truncate long task names
        task_display = task_name[:30] + '...' if len(task_name) > 30 else task_name
        ax.set_title(f"{task_display}\nR²={r2:.3f}, MAE={mae:.3f}",
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for i in range(num_tasks, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{model_name}: Predictions vs True Values',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    return fig


def plot_residuals(
    predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_name: str = "Model",
    figsize: Tuple[int, int] = (18, 10)
) -> plt.Figure:
    """
    Plot residuals (prediction errors) for each task.

    Args:
        predictions_dict: Dict with {task_name: (y_true, y_pred)}
        model_name: Name for plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    num_tasks = len(predictions_dict)
    ncols = 3
    nrows = (num_tasks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for i, (task_name, (y_true, y_pred)) in enumerate(predictions_dict.items()):
        ax = axes[i]

        residuals = y_pred - y_true

        # Residual plot
        ax.scatter(y_true, residuals, alpha=0.5, s=20, edgecolors='none')
        ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.7)

        # Stats
        mean_res = residuals.mean()
        std_res = residuals.std()

        ax.set_xlabel('True Δ Value', fontsize=10)
        ax.set_ylabel('Residual (Pred - True)', fontsize=10)

        # Truncate long task names
        task_display = task_name[:30] + '...' if len(task_name) > 30 else task_name
        ax.set_title(f"{task_display}\nMean={mean_res:.3f}, Std={std_res:.3f}",
                    fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for i in range(num_tasks, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'{model_name}: Residual Analysis',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    return fig


def compare_models(
    metrics_1: pd.DataFrame,
    metrics_2: pd.DataFrame,
    name_1: str = "Model 1",
    name_2: str = "Model 2",
    metric: str = 'MSE',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Compare two models side-by-side.

    Args:
        metrics_1: Metrics DataFrame for model 1
        metrics_2: Metrics DataFrame for model 2
        name_1: Name of model 1
        name_2: Name of model 2
        metric: Metric to compare ('MSE', 'MAE', 'R²')
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    tasks = metrics_1['Task'].values
    x = np.arange(len(tasks))
    width = 0.35

    values_1 = metrics_1[metric].values
    values_2 = metrics_2[metric].values

    # Bar plots
    bars1 = ax.bar(x - width/2, values_1, width, label=name_1, alpha=0.8)
    bars2 = ax.bar(x + width/2, values_2, width, label=name_2, alpha=0.8)

    # Highlight better model
    if metric in ['MSE', 'MAE', 'RMSE']:
        # Lower is better
        for i in range(len(x)):
            if values_1[i] < values_2[i]:
                #bars1[i].set_edgecolor('green')
                bars1[i].set_linewidth(3)
            else:
                #bars2[i].set_edgecolor('green')
                bars2[i].set_linewidth(3)
    else:
        # Higher is better (R²)
        for i in range(len(x)):
            if values_1[i] > values_2[i]:
                #bars1[i].set_edgecolor('green')
                bars1[i].set_linewidth(3)
            else:
                #bars2[i].set_edgecolor('green')
                bars2[i].set_linewidth(3)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t[:20] for t in tasks], rotation=45, ha='right')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Add mean line
    mean_1 = values_1.mean()
    mean_2 = values_2.mean()
    ax.axhline(mean_1, color='C0', linestyle=':', alpha=0.5,
              label=f'{name_1} Mean: {mean_1:.3f}')
    ax.axhline(mean_2, color='C1', linestyle=':', alpha=0.5,
              label=f'{name_2} Mean: {mean_2:.3f}')

    plt.tight_layout()
    return fig

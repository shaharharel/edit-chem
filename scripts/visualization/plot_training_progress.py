"""
Helper function to plot training progress per epoch instead of per step.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def aggregate_history_by_epoch(history_df, steps_per_epoch):
    """
    Aggregate training history from steps to epochs.

    Args:
        history_df: DataFrame with 'step' column and loss columns
        steps_per_epoch: Number of training steps per epoch

    Returns:
        DataFrame with epoch-level aggregated metrics
    """
    # Calculate epoch for each step
    history_df = history_df.copy()
    history_df['epoch'] = history_df['step'] // steps_per_epoch

    # Group by epoch and take mean
    epoch_metrics = history_df.groupby('epoch').mean()

    return epoch_metrics


def plot_training_progress_per_epoch(model, task_names, steps_per_epoch=None):
    """
    Plot training progress per epoch for a trained model.

    Args:
        model: Trained model with get_training_history() method
        task_names: List of task names
        steps_per_epoch: Steps per epoch (auto-calculated if None)
    """
    try:
        history = model.get_training_history()

        # Auto-calculate steps per epoch if not provided
        if steps_per_epoch is None:
            # Find the step where epoch changes (first validation)
            val_rows = history.dropna(subset=['val_loss'])
            if len(val_rows) > 1:
                steps_per_epoch = val_rows.iloc[1]['step'] - val_rows.iloc[0]['step']
            else:
                steps_per_epoch = len(history.dropna(subset=['train_loss']))

        # Aggregate by epoch
        history['epoch'] = history['step'] // steps_per_epoch

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Overall loss
        ax = axes[0]
        if 'train_loss' in history.columns and 'val_loss' in history.columns:
            train_by_epoch = history.groupby('epoch')['train_loss'].mean()
            val_by_epoch = history.groupby('epoch')['val_loss'].mean()

            ax.plot(train_by_epoch.index, train_by_epoch.values,
                   label='Train', linewidth=2, alpha=0.8, marker='o')
            ax.plot(val_by_epoch.index, val_by_epoch.values,
                   label='Val', linewidth=2, alpha=0.8, marker='s')
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss (MSE)', fontsize=11)
            ax.set_title('Overall Loss', fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)

        # Per-task losses
        for i, task_name in enumerate(task_names, 1):
            if i >= len(axes):
                break

            ax = axes[i]

            train_col = f'train_loss_{task_name}'
            val_col = f'val_loss_{task_name}'

            if train_col in history.columns and val_col in history.columns:
                train_by_epoch = history.groupby('epoch')[train_col].mean()
                val_by_epoch = history.groupby('epoch')[val_col].mean()

                if len(train_by_epoch) > 0:
                    ax.plot(train_by_epoch.index, train_by_epoch.values,
                           label='Train', linewidth=2, alpha=0.8, color='#1f77b4', marker='o')
                if len(val_by_epoch) > 0:
                    ax.plot(val_by_epoch.index, val_by_epoch.values,
                           label='Val', linewidth=2, alpha=0.8, color='#ff7f0e', marker='s')

                # Truncate long task names
                task_display = task_name[:30] + '...' if len(task_name) > 30 else task_name
                ax.set_xlabel('Epoch', fontsize=11)
                ax.set_ylabel('Loss (MSE)', fontsize=11)
                ax.set_title(task_display, fontweight='bold', fontsize=11)
                ax.legend()
                ax.grid(alpha=0.3)

        # Hide unused subplots
        for i in range(len(task_names) + 1, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        n_epochs = len(history.groupby('epoch'))
        return fig, n_epochs

    except Exception as e:
        print(f"Could not visualize training history: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

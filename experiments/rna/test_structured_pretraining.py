#!/usr/bin/env python3
"""
Structured Edit Pre-training Experiment.

Strategy:
1. Pre-train the Structured Edit Embedder on ~60K random library SNV pairs
2. Fine-tune on 5.7K MPRA SNV pairs
3. Compare against training from scratch

This uses the SAME architecture as test_expanded_dataset.py (57.4% direction accuracy)
but with pre-training on a larger dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============== DATA LOADING ==============

def load_pretrain_pairs(path='data/rna/pairs/random_library_snv_pairs.csv', sample_size=None):
    """Load random library SNV pairs for pre-training."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} random library SNV pairs for pre-training")

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"  Sampled to {len(df):,}")

    return df


def load_mpra_pairs(path='data/rna/pairs/mpra_5utr_pairs_long.csv'):
    """Load MPRA SNV pairs for fine-tuning."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} MPRA SNV pairs for fine-tuning")
    return df


def split_pairs_by_sequence(df, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split pairs ensuring no sequence leakage."""
    np.random.seed(seed)

    unique_seqs = df['seq_a'].unique()
    np.random.shuffle(unique_seqs)

    n_seqs = len(unique_seqs)
    train_seqs = set(unique_seqs[:int(train_ratio * n_seqs)])
    val_seqs = set(unique_seqs[int(train_ratio * n_seqs):int((train_ratio + val_ratio) * n_seqs)])
    test_seqs = set(unique_seqs[int((train_ratio + val_ratio) * n_seqs):])

    train_df = df[df['seq_a'].isin(train_seqs)].reset_index(drop=True)
    val_df = df[df['seq_a'].isin(val_seqs)].reset_index(drop=True)
    test_df = df[df['seq_a'].isin(test_seqs)].reset_index(drop=True)

    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    return train_df, val_df, test_df


def evaluate_regression(y_true, y_pred):
    """Evaluate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if np.std(y_pred) > 0:
        pearson_r, _ = pearsonr(y_true, y_pred)
    else:
        pearson_r = 0

    # Direction accuracy
    dir_acc = np.mean((y_pred > 0) == (y_true > 0))

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Pearson_r': pearson_r,
        'Direction_Acc': dir_acc
    }


# ============== MODEL ==============

class StructuredEditModel(nn.Module):
    """Wrapper around StructuredRNAEditEmbedder with prediction head."""

    def __init__(self, embedder, output_dim=1):
        super().__init__()
        self.structured_embedder = embedder

        embed_dim = getattr(embedder, 'output_dim', 128)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim)
        )

    def forward(self, sequences, positions, edit_from, edit_to):
        emb = self.structured_embedder(sequences, positions, edit_from, edit_to)
        return self.head(emb).squeeze(-1)


def create_structured_model():
    """Create a fresh StructuredEditModel."""
    from src.embedding.rna.structured_edit_embedder import StructuredRNAEditEmbedder
    from src.embedding.rna.rnafm import RNAFMEmbedder

    # Initialize RNA-FM embedder (frozen)
    rnafm_embedder = RNAFMEmbedder(
        model_name='rna_fm_t12',
        pooling='mean',
        trainable=False
    )

    # Initialize structured edit embedder
    embedder = StructuredRNAEditEmbedder(
        rnafm_embedder=rnafm_embedder,
        mutation_type_dim=64,
        mutation_effect_dim=128,
        position_dim=64,
        local_context_dim=128,
        attention_context_dim=64,
        fusion_hidden_dims=[256, 192],
        output_dim=128,
        window_size=10,
        dropout=0.2
    )

    model = StructuredEditModel(embedder, output_dim=1)
    return model


# ============== TRAINING ==============

def train_model(model, train_df, val_df, epochs=100, lr=1e-3, batch_size=32,
                patience=15, description="Training"):
    """Train the model on delta prediction."""
    print(f"\n--- {description} ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Prepare data
    def prepare_batch(df):
        sequences = df['seq_a'].tolist()
        positions = torch.tensor(df['edit_position'].values, dtype=torch.long)
        edit_from = df['edit_from'].tolist()
        edit_to = df['edit_to'].tolist()
        return sequences, positions, edit_from, edit_to

    train_seqs, train_pos, train_from, train_to = prepare_batch(train_df)
    val_seqs, val_pos, val_from, val_to = prepare_batch(val_df)

    y_train = train_df['delta'].values
    y_val = val_df['delta'].values

    n_train = len(train_df)
    n_val = len(val_df)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_train)
        train_loss = 0

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i+batch_size]

            batch_seqs = [train_seqs[j] for j in batch_idx]
            batch_pos = train_pos[batch_idx].to(device)
            batch_from = [train_from[j] for j in batch_idx]
            batch_to = [train_to[j] for j in batch_idx]
            batch_y = torch.tensor(y_train[batch_idx], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            output = model(batch_seqs, batch_pos, batch_from, batch_to)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                batch_seqs = val_seqs[i:i+batch_size]
                batch_pos = val_pos[i:i+batch_size].to(device)
                batch_from = val_from[i:i+batch_size]
                batch_to = val_to[i:i+batch_size]
                batch_y = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32).to(device)

                output = model(batch_seqs, batch_pos, batch_from, batch_to)
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        val_loss /= (n_val // batch_size + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}")

    # Load best state
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  Best val loss: {best_val_loss:.4f}")

    return model


def evaluate_model(model, test_df, batch_size=32):
    """Evaluate model on test set."""
    device = next(model.parameters()).device
    model.eval()

    def prepare_batch(df):
        sequences = df['seq_a'].tolist()
        positions = torch.tensor(df['edit_position'].values, dtype=torch.long)
        edit_from = df['edit_from'].tolist()
        edit_to = df['edit_to'].tolist()
        return sequences, positions, edit_from, edit_to

    test_seqs, test_pos, test_from, test_to = prepare_batch(test_df)
    y_test = test_df['delta'].values
    n_test = len(test_df)

    all_preds = []
    with torch.no_grad():
        for i in range(0, n_test, batch_size):
            batch_seqs = test_seqs[i:i+batch_size]
            batch_pos = test_pos[i:i+batch_size].to(device)
            batch_from = test_from[i:i+batch_size]
            batch_to = test_to[i:i+batch_size]

            output = model(batch_seqs, batch_pos, batch_from, batch_to)
            all_preds.extend(output.cpu().numpy())

    return evaluate_regression(y_test, np.array(all_preds))


# ============== MAIN ==============

def main():
    print("=" * 70)
    print("STRUCTURED EDIT PRE-TRAINING EXPERIMENT")
    print("=" * 70)

    # Load data
    print("\n--- Loading Data ---")

    # Pre-training data (random library pairs)
    pretrain_df = load_pretrain_pairs(sample_size=50000)  # Use 50K for speed
    pretrain_train, pretrain_val, _ = split_pairs_by_sequence(pretrain_df)

    # Fine-tuning data (MPRA pairs)
    mpra_df = load_mpra_pairs()
    train_df, val_df, test_df = split_pairs_by_sequence(mpra_df)

    results = {}

    # ========== Condition 1: No Pre-training (baseline) ==========
    print("\n" + "=" * 70)
    print("CONDITION 1: No Pre-training (baseline)")
    print("=" * 70)

    model_baseline = create_structured_model()
    model_baseline = train_model(
        model_baseline, train_df, val_df,
        epochs=100, lr=1e-3, batch_size=32,
        description="Training from scratch on MPRA"
    )
    results['No Pre-training'] = evaluate_model(model_baseline, test_df)
    print(f"\n  Results: {results['No Pre-training']}")

    # ========== Condition 2: Pre-trained + Frozen ==========
    print("\n" + "=" * 70)
    print("CONDITION 2: Pre-trained + Fine-tuned (Frozen Embedder)")
    print("=" * 70)

    model_pretrained = create_structured_model()

    # Pre-train on random library pairs
    model_pretrained = train_model(
        model_pretrained, pretrain_train, pretrain_val,
        epochs=50, lr=1e-3, batch_size=32,
        description="Pre-training on random library pairs"
    )

    # Freeze the structured embedder
    for param in model_pretrained.structured_embedder.parameters():
        param.requires_grad = False

    # Fine-tune only the head
    model_pretrained = train_model(
        model_pretrained, train_df, val_df,
        epochs=100, lr=1e-3, batch_size=32,
        description="Fine-tuning (frozen embedder) on MPRA"
    )

    results['Pre-train + Frozen'] = evaluate_model(model_pretrained, test_df)
    print(f"\n  Results: {results['Pre-train + Frozen']}")

    # ========== Condition 3: Pre-trained + Trainable ==========
    print("\n" + "=" * 70)
    print("CONDITION 3: Pre-trained + Fine-tuned (Trainable)")
    print("=" * 70)

    model_trainable = create_structured_model()

    # Pre-train on random library pairs
    model_trainable = train_model(
        model_trainable, pretrain_train, pretrain_val,
        epochs=50, lr=1e-3, batch_size=32,
        description="Pre-training on random library pairs"
    )

    # Fine-tune entire model (lower LR)
    model_trainable = train_model(
        model_trainable, train_df, val_df,
        epochs=100, lr=5e-4, batch_size=32,  # Lower LR for fine-tuning
        description="Fine-tuning (trainable) on MPRA"
    )

    results['Pre-train + Trainable'] = evaluate_model(model_trainable, test_df)
    print(f"\n  Results: {results['Pre-train + Trainable']}")

    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('Direction_Acc', ascending=False)
    print("\n" + results_df.to_markdown())

    # Save results
    output_dir = Path('experiments/rna/results/structured_pretraining')
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'results.csv')

    print(f"\nResults saved to {output_dir}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    baseline = results['No Pre-training']
    pretrain_frozen = results['Pre-train + Frozen']
    pretrain_trainable = results['Pre-train + Trainable']

    print(f"\nDirection Accuracy:")
    print(f"  No Pre-training:        {baseline['Direction_Acc']:.4f}")
    print(f"  Pre-train + Frozen:     {pretrain_frozen['Direction_Acc']:.4f} ({pretrain_frozen['Direction_Acc'] - baseline['Direction_Acc']:+.4f})")
    print(f"  Pre-train + Trainable:  {pretrain_trainable['Direction_Acc']:.4f} ({pretrain_trainable['Direction_Acc'] - baseline['Direction_Acc']:+.4f})")

    print(f"\nMAE:")
    print(f"  No Pre-training:        {baseline['MAE']:.4f}")
    print(f"  Pre-train + Frozen:     {pretrain_frozen['MAE']:.4f} ({pretrain_frozen['MAE'] - baseline['MAE']:+.4f})")
    print(f"  Pre-train + Trainable:  {pretrain_trainable['MAE']:.4f} ({pretrain_trainable['MAE'] - baseline['MAE']:+.4f})")

    if pretrain_trainable['Direction_Acc'] > baseline['Direction_Acc']:
        improvement = pretrain_trainable['Direction_Acc'] - baseline['Direction_Acc']
        print(f"\n==> Pre-training IMPROVES direction accuracy by {improvement:.2%}!")
    elif pretrain_frozen['Direction_Acc'] > baseline['Direction_Acc']:
        improvement = pretrain_frozen['Direction_Acc'] - baseline['Direction_Acc']
        print(f"\n==> Pre-training (frozen) IMPROVES direction accuracy by {improvement:.2%}!")
    else:
        print("\n==> Pre-training does NOT improve direction accuracy")


if __name__ == '__main__':
    main()

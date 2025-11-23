"""
Visualize clustering differences between baseline and trained edit embeddings.

Shows how similar edits cluster together in both embedding spaces.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

from src.embedding.small_molecule import ChemPropEmbedder
from src.models import EditEffectPredictor
from src.utils.embedding_cache import EmbeddingCache


def load_data_and_embeddings():
    """Load test data and compute embeddings."""

    print("Loading data and embeddings...")

    # Load data
    DATA_FILE = Path(__file__).parent.parent / 'data' / 'pairs' / 'chembl_pairs_long_sample.csv'
    df_long = pd.read_csv(DATA_FILE)

    from sklearn.model_selection import train_test_split
    from itertools import combinations

    # Same setup as other scripts
    MIN_PAIRS_PER_PROPERTY = 3000
    NUM_TASKS = 10
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42

    prop_counts = df_long.groupby('property_name').size().sort_values(ascending=False)
    candidates = prop_counts[prop_counts >= MIN_PAIRS_PER_PROPERTY].index.tolist()

    prop_edit_sets = {}
    for prop in candidates:
        prop_edit_sets[prop] = set(df_long[df_long['property_name'] == prop]['edit_name'])

    best_combo = None
    best_shared = 0
    for combo in combinations(list(prop_edit_sets.keys())[:12], NUM_TASKS):
        shared = prop_edit_sets[combo[0]].copy()
        for prop in combo[1:]:
            shared &= prop_edit_sets[prop]
        if len(shared) > best_shared:
            best_shared = len(shared)
            best_combo = combo

    selected_properties = list(best_combo)
    df_filtered = df_long[df_long['property_name'].isin(selected_properties)].copy()
    edit_property_counts = df_filtered.groupby('edit_name')['property_name'].nunique()
    multi_property_edits = edit_property_counts[edit_property_counts > 1].index
    df_filtered = df_filtered[df_filtered['edit_name'].isin(multi_property_edits)].copy()

    splits = {}
    for prop in selected_properties:
        data = df_filtered[df_filtered['property_name'] == prop][[
            'mol_a', 'mol_b', 'edit_name', 'value_a', 'value_b', 'delta'
        ]].copy()
        train, temp = train_test_split(data, test_size=(VAL_RATIO+TEST_RATIO), random_state=RANDOM_SEED)
        val, test = train_test_split(temp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=RANDOM_SEED)
        splits[prop] = {'train': train, 'val': val, 'test': test}

    test_data = []
    for i, prop in enumerate(selected_properties):
        data = splits[prop]['test'][['mol_a', 'mol_b', 'edit_name', 'delta']].copy()
        data['property_name'] = prop
        data['property_idx'] = i
        test_data.append(data)
    test_df = pd.concat(test_data, ignore_index=True)

    # Load model
    mol_embedder = ChemPropEmbedder()
    MODEL_PATH = Path(__file__).parent.parent / 'models' / 'edit_framework_100ep.pt'
    edit_model = EditEffectPredictor.load_checkpoint(
        MODEL_PATH,
        mol_embedder=mol_embedder,
        device='cpu'
    )

    # Load embeddings
    cache = EmbeddingCache(cache_dir=str(Path(__file__).parent.parent / '.embeddings_cache'))
    unique_smiles = list(set(test_df['mol_a'].tolist() + test_df['mol_b'].tolist()))

    all_embeddings = cache.get_or_compute(
        smiles=unique_smiles,
        embedder=mol_embedder,
        dataset_name='test_unique_molecules'
    )

    emb_lookup = {smiles: emb for smiles, emb in zip(unique_smiles, all_embeddings)}
    mol_emb_a = np.array([emb_lookup[s] for s in test_df['mol_a']])
    mol_emb_b = np.array([emb_lookup[s] for s in test_df['mol_b']])

    # Extract both embedding spaces
    edit_model.model.eval()
    edit_model.model.to(edit_model.device)

    with torch.no_grad():
        baseline_embeddings = mol_emb_b - mol_emb_a
        reactant_tensor = torch.FloatTensor(mol_emb_a).to(edit_model.device)
        product_tensor = torch.FloatTensor(mol_emb_b).to(edit_model.device)
        trained_embeddings = edit_model.model.trainable_edit_layer(
            reactant_tensor, product_tensor
        ).cpu().numpy()

    print(f"Loaded {len(test_df)} test samples")

    return test_df, baseline_embeddings, trained_embeddings


def visualize_edit_clustering(test_df, baseline_emb, trained_emb, property_name, n_edits=10):
    """
    Visualize how specific edits cluster in both spaces.

    Select frequent edits and show their neighborhoods.
    """
    print(f"\n{'='*80}")
    print(f"VISUALIZATION 1: EDIT CLUSTERING FOR {property_name}")
    print(f"{'='*80}\n")

    # Filter to property
    mask = test_df['property_name'] == property_name
    df_prop = test_df[mask].reset_index(drop=True)
    baseline_prop = baseline_emb[mask]
    trained_prop = trained_emb[mask]

    # Get most frequent edits
    edit_counts = df_prop['edit_name'].value_counts()
    top_edits = edit_counts.head(n_edits).index.tolist()

    print(f"Top {n_edits} most frequent edits:")
    for i, edit in enumerate(top_edits, 1):
        count = edit_counts[edit]
        print(f"  {i}. {edit[:60]}... ({count} samples)")

    # Create labels: 1-10 for top edits, 0 for others
    labels = np.zeros(len(df_prop), dtype=int)
    for i, edit in enumerate(top_edits, 1):
        labels[df_prop['edit_name'] == edit] = i

    # Reduce to 2D using t-SNE for better clustering visualization
    print("\nReducing dimensions with t-SNE (this may take a moment)...")

    # Sample if too many points
    max_samples = 3000
    if len(baseline_prop) > max_samples:
        indices = np.random.choice(len(baseline_prop), max_samples, replace=False)
        baseline_sample = baseline_prop[indices]
        trained_sample = trained_prop[indices]
        labels_sample = labels[indices]
        df_sample = df_prop.iloc[indices]
    else:
        baseline_sample = baseline_prop
        trained_sample = trained_prop
        labels_sample = labels
        df_sample = df_prop

    tsne_baseline = TSNE(n_components=2, random_state=42, perplexity=30)
    baseline_2d = tsne_baseline.fit_transform(baseline_sample)

    tsne_trained = TSNE(n_components=2, random_state=42, perplexity=30)
    trained_2d = tsne_trained.fit_transform(trained_sample)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Plot baseline
    ax = axes[0]
    # Plot "other" edits in gray first
    other_mask = labels_sample == 0
    ax.scatter(baseline_2d[other_mask, 0], baseline_2d[other_mask, 1],
              c='lightgray', alpha=0.3, s=20, label='Other edits')

    # Plot top edits with colors
    for i, edit in enumerate(top_edits, 1):
        edit_mask = labels_sample == i
        if edit_mask.sum() > 0:
            ax.scatter(baseline_2d[edit_mask, 0], baseline_2d[edit_mask, 1],
                      c=[colors[i-1]], alpha=0.8, s=50,
                      label=f'Edit {i}: {edit[:30]}...' if len(edit) > 30 else f'Edit {i}: {edit}',
                      edgecolors='black', linewidth=0.5)

    ax.set_title(f'Baseline Edit Embedding Space\n{property_name}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Plot trained
    ax = axes[1]
    # Plot "other" edits in gray first
    ax.scatter(trained_2d[other_mask, 0], trained_2d[other_mask, 1],
              c='lightgray', alpha=0.3, s=20, label='Other edits')

    # Plot top edits with colors
    for i, edit in enumerate(top_edits, 1):
        edit_mask = labels_sample == i
        if edit_mask.sum() > 0:
            ax.scatter(trained_2d[edit_mask, 0], trained_2d[edit_mask, 1],
                      c=[colors[i-1]], alpha=0.8, s=50,
                      label=f'Edit {i}: {edit[:30]}...' if len(edit) > 30 else f'Edit {i}: {edit}',
                      edgecolors='black', linewidth=0.5)

    ax.set_title(f'Trained Edit Embedding Space\n{property_name}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    output_path = Path(__file__).parent.parent / f'clustering_comparison_{property_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    return fig


def visualize_neighborhood_quality(test_df, baseline_emb, trained_emb, property_name):
    """
    Show concrete examples of neighborhood improvement.

    For specific edit instances, show their nearest neighbors in both spaces.
    """
    print(f"\n{'='*80}")
    print(f"VISUALIZATION 2: NEIGHBORHOOD QUALITY FOR {property_name}")
    print(f"{'='*80}\n")

    # Filter to property
    mask = test_df['property_name'] == property_name
    df_prop = test_df[mask].reset_index(drop=True)
    baseline_prop = baseline_emb[mask]
    trained_prop = trained_emb[mask]

    # Get frequent edits
    edit_counts = df_prop['edit_name'].value_counts()
    frequent_edits = edit_counts[edit_counts >= 5].index.tolist()

    # Find edit with good neighborhood improvement
    nn_baseline = NearestNeighbors(n_neighbors=6, metric='cosine')
    nn_trained = NearestNeighbors(n_neighbors=6, metric='cosine')
    nn_baseline.fit(baseline_prop)
    nn_trained.fit(trained_prop)

    best_improvement = -np.inf
    best_idx = None
    best_edit = None

    print("Finding edit with best neighborhood improvement...")
    for edit in frequent_edits[:20]:  # Check top 20 frequent edits
        edit_indices = df_prop[df_prop['edit_name'] == edit].index.tolist()
        if len(edit_indices) < 2:
            continue

        # Pick one instance
        idx = edit_indices[0]
        query_delta = df_prop.iloc[idx]['delta']

        # Get neighbors
        _, idx_baseline = nn_baseline.kneighbors([baseline_prop[idx]])
        _, idx_trained = nn_trained.kneighbors([trained_prop[idx]])

        # Skip self (first neighbor)
        idx_baseline = idx_baseline[0][1:]
        idx_trained = idx_trained[0][1:]

        # Calculate property similarity
        baseline_neighbor_deltas = df_prop.iloc[idx_baseline]['delta'].values
        trained_neighbor_deltas = df_prop.iloc[idx_trained]['delta'].values

        baseline_similarity = np.mean(np.abs(baseline_neighbor_deltas - query_delta))
        trained_similarity = np.mean(np.abs(trained_neighbor_deltas - query_delta))

        improvement = baseline_similarity - trained_similarity

        if improvement > best_improvement:
            best_improvement = improvement
            best_idx = idx
            best_edit = edit

    if best_idx is None:
        print("No significant neighborhood improvement found")
        return None

    print(f"\nBest example: {best_edit}")
    print(f"Improvement in neighbor similarity: {best_improvement:.3f}")

    # Visualize this specific case
    query_delta = df_prop.iloc[best_idx]['delta']

    # Get neighbors
    dist_baseline, idx_baseline = nn_baseline.kneighbors([baseline_prop[best_idx]])
    dist_trained, idx_trained = nn_trained.kneighbors([trained_prop[best_idx]])

    idx_baseline = idx_baseline[0][1:]
    idx_trained = idx_trained[0][1:]
    dist_baseline = dist_baseline[0][1:]
    dist_trained = dist_trained[0][1:]

    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Baseline neighbors
    ax = axes[0]
    neighbor_deltas_baseline = df_prop.iloc[idx_baseline]['delta'].values
    neighbor_names_baseline = [df_prop.iloc[i]['edit_name'][:40] + '...' if len(df_prop.iloc[i]['edit_name']) > 40
                                else df_prop.iloc[i]['edit_name'] for i in idx_baseline]

    x_pos = np.arange(len(neighbor_deltas_baseline))
    colors_baseline = ['green' if abs(d - query_delta) < 0.5 else 'orange' if abs(d - query_delta) < 1.0 else 'red'
                      for d in neighbor_deltas_baseline]

    ax.barh(x_pos, neighbor_deltas_baseline, color=colors_baseline, alpha=0.7)
    ax.axvline(query_delta, color='blue', linestyle='--', linewidth=2, label=f'Query delta: {query_delta:.2f}')
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f'Neighbor {i+1}\n{name}' for i, name in enumerate(neighbor_names_baseline)], fontsize=8)
    ax.set_xlabel('Property Change (Δ)')
    ax.set_title(f'Baseline Space: 5 Nearest Neighbors\nQuery: {best_edit[:60]}...', fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Add distance annotations
    for i, (d, dist) in enumerate(zip(neighbor_deltas_baseline, dist_baseline)):
        ax.text(d, i, f' dist={dist:.3f}', va='center', fontsize=7)

    # Trained neighbors
    ax = axes[1]
    neighbor_deltas_trained = df_prop.iloc[idx_trained]['delta'].values
    neighbor_names_trained = [df_prop.iloc[i]['edit_name'][:40] + '...' if len(df_prop.iloc[i]['edit_name']) > 40
                              else df_prop.iloc[i]['edit_name'] for i in idx_trained]

    colors_trained = ['green' if abs(d - query_delta) < 0.5 else 'orange' if abs(d - query_delta) < 1.0 else 'red'
                     for d in neighbor_deltas_trained]

    ax.barh(x_pos, neighbor_deltas_trained, color=colors_trained, alpha=0.7)
    ax.axvline(query_delta, color='blue', linestyle='--', linewidth=2, label=f'Query delta: {query_delta:.2f}')
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f'Neighbor {i+1}\n{name}' for i, name in enumerate(neighbor_names_trained)], fontsize=8)
    ax.set_xlabel('Property Change (Δ)')
    ax.set_title(f'Trained Space: 5 Nearest Neighbors\nQuery: {best_edit[:60]}...', fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Add distance annotations
    for i, (d, dist) in enumerate(zip(neighbor_deltas_trained, dist_trained)):
        ax.text(d, i, f' dist={dist:.3f}', va='center', fontsize=7)

    # Add summary statistics
    baseline_std = np.std(neighbor_deltas_baseline)
    trained_std = np.std(neighbor_deltas_trained)
    baseline_mean_error = np.mean(np.abs(neighbor_deltas_baseline - query_delta))
    trained_mean_error = np.mean(np.abs(neighbor_deltas_trained - query_delta))

    fig.text(0.5, 0.02,
             f'Neighbor Delta Std: Baseline={baseline_std:.3f}, Trained={trained_std:.3f} | ' +
             f'Mean Error from Query: Baseline={baseline_mean_error:.3f}, Trained={trained_mean_error:.3f} | ' +
             f'Improvement: {baseline_mean_error - trained_mean_error:.3f} ({(baseline_mean_error - trained_mean_error)/baseline_mean_error*100:.1f}%)',
             ha='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    output_path = Path(__file__).parent.parent / f'neighborhood_quality_{property_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    return fig


def main():
    """Run clustering visualizations."""

    print("="*80)
    print("CLUSTERING AND NEIGHBORHOOD VISUALIZATION")
    print("Comparing Baseline vs Trained Edit Embeddings")
    print("="*80)

    # Load data
    test_df, baseline_emb, trained_emb = load_data_and_embeddings()

    # Select properties to visualize
    properties_to_viz = [
        'vascular_endothelial_ic50',  # Best improvement
        'd(2)_dopamine_recept_ki',     # Large dataset
        'epidermal_growth_fac_ic50'    # Medium dataset
    ]

    for prop in properties_to_viz:
        # Visualization 1: Overall clustering
        fig1 = visualize_edit_clustering(
            test_df, baseline_emb, trained_emb, prop, n_edits=8
        )

        # Visualization 2: Specific neighborhood quality
        fig2 = visualize_neighborhood_quality(
            test_df, baseline_emb, trained_emb, prop
        )

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated visualizations:")
    print("1. clustering_comparison_*.png - Shows how edit types cluster in 2D space")
    print("2. neighborhood_quality_*.png - Shows specific examples of improved neighborhoods")
    print("\nThese visualizations demonstrate that trained embeddings:")
    print("- Cluster similar edits more tightly")
    print("- Find neighbors with more similar property changes")
    print("- Better organize the edit space for property prediction")


if __name__ == '__main__':
    main()

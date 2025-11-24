"""
Structured edit effect predictor using MMP-derived fragment features.

Similar to EditEffectPredictor but with sophisticated fragment-level analysis.
Requires MMP structural information in the dataset.

Key differences from EditEffectPredictor:
- Uses StructuredEditEmbedder instead of simple difference
- Requires MMP atom indices in dataset
- Richer feature representation with fragments and environment
- Optional property prediction head for consistency loss (future)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import Optional, List, Union, Dict, Tuple
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem

from src.embedding.small_molecule.structured_edit_embedder import StructuredEditEmbedder


# Enable Tensor Cores for L4 GPU
torch.set_float32_matmul_precision('high')


class StructuredEditEffectMLP(pl.LightningModule):
    """
    Multi-layer perceptron for structured edit effect prediction.

    Uses structured edit embeddings from MMP analysis to predict Δproperty.

    Architecture:
        GNN → fragment features → StructuredEditEmbedder → MLP → Δproperty

    Args:
        gnn_dim: Dimension of GNN embeddings (default: 300 for ChemProp)
        edit_mlp_dims: Hidden dimensions for edit embedding MLP
        delta_mlp_dims: Hidden dimensions for delta prediction MLP
        dropout: Dropout probability
        learning_rate: Learning rate for MLP heads
        gnn_learning_rate: Learning rate for GNN (if trainable)
        mol_embedder: Reference to molecule embedder (for GNN parameter grouping)
        n_tasks: Number of properties to predict (1 for single-task)
        task_names: Optional list of task names
        task_weights: Optional dict of task weights for loss
        k_hop_env: Number of hops for attachment environment
        use_local_delta: Whether to use atom-level local deltas
        use_rdkit_fragment_descriptors: Whether to compute fragment descriptors
    """

    def __init__(
        self,
        gnn_dim: int = 300,
        edit_mlp_dims: Optional[List[int]] = None,
        delta_mlp_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        gnn_learning_rate: Optional[float] = None,
        mol_embedder: Optional[nn.Module] = None,
        n_tasks: int = 1,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None,
        k_hop_env: int = 2,
        use_local_delta: bool = True,
        use_rdkit_fragment_descriptors: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['mol_embedder'])

        self.gnn_dim = gnn_dim
        self.learning_rate = learning_rate
        self.gnn_learning_rate = gnn_learning_rate if gnn_learning_rate is not None else 1e-5
        self.n_tasks = n_tasks

        # Molecule embedder (for separate GNN learning rate)
        self.mol_embedder = mol_embedder

        # Task names
        if task_names is None:
            if n_tasks == 1:
                self.task_names = ['delta_property']
            else:
                self.task_names = [f'task_{i}' for i in range(n_tasks)]
        else:
            if len(task_names) != n_tasks:
                raise ValueError(f"Number of task names ({len(task_names)}) must match n_tasks ({n_tasks})")
            self.task_names = task_names

        # Task weights for loss
        if task_weights is None:
            self.task_weights = {name: 1.0 for name in self.task_names}
        else:
            self.task_weights = task_weights

        # Structured edit embedder
        self.structured_edit_embedder = StructuredEditEmbedder(
            gnn_dim=gnn_dim,
            edit_mlp_dims=edit_mlp_dims,
            dropout=dropout,
            k_hop_env=k_hop_env,
            use_local_delta=use_local_delta,
            use_rdkit_fragment_descriptors=use_rdkit_fragment_descriptors
        )

        # Delta prediction MLP
        if delta_mlp_dims is None:
            delta_mlp_dims = [512, 256, 128]

        # Multi-task prediction heads
        if n_tasks == 1:
            # Single-task network
            layers = []
            prev_dim = self.structured_edit_embedder.output_dim

            for hidden_dim in delta_mlp_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            self.delta_predictor = nn.Sequential(*layers)

        else:
            # Multi-task network: shared backbone + task-specific heads
            from src.models.architectures.multi_head import MultiTaskNetwork

            self.multi_task_network = MultiTaskNetwork(
                input_dim=self.structured_edit_embedder.output_dim,
                backbone_hidden_dims=delta_mlp_dims,
                shared_dim=delta_mlp_dims[-1],  # Use last hidden dim as shared_dim
                task_names=self.task_names,
                dropout=dropout
            )

    def forward(
        self,
        # GNN embeddings
        H_A: torch.Tensor,
        H_B: torch.Tensor,
        h_A_global: torch.Tensor,
        h_B_global: torch.Tensor,
        # MMP structural info
        mol_A_list: List[Chem.Mol],
        mol_B_list: List[Chem.Mol],
        removed_atoms_list: List[List[int]],
        added_atoms_list: List[List[int]],
        attach_atoms_list: List[List[int]],
        mapped_pairs_list: Optional[List[List[Tuple[int, int]]]] = None
    ) -> torch.Tensor:
        """
        Forward pass for a batch.

        Args:
            H_A: Atom embeddings for molecules A [batch, max_atoms_A, gnn_dim]
            H_B: Atom embeddings for molecules B [batch, max_atoms_B, gnn_dim]
            h_A_global: Global embeddings for A [batch, gnn_dim]
            h_B_global: Global embeddings for B [batch, gnn_dim]
            mol_A_list: List of RDKit Mol objects for A
            mol_B_list: List of RDKit Mol objects for B
            removed_atoms_list: List of removed atom indices per example
            added_atoms_list: List of added atom indices per example
            attach_atoms_list: List of attachment atom indices per example
            mapped_pairs_list: Optional list of mapped atom pairs per example

        Returns:
            Predictions [batch, n_tasks] or [batch, 1] for single-task
        """
        batch_size = len(mol_A_list)
        edit_embeddings = []

        for i in range(batch_size):
            # Get structured edit embedding for this example
            edit_features = self.structured_edit_embedder(
                H_A=H_A[i],
                H_B=H_B[i],
                h_A_global=h_A_global[i],
                h_B_global=h_B_global[i],
                mol_A=mol_A_list[i],
                mol_B=mol_B_list[i],
                removed_atom_indices_A=removed_atoms_list[i],
                added_atom_indices_B=added_atoms_list[i],
                attach_atom_indices_A=attach_atoms_list[i],
                mapped_atom_pairs=mapped_pairs_list[i] if mapped_pairs_list else None
            )

            edit_embeddings.append(edit_features['edit_embedding'])

        # Stack into batch
        edit_embeddings = torch.stack(edit_embeddings)  # [batch, output_dim]

        # Predict delta
        if self.n_tasks == 1:
            predictions = self.delta_predictor(edit_embeddings)  # [batch, 1]
        else:
            predictions = self.multi_task_network(edit_embeddings)  # [batch, n_tasks]

        return predictions

    def training_step(self, batch, batch_idx):
        """Training step - simplified placeholder."""
        # This would need to be implemented based on your data format
        # For now, return a dummy loss
        return torch.tensor(0.0, requires_grad=True)

    def configure_optimizers(self):
        """Configure optimizer with separate learning rates for GNN and MLP."""
        # Check if we need separate learning rates for GNN
        if self.mol_embedder is not None and hasattr(self.mol_embedder, 'trainable') and self.mol_embedder.trainable:
            # Separate learning rates for GNN and MLP heads
            param_groups = []

            # GNN parameters (lower learning rate)
            if hasattr(self.mol_embedder, 'message_passing'):
                gnn_params = list(self.mol_embedder.message_passing.parameters())
                gnn_trainable_params = [p for p in gnn_params if p.requires_grad]

                if gnn_trainable_params:
                    gnn_param_count = sum(p.numel() for p in gnn_trainable_params)
                    param_groups.append({
                        'params': gnn_trainable_params,
                        'lr': self.gnn_learning_rate,
                        'name': 'gnn'
                    })
                    print(f"\n{'='*70}")
                    print(f"OPTIMIZER SETUP:")
                    print(f"  → GNN: {len(gnn_trainable_params)} tensors, {gnn_param_count:,} params (lr={self.gnn_learning_rate})")

            # All other parameters (MLP heads, edit embedder, etc.)
            mlp_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue

                # Skip GNN parameters (already added)
                is_gnn_param = False
                if hasattr(self.mol_embedder, 'message_passing'):
                    for gnn_param in self.mol_embedder.message_passing.parameters():
                        if param is gnn_param:
                            is_gnn_param = True
                            break

                if not is_gnn_param:
                    mlp_params.append(param)

            if mlp_params:
                mlp_param_count = sum(p.numel() for p in mlp_params)
                param_groups.append({
                    'params': mlp_params,
                    'lr': self.learning_rate,
                    'name': 'mlp_heads'
                })
                print(f"  → MLP: {len(mlp_params)} tensors, {mlp_param_count:,} params (lr={self.learning_rate})")
                print(f"  → TOTAL: {gnn_param_count + mlp_param_count:,} trainable parameters")
                print(f"{'='*70}\n")

            optimizer = torch.optim.Adam(param_groups)
        else:
            # Single learning rate for all parameters
            mlp_param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"\n{'='*70}")
            print(f"OPTIMIZER SETUP:")
            print(f"  → MLP only: {mlp_param_count:,} params (lr={self.learning_rate})")
            print(f"  → GNN: FROZEN (not included in optimizer)")
            print(f"{'='*70}\n")
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


class StructuredEditEffectPredictor:
    """
    High-level API for structured edit effect prediction.

    This is the main interface that users interact with. It handles:
    - GNN encoding to get atom embeddings
    - MMP structural feature extraction
    - Training and prediction

    Args:
        mol_embedder: Molecule embedder (e.g., ChemPropEmbedder with GNN)
        gnn_dim: Dimension of GNN embeddings
        edit_mlp_dims: Hidden dimensions for edit embedding MLP
        delta_mlp_dims: Hidden dimensions for delta prediction MLP
        dropout: Dropout probability
        learning_rate: Learning rate for MLP heads
        gnn_learning_rate: Learning rate for GNN (if trainable)
        batch_size: Batch size
        max_epochs: Maximum training epochs
        device: Device to use ('cuda', 'cpu', or None for auto)
        task_names: List of task names for multi-task learning
        task_weights: Dict of task weights for multi-task loss weighting
        k_hop_env: Number of hops for attachment environment
        use_local_delta: Whether to use atom-level local deltas
        use_rdkit_fragment_descriptors: Whether to compute fragment descriptors
    """

    def __init__(
        self,
        mol_embedder,
        gnn_dim: int = 300,
        edit_mlp_dims: Optional[List[int]] = None,
        delta_mlp_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        gnn_learning_rate: Optional[float] = None,
        batch_size: int = 32,
        max_epochs: int = 100,
        device: Optional[str] = None,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None,
        k_hop_env: int = 2,
        use_local_delta: bool = True,
        use_rdkit_fragment_descriptors: bool = True
    ):
        self.mol_embedder = mol_embedder
        self.gnn_dim = gnn_dim
        self.edit_mlp_dims = edit_mlp_dims
        self.delta_mlp_dims = delta_mlp_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.gnn_learning_rate = gnn_learning_rate if gnn_learning_rate is not None else 1e-5
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.task_names = task_names
        self.task_weights = task_weights
        self.k_hop_env = k_hop_env
        self.use_local_delta = use_local_delta
        self.use_rdkit_fragment_descriptors = use_rdkit_fragment_descriptors

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.trainer = None
        self.n_tasks = len(task_names) if task_names is not None else 1

    def fit(self, train_data: Dict, val_data: Optional[Dict] = None, verbose: bool = True):
        """
        Train the model.

        Args:
            train_data: Dictionary with training data
            val_data: Optional validation data
            verbose: Show training progress

        Expected data format:
            {
                'smiles_A': List[str],
                'smiles_B': List[str],
                'removed_atoms_A': List[List[int]],
                'added_atoms_B': List[List[int]],
                'attach_atoms_A': List[List[int]],
                'mapped_pairs': List[List[Tuple[int, int]]] (optional),
                'delta_y': np.ndarray [n_samples, n_tasks] or [n_samples,]
            }
        """
        print(f"\n{'='*70}")
        print("Structured Edit Effect Predictor - Training")
        print(f"{'='*70}\n")

        # TODO: Implement full training loop
        # This requires:
        # 1. Processing MMP data format
        # 2. Computing GNN embeddings
        # 3. Creating PyTorch datasets
        # 4. Training with PyTorch Lightning

        raise NotImplementedError(
            "Training loop for StructuredEditEffectPredictor needs to be implemented.\n"
            "This requires dataset processing for MMP structural features.\n"
            "See specification in docstring for expected data format."
        )

    def predict(self, test_data: Dict) -> np.ndarray:
        """
        Predict on test data.

        Args:
            test_data: Dictionary with test data (same format as fit())

        Returns:
            Predictions as numpy array
        """
        raise NotImplementedError(
            "Prediction for StructuredEditEffectPredictor needs to be implemented."
        )

"""
Edit effect predictor: f(molecule, edit) → Δproperty

This is the CAUSAL model that predicts how an edit changes a property.
For baseline property prediction, use PropertyPredictor instead.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Union, Tuple
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class EditEffectMLP(pl.LightningModule):
    """
    Multi-layer perceptron for edit effect prediction: f(molecule, edit) → ΔY

    Architecture automatically scales down from input_dim to 1:
    [mol_dim + edit_dim] → [input_dim/2] → [input_dim/4] → ... → [1]

    Args:
        mol_dim: Molecule embedding dimension
        edit_dim: Edit embedding dimension
        hidden_dims: Optional list of hidden dimensions. If None, auto-generates halving layers
        dropout: Dropout probability
        learning_rate: Learning rate for Adam optimizer
        activation: Activation function ('relu', 'elu', 'gelu')
    """

    def __init__(
        self,
        mol_dim: int,
        edit_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        activation: str = 'relu'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.mol_dim = mol_dim
        self.edit_dim = edit_dim
        self.input_dim = mol_dim + edit_dim
        self.learning_rate = learning_rate

        # Auto-generate hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = []
            current_dim = self.input_dim

            # Halve until we reach 64 (or until we've halved 3 times max)
            min_hidden_dim = 64
            max_layers = 3

            for _ in range(max_layers):
                current_dim = current_dim // 2
                if current_dim < min_hidden_dim:
                    break
                hidden_dims.append(current_dim)

            # Ensure at least one hidden layer
            if len(hidden_dims) == 0:
                hidden_dims = [max(self.input_dim // 2, 64)]

        self.hidden_dims = hidden_dims

        # Choose activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                act_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, mol_emb, edit_emb):
        """
        Forward pass: (molecule_embedding, edit_embedding) → delta_property

        Args:
            mol_emb: Molecule embedding tensor [batch_size, mol_dim]
            edit_emb: Edit embedding tensor [batch_size, edit_dim]

        Returns:
            Predicted delta [batch_size]
        """
        # Concatenate molecule and edit embeddings
        x = torch.cat([mol_emb, edit_emb], dim=-1)
        return self.network(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        mol_emb, edit_emb, delta_y = batch
        delta_pred = self(mol_emb, edit_emb)
        loss = nn.functional.mse_loss(delta_pred, delta_y)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae', nn.functional.l1_loss(delta_pred, delta_y))

        return loss

    def validation_step(self, batch, batch_idx):
        mol_emb, edit_emb, delta_y = batch
        delta_pred = self(mol_emb, edit_emb)
        loss = nn.functional.mse_loss(delta_pred, delta_y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', nn.functional.l1_loss(delta_pred, delta_y), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        mol_emb, edit_emb, delta_y = batch
        delta_pred = self(mol_emb, edit_emb)
        loss = nn.functional.mse_loss(delta_pred, delta_y)

        self.log('test_loss', loss)
        self.log('test_mae', nn.functional.l1_loss(delta_pred, delta_y))
        self.log('test_rmse', torch.sqrt(loss))

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Learning rate scheduler with plateau reduction
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class EditEffectPredictor:
    """
    High-level wrapper for causal edit effect prediction.

    This model learns F(m, e, c) = E[ΔY | m, e, c], the expected change
    in property Y when applying edit e to molecule m in context c.

    Usage:
        >>> from src.embedding.small_molecule import FingerprintEmbedder, EditEmbedder
        >>> from src.models.edit_effect_predictor import EditEffectPredictor
        >>>
        >>> # Create embedders
        >>> mol_embedder = FingerprintEmbedder(fp_type='morgan', radius=2, n_bits=512)
        >>> edit_embedder = EditEmbedder(mol_embedder)
        >>>
        >>> # Create predictor
        >>> predictor = EditEffectPredictor(
        ...     mol_embedder=mol_embedder,
        ...     edit_embedder=edit_embedder
        ... )
        >>>
        >>> # Train on paired data
        >>> smiles_a = ['CCO', 'c1ccccc1', ...]  # Before edit
        >>> smiles_b = ['CC(=O)O', 'c1ccncc1', ...]  # After edit
        >>> delta_y = [1.5, -0.8, ...]  # Property changes
        >>> predictor.fit(smiles_a, smiles_b, delta_y)
        >>>
        >>> # Predict edit effect
        >>> delta_pred = predictor.predict('CCO', 'CC(=O)O')
    """

    def __init__(
        self,
        mol_embedder,
        edit_embedder,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        device: Optional[str] = None
    ):
        """
        Initialize edit effect predictor.

        Args:
            mol_embedder: MoleculeEmbedder instance
            edit_embedder: EditEmbedder instance
            hidden_dims: Hidden layer dimensions (None for auto)
            dropout: Dropout probability
            learning_rate: Learning rate
            batch_size: Batch size
            max_epochs: Maximum training epochs
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.mol_embedder = mol_embedder
        self.edit_embedder = edit_embedder
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.trainer = None

    def fit(
        self,
        smiles_a: List[str],
        smiles_b: List[str],
        delta_y: Union[List[float], np.ndarray],
        smiles_a_val: Optional[List[str]] = None,
        smiles_b_val: Optional[List[str]] = None,
        delta_y_val: Optional[Union[List[float], np.ndarray]] = None,
        verbose: bool = True
    ):
        """
        Train the model on molecular pairs.

        Args:
            smiles_a: SMILES before edit (reactants)
            smiles_b: SMILES after edit (products)
            delta_y: Property changes (y_b - y_a)
            smiles_a_val: Optional validation reactants
            smiles_b_val: Optional validation products
            delta_y_val: Optional validation deltas
            verbose: Show training progress
        """
        # Embed molecules and edits
        print(f"Embedding {len(smiles_a)} training molecule pairs...")
        mol_emb_train = self.mol_embedder.encode(smiles_a)
        edit_emb_train = self.edit_embedder.encode_from_smiles(smiles_a, smiles_b)
        delta_y = np.array(delta_y, dtype=np.float32)

        # Convert to tensors
        mol_tensor = torch.FloatTensor(mol_emb_train)
        edit_tensor = torch.FloatTensor(edit_emb_train)
        delta_tensor = torch.FloatTensor(delta_y)

        train_dataset = TensorDataset(mol_tensor, edit_tensor, delta_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        # Validation data
        val_loader = None
        if smiles_a_val is not None and smiles_b_val is not None and delta_y_val is not None:
            print(f"Embedding {len(smiles_a_val)} validation pairs...")
            mol_emb_val = self.mol_embedder.encode(smiles_a_val)
            edit_emb_val = self.edit_embedder.encode_from_smiles(smiles_a_val, smiles_b_val)
            delta_y_val = np.array(delta_y_val, dtype=np.float32)

            mol_val_tensor = torch.FloatTensor(mol_emb_val)
            edit_val_tensor = torch.FloatTensor(edit_emb_val)
            delta_val_tensor = torch.FloatTensor(delta_y_val)

            val_dataset = TensorDataset(mol_val_tensor, edit_val_tensor, delta_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=0
            )

        # Initialize model
        mol_dim = mol_emb_train.shape[1]
        edit_dim = edit_emb_train.shape[1]

        self.model = EditEffectMLP(
            mol_dim=mol_dim,
            edit_dim=edit_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate
        )

        print(f"\nModel architecture:")
        print(f"  Molecule input: {mol_dim}")
        print(f"  Edit input: {edit_dim}")
        print(f"  Combined input: {mol_dim + edit_dim}")
        print(f"  Hidden: {self.model.hidden_dims}")
        print(f"  Output: 1")
        print(f"  Total params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
            logger=verbose,
            enable_checkpointing=False
        )

        # Train
        print(f"\nTraining for up to {self.max_epochs} epochs...")
        self.trainer.fit(self.model, train_loader, val_loader)

        print("Training complete!")

    def predict(
        self,
        smiles_a: Union[str, List[str]],
        smiles_b: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Predict edit effects.

        Args:
            smiles_a: Molecule before edit (single or list)
            smiles_b: Molecule after edit (single or list)

        Returns:
            Predicted delta (property change)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        is_single = isinstance(smiles_a, str)
        if is_single:
            smiles_a = [smiles_a]
            smiles_b = [smiles_b]

        # Embed
        mol_emb = self.mol_embedder.encode(smiles_a)
        edit_emb = self.edit_embedder.encode_from_smiles(smiles_a, smiles_b)

        mol_tensor = torch.FloatTensor(mol_emb).to(self.device)
        edit_tensor = torch.FloatTensor(edit_emb).to(self.device)

        # Predict
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            delta_pred = self.model(mol_tensor, edit_tensor).cpu().numpy()

        if is_single:
            return delta_pred[0]
        return delta_pred

    def predict_from_edit_smiles(
        self,
        smiles_a: Union[str, List[str]],
        edit_smiles: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Predict edit effects using reaction SMILES format.

        Args:
            smiles_a: Starting molecule(s)
            edit_smiles: Edit in format "reactant>>product"

        Returns:
            Predicted delta (property change)
        """
        is_single = isinstance(smiles_a, str)

        if is_single:
            if '>>' not in edit_smiles:
                raise ValueError("edit_smiles must be in format 'reactant>>product'")
            _, prod = edit_smiles.split('>>')
            return self.predict(smiles_a, prod)
        else:
            smiles_b = []
            for edit in edit_smiles:
                if '>>' not in edit:
                    raise ValueError("edit_smiles must be in format 'reactant>>product'")
                _, prod = edit.split('>>')
                smiles_b.append(prod)

            return self.predict(smiles_a, smiles_b)

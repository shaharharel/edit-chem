"""
Property predictor: SMILES → property value.

This is the baseline (non-causal) model that predicts Y(molecule).
For causal edit prediction, use EditEffectPredictor instead.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Union
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class PropertyPredictorMLP(pl.LightningModule):
    """
    Multi-layer perceptron for property prediction: f(molecule) → Y

    Architecture automatically scales down from input_dim to 1:
    [input_dim] → [input_dim/2] → [input_dim/4] → ... → [1]

    Args:
        input_dim: Input embedding dimension
        hidden_dims: Optional list of hidden dimensions. If None, auto-generates halving layers
        dropout: Dropout probability
        learning_rate: Learning rate for Adam optimizer
        activation: Activation function ('relu', 'elu', 'gelu')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        activation: str = 'relu'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.learning_rate = learning_rate

        # Auto-generate hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = []
            current_dim = input_dim

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
                hidden_dims = [max(input_dim // 2, 64)]

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
        prev_dim = input_dim

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

    def forward(self, x):
        """Forward pass: embedding → property value"""
        return self.network(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae', nn.functional.l1_loss(y_pred, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', nn.functional.l1_loss(y_pred, y), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.functional.mse_loss(y_pred, y)

        self.log('test_loss', loss)
        self.log('test_mae', nn.functional.l1_loss(y_pred, y))
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


class PropertyPredictor:
    """
    High-level wrapper for property prediction.

    Usage:
        >>> from src.embedding.small_molecule import FingerprintEmbedder
        >>> from src.models.property_predictor import PropertyPredictor
        >>>
        >>> # Create embedder
        >>> embedder = FingerprintEmbedder(fp_type='morgan', radius=2, n_bits=512)
        >>>
        >>> # Create predictor
        >>> predictor = PropertyPredictor(embedder=embedder)
        >>>
        >>> # Train
        >>> smiles_train = ['CCO', 'c1ccccc1', ...]
        >>> y_train = [3.5, 7.2, ...]
        >>> predictor.fit(smiles_train, y_train)
        >>>
        >>> # Predict
        >>> y_pred = predictor.predict(['CCO'])
    """

    def __init__(
        self,
        embedder,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        device: Optional[str] = None
    ):
        """
        Initialize property predictor.

        Args:
            embedder: MoleculeEmbedder instance (FingerprintEmbedder, ChemBERTaEmbedder, etc.)
            hidden_dims: Hidden layer dimensions (None for auto)
            dropout: Dropout probability
            learning_rate: Learning rate
            batch_size: Batch size
            max_epochs: Maximum training epochs
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.embedder = embedder
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
        smiles_train: List[str],
        y_train: Union[List[float], np.ndarray],
        smiles_val: Optional[List[str]] = None,
        y_val: Optional[Union[List[float], np.ndarray]] = None,
        verbose: bool = True
    ):
        """
        Train the model.

        Args:
            smiles_train: Training SMILES
            y_train: Training property values
            smiles_val: Optional validation SMILES
            y_val: Optional validation property values
            verbose: Show training progress
        """
        # Embed molecules
        print(f"Embedding {len(smiles_train)} training molecules...")
        X_train = self.embedder.encode(smiles_train)
        y_train = np.array(y_train, dtype=np.float32)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        # Validation data
        val_loader = None
        if smiles_val is not None and y_val is not None:
            print(f"Embedding {len(smiles_val)} validation molecules...")
            X_val = self.embedder.encode(smiles_val)
            y_val = np.array(y_val, dtype=np.float32)

            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)

            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=0
            )

        # Initialize model
        input_dim = X_train.shape[1]
        self.model = PropertyPredictorMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate
        )

        print(f"\nModel architecture:")
        print(f"  Input: {input_dim}")
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

    def predict(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Predict property values.

        Args:
            smiles: Single SMILES or list of SMILES

        Returns:
            Predicted property value(s)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        is_single = isinstance(smiles, str)
        if is_single:
            smiles = [smiles]

        # Embed
        X = self.embedder.encode(smiles)
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy()

        if is_single:
            return y_pred[0]
        return y_pred

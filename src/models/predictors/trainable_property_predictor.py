"""
End-to-end trainable property predictor with trainable encoder backbone.

This predictor computes embeddings on-the-fly during training, allowing
gradients to flow back through the encoder (GNN or transformer) for joint optimization.

Unlike PropertyPredictor which uses pre-computed embeddings, this class:
- Stores the embedder as part of the model
- Computes embeddings in the forward pass
- Supports separate learning rates for encoder and MLP

Supports any trainable encoder that implements:
- trainable: bool attribute
- encode_trainable(smiles) -> torch.Tensor method
- get_encoder_parameters() -> List[Parameter] method

Usage:
    from src.models.predictors.trainable_property_predictor import TrainablePropertyPredictor

    embedder = create_embedder('chemprop_dmpnn', trainable_encoder=True)
    # or: embedder = create_embedder('chemberta', trainable_encoder=True)
    predictor = TrainablePropertyPredictor(
        embedder=embedder,
        encoder_learning_rate=1e-5,  # Lower LR for encoder (GNN/transformer)
        mlp_learning_rate=1e-3,      # Higher LR for MLP heads
        ...
    )
    predictor.fit(smiles_train, y_train, smiles_val, y_val)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import Optional, List, Union
from torch.utils.data import DataLoader, Dataset


class SMILESDataset(Dataset):
    """Dataset that stores SMILES strings for on-the-fly embedding."""

    def __init__(self, smiles: List[str], y: np.ndarray):
        """
        Args:
            smiles: List of SMILES strings
            y: Target values [n_samples] for single-task or [n_samples, n_tasks] for multi-task
        """
        self.smiles = smiles
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.y[idx]


def smiles_collate_fn(batch):
    """Custom collate function to batch SMILES strings."""
    smiles = [item[0] for item in batch]
    y = torch.stack([item[1] for item in batch])
    return smiles, y


class TrainablePropertyMLP(pl.LightningModule):
    """
    End-to-end trainable property predictor with encoder + MLP.

    The encoder (GNN or transformer) is part of the model and gets updated during training.
    Uses separate learning rates for encoder and MLP parameters.
    """

    def __init__(
        self,
        embedder: nn.Module,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        encoder_learning_rate: float = 1e-5,
        mlp_learning_rate: float = 1e-3,
        n_tasks: int = 1,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['embedder'])

        self.embedder = embedder
        self.input_dim = input_dim
        self.encoder_learning_rate = encoder_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.n_tasks = n_tasks

        # Task names
        if task_names is None:
            if n_tasks == 1:
                self.task_names = ['property']
            else:
                self.task_names = [f'task_{i}' for i in range(n_tasks)]
        else:
            self.task_names = task_names

        # Task weights for loss
        if task_weights is None:
            self.task_weights = {name: 1.0 for name in self.task_names}
        else:
            self.task_weights = task_weights

        # Auto-generate hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = []
            current_dim = input_dim
            for _ in range(3):
                current_dim = current_dim // 2
                if current_dim < 64:
                    break
                hidden_dims.append(current_dim)
            if len(hidden_dims) == 0:
                hidden_dims = [max(input_dim // 2, 64)]

        self.hidden_dims = hidden_dims

        # Build network
        if n_tasks == 1:
            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
            self.multi_task_network = None
        else:
            from ..architectures.multi_head import MultiTaskNetwork

            shared_dim = hidden_dims[-1] if hidden_dims else max(input_dim // 4, 64)
            head_hidden_dim = max(shared_dim // 2, 32) if head_hidden_dims is None else None

            self.multi_task_network = MultiTaskNetwork(
                input_dim=input_dim,
                task_names=self.task_names,
                backbone_hidden_dims=hidden_dims,
                shared_dim=shared_dim,
                head_hidden_dim=head_hidden_dim,
                head_hidden_dims=head_hidden_dims,
                dropout=dropout
            )
            self.network = None

    def forward(self, smiles: List[str]):
        """
        Forward pass: SMILES → embeddings → predictions

        Args:
            smiles: List of SMILES strings

        Returns:
            Single-task: predictions [batch_size]
            Multi-task: dict {task_name: predictions [batch_size]}
        """
        # Compute embeddings on-the-fly with gradient tracking
        embeddings = self.embedder.encode_trainable(smiles)

        if self.n_tasks == 1:
            return self.network(embeddings).squeeze(-1)
        else:
            return self.multi_task_network(embeddings)

    def training_step(self, batch, batch_idx):
        smiles, y = batch

        if self.n_tasks == 1:
            y_pred = self(smiles)
            loss = nn.functional.mse_loss(y_pred, y)
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_mae', nn.functional.l1_loss(y_pred, y))
        else:
            y_pred_dict = self(smiles)

            total_loss = 0.0
            total_weight = 0.0

            for i, task_name in enumerate(self.task_names):
                y_task = y[:, i]
                y_pred_task = y_pred_dict[task_name]

                mask = ~torch.isnan(y_task)
                if mask.sum() > 0:
                    y_task_valid = y_task[mask]
                    y_pred_task_valid = y_pred_task[mask]

                    task_loss = nn.functional.mse_loss(y_pred_task_valid, y_task_valid)
                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight

                    self.log(f'train_loss_{task_name}', task_loss)

            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        smiles, y = batch

        if self.n_tasks == 1:
            y_pred = self(smiles)
            loss = nn.functional.mse_loss(y_pred, y)
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_mae', nn.functional.l1_loss(y_pred, y), prog_bar=True)
        else:
            y_pred_dict = self(smiles)

            total_loss = 0.0
            total_weight = 0.0

            for i, task_name in enumerate(self.task_names):
                y_task = y[:, i]
                y_pred_task = y_pred_dict[task_name]

                mask = ~torch.isnan(y_task)
                if mask.sum() > 0:
                    y_task_valid = y_task[mask]
                    y_pred_task_valid = y_pred_task[mask]

                    task_loss = nn.functional.mse_loss(y_pred_task_valid, y_task_valid)
                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight

                    self.log(f'val_loss_{task_name}', task_loss)
                    self.log(f'val_mae_{task_name}',
                            nn.functional.l1_loss(y_pred_task_valid, y_task_valid))

            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=self.device)

            self.log('val_loss', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer with separate learning rates for encoder and MLP."""
        param_groups = []

        # Get encoder parameters using generalized interface
        encoder_params = []
        if hasattr(self.embedder, 'get_encoder_parameters'):
            encoder_params = self.embedder.get_encoder_parameters()
        elif hasattr(self.embedder, 'message_passing') and self.embedder.trainable:
            # Fallback for backward compatibility with older embedders
            encoder_params = list(self.embedder.message_passing.parameters())

        encoder_trainable = [p for p in encoder_params if p.requires_grad]

        if encoder_trainable:
            encoder_param_count = sum(p.numel() for p in encoder_trainable)
            param_groups.append({
                'params': encoder_trainable,
                'lr': self.encoder_learning_rate,
                'name': 'encoder'
            })
            print(f"\n{'='*70}")
            print(f"OPTIMIZER SETUP (End-to-End Trainable):")
            encoder_type = type(self.embedder).__name__
            print(f"  → Encoder ({encoder_type}): {len(encoder_trainable)} tensors, {encoder_param_count:,} params (lr={self.encoder_learning_rate})")

        # MLP parameters (everything except encoder)
        mlp_params = []
        encoder_param_ids = {id(p) for p in encoder_params}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) not in encoder_param_ids:
                mlp_params.append(param)

        if mlp_params:
            mlp_param_count = sum(p.numel() for p in mlp_params)
            param_groups.append({
                'params': mlp_params,
                'lr': self.mlp_learning_rate,
                'name': 'mlp_heads'
            })
            if encoder_trainable:
                print(f"  → MLP: {len(mlp_params)} tensors, {mlp_param_count:,} params (lr={self.mlp_learning_rate})")
                total = sum(p.numel() for g in param_groups for p in g['params'])
                print(f"  → TOTAL: {total:,} trainable parameters")
                print(f"{'='*70}\n")

        optimizer = torch.optim.Adam(param_groups)

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
                'monitor': 'train_loss',  # Use train_loss to avoid issues when no val data
                'frequency': 1,
                'strict': False  # Don't raise error if metric not found
            }
        }


class TrainablePropertyPredictor:
    """
    High-level API for end-to-end trainable property prediction.

    Unlike PropertyPredictor, this class:
    - Does NOT pre-compute embeddings
    - Computes embeddings on-the-fly during training
    - Allows gradients to flow back through the encoder (GNN or transformer)
    - Uses separate learning rates for encoder and MLP

    Usage:
        embedder = create_embedder('chemprop_dmpnn', trainable_encoder=True)
        # or: embedder = create_embedder('chemberta', trainable_encoder=True)
        predictor = TrainablePropertyPredictor(
            embedder=embedder,
            encoder_learning_rate=1e-5,
            mlp_learning_rate=1e-3,
            ...
        )
        predictor.fit(smiles_train, y_train, smiles_val, y_val)
        y_pred = predictor.predict(smiles_test)
    """

    def __init__(
        self,
        embedder,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        encoder_learning_rate: float = 1e-5,
        mlp_learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 100,
        device: Optional[str] = None,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None
    ):
        self.embedder = embedder
        self.hidden_dims = hidden_dims
        self.head_hidden_dims = head_hidden_dims
        self.dropout = dropout
        self.encoder_learning_rate = encoder_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.task_names = task_names
        self.task_weights = task_weights

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.trainer = None
        self.n_tasks = len(task_names) if task_names is not None else 1

    def fit(
        self,
        smiles_train: List[str],
        y_train: Union[List[float], np.ndarray],
        smiles_val: Optional[List[str]] = None,
        y_val: Optional[Union[List[float], np.ndarray]] = None,
        verbose: bool = True
    ):
        """
        Train the model end-to-end.

        Args:
            smiles_train: Training SMILES strings
            y_train: Training property values
            smiles_val: Optional validation SMILES strings
            y_val: Optional validation property values
            verbose: Show training progress
        """
        print(f"\n{'='*70}")
        print("End-to-End Trainable Property Predictor - Training")
        print(f"{'='*70}\n")

        y_train = np.array(y_train, dtype=np.float32)

        # Handle multi-task labels
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1) if self.n_tasks == 1 else y_train.reshape(-1, 1)

        if self.n_tasks == 1:
            y_train = y_train.squeeze()

        # Create datasets
        train_dataset = SMILESDataset(smiles_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=smiles_collate_fn,
            num_workers=0  # SMILES processing is fast, no need for multiprocessing
        )

        val_loader = None
        if smiles_val is not None and y_val is not None:
            y_val = np.array(y_val, dtype=np.float32)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1) if self.n_tasks == 1 else y_val.reshape(-1, 1)
            if self.n_tasks == 1:
                y_val = y_val.squeeze()

            val_dataset = SMILESDataset(smiles_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=smiles_collate_fn,
                num_workers=0
            )

        # Initialize model
        input_dim = self.embedder.embedding_dim
        self.model = TrainablePropertyMLP(
            embedder=self.embedder,
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            head_hidden_dims=self.head_hidden_dims,
            dropout=self.dropout,
            encoder_learning_rate=self.encoder_learning_rate,
            mlp_learning_rate=self.mlp_learning_rate,
            n_tasks=self.n_tasks,
            task_names=self.task_names,
            task_weights=self.task_weights
        )

        # Move embedder to device (handle both GNN and transformer embedders)
        if hasattr(self.embedder, 'message_passing'):
            self.embedder.message_passing.to(self.device)
        if hasattr(self.embedder, 'aggregation'):
            self.embedder.aggregation.to(self.device)
        if hasattr(self.embedder, 'model'):  # For transformer-based embedders (ChemBERTa)
            self.embedder.model.to(self.device)

        encoder_type = type(self.embedder).__name__
        print(f"\nModel architecture:")
        print(f"  {'='*70}")
        print(f"  Encoder: {encoder_type} (trainable={self.embedder.trainable})")
        print(f"  Embedding dim: {input_dim}")
        print(f"  Shared backbone: {self.model.hidden_dims}")
        if self.n_tasks == 1:
            print(f"  Output: 1 (single-task)")
        else:
            print(f"  Multi-task heads: {self.n_tasks} tasks")
        print(f"  {'='*70}")
        print(f"  Encoder learning rate: {self.encoder_learning_rate}")
        print(f"  MLP learning rate: {self.mlp_learning_rate}")
        print(f"  {'='*70}")

        # Trainer
        from pytorch_lightning.loggers import CSVLogger
        import tempfile

        self.log_dir = tempfile.mkdtemp(prefix='trainable_prop_pred_')
        csv_logger = CSVLogger(self.log_dir, name='training')

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
            logger=csv_logger,
            enable_checkpointing=False
        )

        print(f"\nTraining for up to {self.max_epochs} epochs...")
        self.trainer.fit(self.model, train_loader, val_loader)

        print("Training complete!")

    def predict(self, smiles: Union[str, List[str]]) -> Union[np.ndarray, dict]:
        """
        Predict property values.

        Args:
            smiles: Single SMILES or list of SMILES

        Returns:
            Single-task: numpy array of predictions
            Multi-task: dict {task_name: predictions}
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        is_single = isinstance(smiles, str)
        if is_single:
            smiles = [smiles]

        self.model.eval()
        self.model.to(self.device)

        # Move embedder components to device (handle both GNN and transformer embedders)
        if hasattr(self.embedder, 'message_passing'):
            self.embedder.message_passing.to(self.device)
        if hasattr(self.embedder, 'aggregation'):
            self.embedder.aggregation.to(self.device)
        if hasattr(self.embedder, 'model'):  # For transformer-based embedders (ChemBERTa)
            self.embedder.model.to(self.device)

        with torch.no_grad():
            if self.n_tasks == 1:
                y_pred = self.model(smiles).cpu().numpy()
                if is_single:
                    return y_pred[0]
                return y_pred
            else:
                y_pred_dict = self.model(smiles)
                result = {}
                for task_name in self.task_names:
                    preds = y_pred_dict[task_name].cpu().numpy()
                    if is_single:
                        result[task_name] = preds[0]
                    else:
                        result[task_name] = preds
                return result

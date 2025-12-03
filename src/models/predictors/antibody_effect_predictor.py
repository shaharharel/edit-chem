"""
Antibody edit effect predictor: f(antibody, mutation) → Δproperty

This module provides the predictor for antibody mutation effects,
analogous to the small molecule EditEffectPredictor but designed
for paired heavy-light chain inputs with structured mutation information.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Union, Dict, Any, Tuple
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Enable Tensor Cores for L4 GPU
torch.set_float32_matmul_precision('high')


class AntibodyEffectMLP(pl.LightningModule):
    """
    Multi-layer perceptron for antibody mutation effect prediction.

    Takes antibody context embedding (h_context) and edit embedding (h_edit)
    as separate inputs, concatenates them, and predicts delta property values.

    Supports:
    - Single and multi-task learning
    - Configurable architecture with auto-scaling or explicit hidden dims
    - Optional separate learning rates for embedder and predictor

    Args:
        context_dim: Antibody context embedding dimension
        edit_dim: Edit embedding dimension
        hidden_dims: Optional list of hidden dimensions. If None, auto-generates
        head_hidden_dims: Optional hidden dims for task-specific heads
        dropout: Dropout probability
        learning_rate: Learning rate for optimizer
        activation: Activation function ('relu', 'elu', 'gelu')
        n_tasks: Number of properties to predict
        task_names: Optional list of task names
        task_weights: Optional dict of task weights for loss
        embedder: Reference to antibody embedder (for parameter grouping)
        embedder_learning_rate: Separate LR for embedder parameters
    """

    def __init__(
        self,
        context_dim: int,
        edit_dim: int,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        activation: str = 'relu',
        n_tasks: int = 1,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None,
        embedder: Optional[nn.Module] = None,
        embedder_learning_rate: Optional[float] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['embedder'])

        self.context_dim = context_dim
        self.edit_dim = edit_dim
        self.input_dim = context_dim + edit_dim
        self.learning_rate = learning_rate
        self.embedder_learning_rate = embedder_learning_rate or 1e-5
        self.n_tasks = n_tasks
        self.head_hidden_dims = head_hidden_dims
        self.embedder = embedder

        # Task names
        if task_names is None:
            if n_tasks == 1:
                self.task_names = ['delta_value']
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
        if n_tasks == 1:
            # Single-task network
            layers = []
            prev_dim = self.input_dim

            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    act_fn(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
            self.multi_task_network = None

        else:
            # Multi-task network
            from ..architectures.multi_head import MultiTaskNetwork

            shared_dim = hidden_dims[-1] if hidden_dims else max(self.input_dim // 4, 64)
            head_hidden_dim = max(shared_dim // 2, 32) if head_hidden_dims is None else None

            self.multi_task_network = MultiTaskNetwork(
                input_dim=self.input_dim,
                task_names=self.task_names,
                backbone_hidden_dims=hidden_dims,
                shared_dim=shared_dim,
                head_hidden_dim=head_hidden_dim,
                head_hidden_dims=head_hidden_dims,
                dropout=dropout,
                activation=activation
            )
            self.network = None

    def forward(
        self,
        h_context: torch.Tensor,
        h_edit: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            h_context: Context embedding [batch_size, context_dim]
            h_edit: Edit embedding [batch_size, edit_dim]

        Returns:
            Single-task: Predicted delta [batch_size]
            Multi-task: Dict {task_name: predictions [batch_size]}
        """
        # Concatenate context and edit embeddings
        x = torch.cat([h_context, h_edit], dim=-1)

        if self.n_tasks == 1:
            return self.network(x).squeeze(-1)
        else:
            return self.multi_task_network(x)

    def training_step(self, batch, batch_idx):
        h_context, h_edit, delta_y = batch[:3]

        if self.n_tasks == 1:
            delta_pred = self(h_context, h_edit)
            loss = nn.functional.mse_loss(delta_pred, delta_y)

            self.log('train_loss', loss, prog_bar=True)
            self.log('train_mae', nn.functional.l1_loss(delta_pred, delta_y))

        else:
            delta_pred_dict = self(h_context, h_edit)

            total_loss = 0.0
            total_weight = 0.0

            for i, task_name in enumerate(self.task_names):
                delta_task = delta_y[:, i]
                delta_pred_task = delta_pred_dict[task_name]

                # Handle NaN labels (sparse multi-task)
                mask = ~torch.isnan(delta_task)
                if mask.sum() > 0:
                    delta_task_valid = delta_task[mask]
                    delta_pred_task_valid = delta_pred_task[mask]

                    task_loss = nn.functional.mse_loss(delta_pred_task_valid, delta_task_valid)
                    task_mae = nn.functional.l1_loss(delta_pred_task_valid, delta_task_valid)

                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight

                    self.log(f'train_loss_{task_name}', task_loss)
                    self.log(f'train_mae_{task_name}', task_mae)

            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=delta_y.device)

            self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        h_context, h_edit, delta_y = batch[:3]

        if self.n_tasks == 1:
            delta_pred = self(h_context, h_edit)
            loss = nn.functional.mse_loss(delta_pred, delta_y)

            self.log('val_loss', loss, prog_bar=True)
            self.log('val_mae', nn.functional.l1_loss(delta_pred, delta_y), prog_bar=True)

        else:
            delta_pred_dict = self(h_context, h_edit)

            total_loss = 0.0
            total_weight = 0.0

            for i, task_name in enumerate(self.task_names):
                delta_task = delta_y[:, i]
                delta_pred_task = delta_pred_dict[task_name]

                mask = ~torch.isnan(delta_task)
                if mask.sum() > 0:
                    delta_task_valid = delta_task[mask]
                    delta_pred_task_valid = delta_pred_task[mask]

                    task_loss = nn.functional.mse_loss(delta_pred_task_valid, delta_task_valid)
                    task_mae = nn.functional.l1_loss(delta_pred_task_valid, delta_task_valid)

                    weight = self.task_weights[task_name]
                    total_loss += weight * task_loss
                    total_weight += weight

                    self.log(f'val_loss_{task_name}', task_loss)
                    self.log(f'val_mae_{task_name}', task_mae, prog_bar=True)

            if total_weight > 0:
                loss = total_loss / total_weight
            else:
                loss = torch.tensor(0.0, device=delta_y.device)

            self.log('val_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        h_context, h_edit, delta_y = batch[:3]

        if self.n_tasks == 1:
            delta_pred = self(h_context, h_edit)
            loss = nn.functional.mse_loss(delta_pred, delta_y)

            self.log('test_loss', loss)
            self.log('test_mae', nn.functional.l1_loss(delta_pred, delta_y))
            self.log('test_rmse', torch.sqrt(loss))

        else:
            delta_pred_dict = self(h_context, h_edit)

            total_loss = 0.0
            for i, task_name in enumerate(self.task_names):
                delta_task = delta_y[:, i]
                delta_pred_task = delta_pred_dict[task_name]

                task_loss = nn.functional.mse_loss(delta_pred_task, delta_task)
                task_mae = nn.functional.l1_loss(delta_pred_task, delta_task)
                task_rmse = torch.sqrt(task_loss)

                weight = self.task_weights[task_name]
                total_loss += weight * task_loss

                self.log(f'test_loss_{task_name}', task_loss)
                self.log(f'test_mae_{task_name}', task_mae)
                self.log(f'test_rmse_{task_name}', task_rmse)

            total_weight = sum(self.task_weights.values())
            loss = total_loss / total_weight
            self.log('test_loss', loss)

        return loss

    def configure_optimizers(self):
        # Check if embedder needs separate learning rate
        if self.embedder is not None and hasattr(self.embedder, 'trainable') and self.embedder.trainable:
            param_groups = []

            # Embedder parameters
            embedder_params = list(self.embedder.parameters())
            embedder_trainable = [p for p in embedder_params if p.requires_grad]

            if embedder_trainable:
                embedder_param_count = sum(p.numel() for p in embedder_trainable)
                param_groups.append({
                    'params': embedder_trainable,
                    'lr': self.embedder_learning_rate,
                    'name': 'embedder'
                })
                print(f"\nOPTIMIZER SETUP:")
                print(f"  → Embedder: {len(embedder_trainable)} tensors, {embedder_param_count:,} params (lr={self.embedder_learning_rate})")

            # MLP parameters
            mlp_params = []
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue

                is_embedder_param = any(param is ep for ep in embedder_params)
                if not is_embedder_param:
                    mlp_params.append(param)

            if mlp_params:
                mlp_param_count = sum(p.numel() for p in mlp_params)
                param_groups.append({
                    'params': mlp_params,
                    'lr': self.learning_rate,
                    'name': 'mlp'
                })
                print(f"  → MLP: {len(mlp_params)} tensors, {mlp_param_count:,} params (lr={self.learning_rate})")

            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
                'monitor': 'val_loss'
            }
        }


class AntibodyEffectPredictor:
    """
    High-level wrapper for antibody mutation effect prediction.

    This predictor learns F(ab, mutation) = E[ΔY | ab, mutation], the expected
    change in property Y when applying mutation to antibody ab.

    Supports:
    - Various antibody language models (IgT5, IgBert, AntiBERTa2, etc.)
    - Structured edit embeddings with identity and location features
    - Optional structural encoders (GVP, SE3-Transformer, Equiformer)
    - Multi-task learning for multiple property changes

    Example:
        >>> from src.embedding.antibody import IgT5Embedder
        >>> from src.embedding.antibody import StructuredAntibodyEditEmbedder
        >>> from src.models.predictors import AntibodyEffectPredictor
        >>>
        >>> # Create embedders
        >>> ab_embedder = IgT5Embedder()
        >>> edit_embedder = StructuredAntibodyEditEmbedder(ab_embedder)
        >>>
        >>> # Create predictor
        >>> predictor = AntibodyEffectPredictor(
        ...     ab_embedder=ab_embedder,
        ...     edit_embedder=edit_embedder,
        ...     task_names=['delta_binding', 'delta_expression']
        ... )
        >>>
        >>> # Train
        >>> predictor.fit(dataset)
        >>>
        >>> # Predict
        >>> delta_pred = predictor.predict(ab_pair)
    """

    def __init__(
        self,
        ab_embedder,
        edit_embedder,
        hidden_dims: Optional[List[int]] = None,
        head_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        embedder_learning_rate: Optional[float] = None,
        batch_size: int = 32,
        max_epochs: int = 100,
        device: Optional[str] = None,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None,
    ):
        """
        Initialize antibody effect predictor.

        Args:
            ab_embedder: AntibodyEmbedder instance (IgT5, IgBert, etc.)
            edit_embedder: AntibodyEditEmbedder or StructuredAntibodyEditEmbedder
            hidden_dims: Hidden layer dimensions (None for auto)
            head_hidden_dims: Hidden dims for task heads
            dropout: Dropout probability
            learning_rate: Learning rate for MLP
            embedder_learning_rate: Learning rate for embedder (if trainable)
            batch_size: Batch size
            max_epochs: Maximum training epochs
            device: 'cuda', 'cpu', or None (auto-detect)
            task_names: List of task names for multi-task learning
            task_weights: Dict of task weights for loss
        """
        self.ab_embedder = ab_embedder
        self.edit_embedder = edit_embedder
        self.hidden_dims = hidden_dims
        self.head_hidden_dims = head_hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.embedder_learning_rate = embedder_learning_rate or 1e-5
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.task_names = task_names
        self.task_weights = task_weights

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.model = None
        self.trainer = None
        self.n_tasks = len(task_names) if task_names is not None else 1

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ):
        """
        Train the model.

        Args:
            train_loader: Training DataLoader
            val_loader: Optional validation DataLoader
            verbose: Show training progress
        """
        # Get dimensions from first batch
        sample_batch = next(iter(train_loader))

        # Determine dimensions
        context_dim = self.ab_embedder.embedding_dim
        edit_dim = self.edit_embedder.embedding_dim

        # Initialize model
        self.model = AntibodyEffectMLP(
            context_dim=context_dim,
            edit_dim=edit_dim,
            hidden_dims=self.hidden_dims,
            head_hidden_dims=self.head_hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            embedder=self.ab_embedder if hasattr(self.ab_embedder, 'trainable') and self.ab_embedder.trainable else None,
            embedder_learning_rate=self.embedder_learning_rate,
            n_tasks=self.n_tasks,
            task_names=self.task_names,
            task_weights=self.task_weights,
        )

        print(f"\nModel architecture:")
        print(f"  Antibody embedder: {self.ab_embedder.name}")
        print(f"  Edit embedder: {self.edit_embedder.__class__.__name__}")
        print(f"  Context dim: {context_dim}")
        print(f"  Edit dim: {edit_dim}")
        print(f"  Combined input: {context_dim + edit_dim}")
        print(f"  Hidden dims: {self.model.hidden_dims}")
        if self.n_tasks > 1:
            print(f"  Tasks: {self.task_names}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Trainer
        from pytorch_lightning.loggers import CSVLogger
        import tempfile

        self.log_dir = tempfile.mkdtemp(prefix='antibody_effect_')
        csv_logger = CSVLogger(self.log_dir, name='training')

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='gpu' if self.device in ['cuda', 'mps'] else 'cpu',
            devices=1,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
            logger=csv_logger,
            enable_checkpointing=False
        )

        print(f"\nTraining for up to {self.max_epochs} epochs...")
        self.trainer.fit(self.model, train_loader, val_loader)
        print("Training complete!")

    def fit_from_dataset(
        self,
        dataset,
        val_ratio: float = 0.1,
        by_antibody: bool = True,
        verbose: bool = True,
    ):
        """
        Train from AbPairDataset.

        Args:
            dataset: AbPairDataset instance
            val_ratio: Fraction for validation
            by_antibody: Split by antibody ID to prevent leakage
            verbose: Show progress
        """
        from src.data.antibody import AbPairCollator

        # Split dataset
        train_ds, val_ds, _ = dataset.split(
            train_ratio=1.0 - val_ratio,
            val_ratio=val_ratio,
            test_ratio=0.0,
            by_antibody=by_antibody,
        )

        # Create collator with embedder for on-the-fly embedding
        collator = AbPairCollator(embedder=self.ab_embedder)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

        self.fit(train_loader, val_loader, verbose=verbose)

    def predict(
        self,
        heavy_wt: Union[str, List[str]],
        light_wt: Union[str, List[str]],
        heavy_mut: Union[str, List[str]],
        light_mut: Union[str, List[str]],
        mutations: Optional[List] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict mutation effects.

        Args:
            heavy_wt: Wild-type heavy chain(s)
            light_wt: Wild-type light chain(s)
            heavy_mut: Mutant heavy chain(s)
            light_mut: Mutant light chain(s)
            mutations: Optional mutation information for structured embedder

        Returns:
            Single-task: numpy array of predictions
            Multi-task: dict {task_name: predictions}
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        is_single = isinstance(heavy_wt, str)
        if is_single:
            heavy_wt = [heavy_wt]
            light_wt = [light_wt]
            heavy_mut = [heavy_mut]
            light_mut = [light_mut]
            if mutations is not None:
                mutations = [mutations]

        # Get embeddings
        batch_size = len(heavy_wt)
        h_context_list = []
        h_edit_list = []

        for i in range(batch_size):
            # Get context and edit embeddings
            if hasattr(self.edit_embedder, 'embed'):
                # StructuredAntibodyEditEmbedder
                output = self.edit_embedder.embed(
                    heavy_wt=heavy_wt[i],
                    light_wt=light_wt[i],
                    mutations=mutations[i] if mutations else None,
                )
                h_context = output.h_context
                h_edit = output.h_edit
            else:
                # Simple AntibodyEditEmbedder
                output = self.edit_embedder.encode(
                    heavy_wt=heavy_wt[i],
                    light_wt=light_wt[i],
                    heavy_mut=heavy_mut[i],
                    light_mut=light_mut[i],
                )
                h_context = output.global_embedding
                h_edit = output.edit_embedding

            h_context_list.append(h_context)
            h_edit_list.append(h_edit)

        # Stack
        h_context = torch.stack(h_context_list).to(self.device)
        h_edit = torch.stack(h_edit_list).to(self.device)

        # Predict
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            if self.n_tasks == 1:
                delta_pred = self.model(h_context, h_edit).cpu().numpy()
                if is_single:
                    return delta_pred[0]
                return delta_pred
            else:
                delta_pred_dict = self.model(h_context, h_edit)
                result = {}
                for task_name in self.task_names:
                    preds = delta_pred_dict[task_name].cpu().numpy()
                    if is_single:
                        result[task_name] = preds[0]
                    else:
                        result[task_name] = preds
                return result

    def predict_from_pair(self, pair) -> Union[float, Dict[str, float]]:
        """
        Predict mutation effect from AbEditPair.

        Args:
            pair: AbEditPair instance

        Returns:
            Predicted delta value(s)
        """
        return self.predict(
            heavy_wt=pair.heavy_wt,
            light_wt=pair.light_wt,
            heavy_mut=pair.heavy_mut,
            light_mut=pair.light_mut,
            mutations=pair.mutations,
        )

    def evaluate(
        self,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            test_loader: Test DataLoader

        Returns:
            Dict with test metrics (loss, mae, rmse, etc.)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        self.model.eval()
        self.model.to(self.device)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                h_context, h_edit, delta_y = batch[:3]
                h_context = h_context.to(self.device)
                h_edit = h_edit.to(self.device)

                if self.n_tasks == 1:
                    preds = self.model(h_context, h_edit)
                    all_preds.append(preds.cpu())
                    all_targets.append(delta_y)
                else:
                    pred_dict = self.model(h_context, h_edit)
                    # Stack predictions
                    preds = torch.stack([pred_dict[t] for t in self.task_names], dim=1)
                    all_preds.append(preds.cpu())
                    all_targets.append(delta_y)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metrics
        metrics = {}

        if self.n_tasks == 1:
            mse = nn.functional.mse_loss(all_preds, all_targets)
            mae = nn.functional.l1_loss(all_preds, all_targets)
            rmse = torch.sqrt(mse)

            # Correlation
            preds_np = all_preds.numpy()
            targets_np = all_targets.numpy()
            corr = np.corrcoef(preds_np, targets_np)[0, 1]

            metrics = {
                'test_loss': mse.item(),
                'test_mae': mae.item(),
                'test_rmse': rmse.item(),
                'test_pearson': corr,
            }
        else:
            for i, task_name in enumerate(self.task_names):
                task_preds = all_preds[:, i]
                task_targets = all_targets[:, i]

                # Handle NaN
                mask = ~torch.isnan(task_targets)
                if mask.sum() > 0:
                    task_preds = task_preds[mask]
                    task_targets = task_targets[mask]

                    mse = nn.functional.mse_loss(task_preds, task_targets)
                    mae = nn.functional.l1_loss(task_preds, task_targets)
                    rmse = torch.sqrt(mse)

                    preds_np = task_preds.numpy()
                    targets_np = task_targets.numpy()
                    corr = np.corrcoef(preds_np, targets_np)[0, 1]

                    metrics[f'test_loss_{task_name}'] = mse.item()
                    metrics[f'test_mae_{task_name}'] = mae.item()
                    metrics[f'test_rmse_{task_name}'] = rmse.item()
                    metrics[f'test_pearson_{task_name}'] = corr

        return metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'hyperparameters': {
                'context_dim': self.model.context_dim,
                'edit_dim': self.model.edit_dim,
                'hidden_dims': self.hidden_dims,
                'head_hidden_dims': self.head_hidden_dims,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'task_names': self.task_names,
                'task_weights': self.task_weights,
                'n_tasks': self.n_tasks,
            }
        }

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        ab_embedder,
        edit_embedder,
        device: Optional[str] = None,
    ):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        hparams = checkpoint['hyperparameters']

        predictor = cls(
            ab_embedder=ab_embedder,
            edit_embedder=edit_embedder,
            hidden_dims=hparams['hidden_dims'],
            head_hidden_dims=hparams.get('head_hidden_dims'),
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate'],
            device=device,
            task_names=hparams['task_names'],
            task_weights=hparams['task_weights'],
        )

        predictor.model = AntibodyEffectMLP(
            context_dim=hparams['context_dim'],
            edit_dim=hparams['edit_dim'],
            hidden_dims=hparams['hidden_dims'],
            head_hidden_dims=hparams.get('head_hidden_dims'),
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate'],
            n_tasks=hparams['n_tasks'],
            task_names=hparams['task_names'],
            task_weights=hparams['task_weights'],
        )

        predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.model.to(predictor.device)
        predictor.model.eval()

        return predictor

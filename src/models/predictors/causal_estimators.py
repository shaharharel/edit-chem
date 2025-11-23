"""
Causal estimators for edit effects.

Implements:
1. Naive estimator: E[Y|e] (ignores confounding)
2. IPW (Inverse Probability Weighting): Reweights by propensity scores
3. Doubly Robust (DR): Combines outcome modeling + propensity weighting

These estimators handle selection bias when edits are not randomly assigned.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Optional, List, Union, Tuple, Dict
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import logging

logger = logging.getLogger(__name__)


class PropensityScoreModel(pl.LightningModule):
    """
    Propensity score model: P(edit=e | molecule=m, context=c)

    This estimates the probability that a particular edit was applied
    to a molecule, which is needed for IPW and DR estimators.

    For multi-class edit selection, uses softmax over all observed edits.
    For binary (edit applied or not), uses sigmoid.
    """

    def __init__(
        self,
        input_dim: int,
        n_edits: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3
    ):
        """
        Args:
            input_dim: Molecule embedding dimension
            n_edits: Number of unique edits (for multi-class)
            hidden_dims: Hidden layer dimensions (None for auto)
            dropout: Dropout probability
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.n_edits = n_edits
        self.learning_rate = learning_rate

        # Auto-generate hidden dims
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

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer (logits for each edit)
        layers.append(nn.Linear(prev_dim, n_edits))

        self.network = nn.Sequential(*layers)

    def forward(self, mol_emb):
        """
        Forward pass: molecule_embedding → edit probabilities

        Args:
            mol_emb: Molecule embedding [batch_size, input_dim]

        Returns:
            Logits [batch_size, n_edits]
        """
        return self.network(mol_emb)

    def training_step(self, batch, batch_idx):
        mol_emb, edit_idx = batch
        logits = self(mol_emb)
        loss = nn.functional.cross_entropy(logits, edit_idx)

        # Compute accuracy
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == edit_idx).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        mol_emb, edit_idx = batch
        logits = self(mol_emb)
        loss = nn.functional.cross_entropy(logits, edit_idx)

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == edit_idx).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }

    def predict_proba(self, mol_emb_tensor):
        """
        Predict edit probabilities.

        Args:
            mol_emb_tensor: Molecule embeddings [batch_size, input_dim]

        Returns:
            Probabilities [batch_size, n_edits]
        """
        self.eval()
        with torch.no_grad():
            logits = self(mol_emb_tensor)
            probs = torch.softmax(logits, dim=-1)
        return probs


class IPWEstimator:
    """
    Inverse Probability Weighting (IPW) estimator for edit effects.

    Reweights observations by inverse propensity scores to remove selection bias:

        E[ΔY | do(e)] ≈ E[ΔY · I(edit=e) / π_e(m)]

    where π_e(m) = P(edit=e | molecule=m) is the propensity score.

    This corrects for confounding when edit selection is non-random.
    """

    def __init__(
        self,
        mol_embedder,
        edit_embedder,
        clip_propensity: Tuple[float, float] = (0.01, 0.99),
        batch_size: int = 32,
        max_epochs: int = 50,
        device: Optional[str] = None
    ):
        """
        Args:
            mol_embedder: MoleculeEmbedder instance
            edit_embedder: EditEmbedder instance
            clip_propensity: Min/max values to clip propensity scores (stabilization)
            batch_size: Batch size for propensity model training
            max_epochs: Max epochs for propensity model
            device: 'cuda', 'cpu', or None (auto)
        """
        self.mol_embedder = mol_embedder
        self.edit_embedder = edit_embedder
        self.clip_min, self.clip_max = clip_propensity
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.propensity_model = None
        self.edit_to_idx = None
        self.idx_to_edit = None

    def fit_propensity_model(
        self,
        smiles_a: List[str],
        smiles_b: List[str],
        verbose: bool = True
    ):
        """
        Fit propensity score model: P(edit | molecule).

        Args:
            smiles_a: Reactant SMILES
            smiles_b: Product SMILES (defines which edit was applied)
            verbose: Show training progress
        """
        logger.info("Fitting propensity score model...")

        # Embed molecules
        mol_emb = self.mol_embedder.encode(smiles_a)

        # Create edit index mapping
        # For simplicity, we'll use smiles_b as a proxy for edit identity
        # In practice, you might want to use edit_smiles or edit_name
        unique_products = sorted(set(smiles_b))
        self.edit_to_idx = {edit: idx for idx, edit in enumerate(unique_products)}
        self.idx_to_edit = {idx: edit for edit, idx in self.edit_to_idx.items()}

        edit_indices = np.array([self.edit_to_idx[s] for s in smiles_b])

        logger.info(f"Found {len(unique_products)} unique edits")

        # Split into train/val
        from sklearn.model_selection import train_test_split
        mol_train, mol_val, edit_train, edit_val = train_test_split(
            mol_emb, edit_indices, test_size=0.2, random_state=42
        )

        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(mol_train),
            torch.LongTensor(edit_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(mol_val),
            torch.LongTensor(edit_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

        # Initialize model
        self.propensity_model = PropensityScoreModel(
            input_dim=mol_emb.shape[1],
            n_edits=len(unique_products),
            learning_rate=1e-3
        )

        # Train
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
            logger=False,
            enable_checkpointing=False
        )

        trainer.fit(self.propensity_model, train_loader, val_loader)
        logger.info("Propensity model training complete")

    def estimate_effect(
        self,
        smiles_a: List[str],
        smiles_b: List[str],
        delta_y: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate average treatment effect using IPW.

        Args:
            smiles_a: Reactant SMILES
            smiles_b: Product SMILES
            delta_y: Observed property changes

        Returns:
            Dictionary with ATE estimate and diagnostics
        """
        if self.propensity_model is None:
            raise RuntimeError("Propensity model not fitted! Call fit_propensity_model() first.")

        # Embed molecules
        mol_emb = self.mol_embedder.encode(smiles_a)
        mol_tensor = torch.FloatTensor(mol_emb).to(self.device)

        # Get propensity scores
        self.propensity_model.to(self.device)
        self.propensity_model.eval()

        with torch.no_grad():
            all_probs = self.propensity_model.predict_proba(mol_tensor).cpu().numpy()

        # Get propensity for actual edit applied
        edit_indices = [self.edit_to_idx.get(s, 0) for s in smiles_b]
        propensities = np.array([all_probs[i, edit_indices[i]] for i in range(len(edit_indices))])

        # Clip propensities for stability
        propensities_clipped = np.clip(propensities, self.clip_min, self.clip_max)

        # IPW weights
        weights = 1.0 / propensities_clipped

        # Weighted average
        ate = np.average(delta_y, weights=weights)

        # Diagnostics
        effective_n = (weights.sum() ** 2) / (weights ** 2).sum()

        return {
            'ate': ate,
            'propensity_mean': propensities.mean(),
            'propensity_min': propensities.min(),
            'propensity_max': propensities.max(),
            'effective_sample_size': effective_n,
            'max_weight': weights.max(),
            'mean_weight': weights.mean()
        }


class DoublyRobustEstimator:
    """
    Doubly Robust (DR) estimator for edit effects.

    Combines outcome modeling and propensity weighting for robustness:

        E[ΔY | do(e)] ≈ E[μ_e(m) - μ_0(m)] + E[(ΔY - μ_e(m)) · I(e) / π_e(m)]

    where:
    - μ_e(m) = E[ΔY | m, e] is the outcome model
    - π_e(m) = P(e | m) is the propensity score

    This estimator is "doubly robust": it gives correct estimates if EITHER
    the outcome model OR the propensity model is correct (but not necessarily both).

    Implementation uses cross-fitting to avoid overfitting bias.
    """

    def __init__(
        self,
        mol_embedder,
        edit_embedder,
        n_folds: int = 5,
        clip_propensity: Tuple[float, float] = (0.01, 0.99),
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 50,
        device: Optional[str] = None
    ):
        """
        Args:
            mol_embedder: MoleculeEmbedder instance
            edit_embedder: EditEmbedder instance
            n_folds: Number of cross-fitting folds
            clip_propensity: Min/max for propensity clipping
            hidden_dims: Hidden dims for outcome model
            dropout: Dropout for outcome model
            learning_rate: Learning rate
            batch_size: Batch size
            max_epochs: Max epochs
            device: Device
        """
        self.mol_embedder = mol_embedder
        self.edit_embedder = edit_embedder
        self.n_folds = n_folds
        self.clip_min, self.clip_max = clip_propensity
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.outcome_models = []
        self.propensity_models = []
        self.edit_to_idx = None

    def fit(
        self,
        smiles_a: List[str],
        smiles_b: List[str],
        delta_y: np.ndarray,
        verbose: bool = True
    ):
        """
        Fit DR estimator using cross-fitting.

        Cross-fitting procedure:
        1. Split data into K folds
        2. For each fold k:
            - Train outcome model on folds != k
            - Train propensity model on folds != k
            - Use these models to predict on fold k
        3. Combine pseudo-outcomes from all folds

        Args:
            smiles_a: Reactant SMILES
            smiles_b: Product SMILES
            delta_y: Property changes
            verbose: Show progress
        """
        logger.info(f"Fitting Doubly Robust estimator with {self.n_folds}-fold cross-fitting...")

        # Embed molecules and edits
        mol_emb = self.mol_embedder.encode(smiles_a)
        edit_emb = self.edit_embedder.encode_from_smiles(smiles_a, smiles_b)

        # Create edit index mapping
        unique_products = sorted(set(smiles_b))
        self.edit_to_idx = {edit: idx for idx, edit in enumerate(unique_products)}
        edit_indices = np.array([self.edit_to_idx[s] for s in smiles_b])

        # K-fold cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Store pseudo-outcomes for final estimate
        self.pseudo_outcomes = np.zeros_like(delta_y)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(mol_emb)):
            logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")

            # Split data
            mol_train = mol_emb[train_idx]
            mol_test = mol_emb[test_idx]
            edit_train = edit_emb[train_idx]
            edit_test = edit_emb[test_idx]
            delta_train = delta_y[train_idx]
            delta_test = delta_y[test_idx]
            edit_idx_train = edit_indices[train_idx]
            edit_idx_test = edit_indices[test_idx]

            # 1. Train outcome model on train fold
            outcome_model = self._train_outcome_model(
                mol_train, edit_train, delta_train,
                verbose=verbose and fold_idx == 0  # Only show progress for first fold
            )
            self.outcome_models.append(outcome_model)

            # 2. Train propensity model on train fold
            propensity_model = self._train_propensity_model(
                mol_train, edit_idx_train,
                verbose=verbose and fold_idx == 0
            )
            self.propensity_models.append(propensity_model)

            # 3. Compute pseudo-outcomes for test fold
            # μ_e(m) from outcome model
            mol_test_tensor = torch.FloatTensor(mol_test).to(self.device)
            edit_test_tensor = torch.FloatTensor(edit_test).to(self.device)

            outcome_model.to(self.device)
            outcome_model.eval()

            with torch.no_grad():
                mu_predictions = outcome_model(mol_test_tensor, edit_test_tensor).cpu().numpy()

            # π_e(m) from propensity model
            propensity_model.to(self.device)
            propensity_model.eval()

            with torch.no_grad():
                all_probs = propensity_model.predict_proba(mol_test_tensor).cpu().numpy()

            propensities = np.array([all_probs[i, edit_idx_test[i]] for i in range(len(edit_idx_test))])
            propensities = np.clip(propensities, self.clip_min, self.clip_max)

            # DR pseudo-outcome: μ_e(m) + (Y - μ_e(m)) / π_e(m)
            residuals = delta_test - mu_predictions
            pseudo_outcome = mu_predictions + residuals / propensities

            self.pseudo_outcomes[test_idx] = pseudo_outcome

        logger.info("Cross-fitting complete")

    def _train_outcome_model(self, mol_emb, edit_emb, delta_y, verbose):
        """Train outcome model μ(m, e) on given data."""
        from .edit_effect_predictor import EditEffectMLP

        # Create dataset
        dataset = TensorDataset(
            torch.FloatTensor(mol_emb),
            torch.FloatTensor(edit_emb),
            torch.FloatTensor(delta_y)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # Initialize model
        model = EditEffectMLP(
            mol_dim=mol_emb.shape[1],
            edit_dim=edit_emb.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate
        )

        # Train
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            enable_progress_bar=verbose,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False
        )

        trainer.fit(model, loader)

        return model

    def _train_propensity_model(self, mol_emb, edit_idx, verbose):
        """Train propensity model π(e | m) on given data."""
        # Create dataset
        dataset = TensorDataset(
            torch.FloatTensor(mol_emb),
            torch.LongTensor(edit_idx)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # Initialize model
        model = PropensityScoreModel(
            input_dim=mol_emb.shape[1],
            n_edits=len(self.edit_to_idx),
            learning_rate=self.learning_rate
        )

        # Train
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            enable_progress_bar=verbose,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False
        )

        trainer.fit(model, loader)

        return model

    def estimate_effect(self) -> Dict[str, float]:
        """
        Estimate average treatment effect using DR estimator.

        Returns:
            Dictionary with ATE estimate and diagnostics
        """
        if len(self.outcome_models) == 0:
            raise RuntimeError("DR estimator not fitted! Call fit() first.")

        # ATE is simply the mean of pseudo-outcomes
        ate = self.pseudo_outcomes.mean()
        ate_std = self.pseudo_outcomes.std()
        ate_se = ate_std / np.sqrt(len(self.pseudo_outcomes))

        return {
            'ate': ate,
            'ate_std': ate_std,
            'ate_se': ate_se,
            'ate_ci_lower': ate - 1.96 * ate_se,
            'ate_ci_upper': ate + 1.96 * ate_se,
            'n_samples': len(self.pseudo_outcomes)
        }

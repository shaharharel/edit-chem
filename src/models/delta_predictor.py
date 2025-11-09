"""Delta property predictor: f(molecule, edit) → Δproperty."""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.embeddings.fingerprints import FingerprintGenerator

logger = logging.getLogger(__name__)

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install torch for neural network models.")


class DeltaPropertyMLP(nn.Module):
    """
    Multi-layer perceptron for delta property prediction.

    Architecture:
    - Input: Concatenated (molecule_fp, edit_fp)
    - Hidden layers with ReLU + Dropout
    - Output: Single value (Δ property)
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.2):
        """
        Initialize MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x).squeeze(-1)


class DeltaPropertyPredictor:
    """
    Predict change in molecular property given a molecule and an edit.

    Model: Δproperty = f(molecule_features, edit_features)

    This is the core prediction model - given a starting molecule and
    a proposed edit, predict how much the property will change.
    """

    def __init__(self,
                 model_type: str = 'random_forest',
                 embedding_type: str = 'morgan',
                 **model_kwargs):
        """
        Initialize delta predictor.

        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'neural_network'
            embedding_type: Type of molecular fingerprint
            **model_kwargs: Additional arguments for the model
                For neural_network:
                    - hidden_dims: List[int] = [512, 256, 128]
                    - dropout: float = 0.2
                    - learning_rate: float = 0.001
                    - batch_size: int = 32
                    - epochs: int = 100
                    - device: str = 'cuda' or 'cpu'
        """
        self.model_type = model_type
        self.embedding_type = embedding_type

        # Initialize featurizer
        self.featurizer = FingerprintGenerator(fp_type=embedding_type)

        # Initialize model
        if model_type == 'random_forest':
            default_kwargs = {'n_estimators': 100, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1}
            default_kwargs.update(model_kwargs)
            self.model = RandomForestRegressor(**default_kwargs)

        elif model_type == 'gradient_boosting':
            default_kwargs = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
            default_kwargs.update(model_kwargs)
            self.model = GradientBoostingRegressor(**default_kwargs)

        elif model_type == 'neural_network':
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for neural_network model. Install with: pip install torch")

            # Store hyperparameters for neural network
            self.hidden_dims = model_kwargs.get('hidden_dims', [512, 256, 128])
            self.dropout = model_kwargs.get('dropout', 0.2)
            self.learning_rate = model_kwargs.get('learning_rate', 0.001)
            self.batch_size = model_kwargs.get('batch_size', 32)
            self.epochs = model_kwargs.get('epochs', 100)
            self.device = model_kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

            # Model will be initialized in train() when we know input_dim
            self.model = None

        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose from: random_forest, gradient_boosting, neural_network")

        self.is_trained = False

        logger.info(f"Initialized {model_type} delta predictor with {embedding_type} features")

    def featurize(self, smiles: str, edit) -> Optional[np.ndarray]:
        """
        Convert (molecule, edit) pair to feature vector.

        Features:
        - Molecule fingerprint (e.g., 2048-bit Morgan)
        - Edit fingerprint (difference: to_group - from_group)
        - Combined into single vector

        Args:
            smiles: SMILES string
            edit: Edit object

        Returns:
            Feature vector or None if featurization failed
        """
        return self.featurizer.generate_edit_features(smiles, edit)

    def train(self,
              X_train: List[Tuple[str, object]],
              y_train: List[float],
              cv_folds: int = 5,
              X_val: Optional[List[Tuple[str, object]]] = None,
              y_val: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X_train: List of (smiles, edit) tuples
            y_train: List of delta property values
            cv_folds: Number of cross-validation folds (sklearn models only)
            X_val: Optional validation set (neural network only)
            y_val: Optional validation labels (neural network only)

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training on {len(X_train)} examples...")

        # Featurize training data
        logger.info("Featurizing training data...")
        X_features = []
        valid_indices = []

        for i, (smiles, edit) in enumerate(X_train):
            feat = self.featurize(smiles, edit)
            if feat is not None:
                X_features.append(feat)
                valid_indices.append(i)
            else:
                logger.warning(f"Failed to featurize example {i}")

        X_features = np.array(X_features)
        y_train_valid = np.array([y_train[i] for i in valid_indices])

        logger.info(f"Valid features: {len(X_features)} / {len(X_train)}")

        if len(X_features) == 0:
            logger.error("No valid features generated!")
            return {}

        # Neural network training
        if self.model_type == 'neural_network':
            return self._train_neural_network(X_features, y_train_valid, X_val, y_val)

        # Sklearn model training
        # Cross-validation
        logger.info(f"Running {cv_folds}-fold cross-validation...")
        cv_r2 = cross_val_score(self.model, X_features, y_train_valid,
                                cv=cv_folds, scoring='r2', n_jobs=-1)
        cv_mae = cross_val_score(self.model, X_features, y_train_valid,
                                 cv=cv_folds, scoring='neg_mean_absolute_error', n_jobs=-1)

        logger.info(f"CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
        logger.info(f"CV MAE: {-cv_mae.mean():.3f} ± {cv_mae.std():.3f}")

        # Train on full data
        logger.info("Training on full dataset...")
        self.model.fit(X_features, y_train_valid)

        self.is_trained = True

        # Training set performance
        y_pred_train = self.model.predict(X_features)
        train_r2 = r2_score(y_train_valid, y_pred_train)
        train_mae = mean_absolute_error(y_train_valid, y_pred_train)

        logger.info(f"Training R²: {train_r2:.3f}")
        logger.info(f"Training MAE: {train_mae:.3f}")

        return {
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std(),
            'cv_mae_mean': -cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'train_r2': train_r2,
            'train_mae': train_mae,
            'n_train': len(X_features)
        }

    def _train_neural_network(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: Optional[List[Tuple[str, object]]] = None,
                             y_val: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Train neural network model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation set (smiles, edit) tuples
            y_val: Optional validation labels

        Returns:
            Dictionary of training metrics
        """
        # Initialize model with correct input dimension
        input_dim = X_train.shape[1]
        self.model = DeltaPropertyMLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        logger.info(f"Initialized neural network with input_dim={input_dim}")
        logger.info(f"Architecture: {input_dim} -> {self.hidden_dims} -> 1")
        logger.info(f"Device: {self.device}")

        # Prepare data
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Prepare validation data if provided
        X_val_features = None
        if X_val is not None and y_val is not None:
            logger.info("Featurizing validation data...")
            X_val_features = []
            for smiles, edit in X_val:
                feat = self.featurize(smiles, edit)
                if feat is not None:
                    X_val_features.append(feat)

            if len(X_val_features) > 0:
                X_val_features = np.array(X_val_features)
                y_val = np.array(y_val)[:len(X_val_features)]
                X_val_tensor = torch.FloatTensor(X_val_features).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            else:
                X_val_features = None

        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        logger.info(f"Training for {self.epochs} epochs...")
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)

            # Validation
            if X_val_features is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = criterion(val_predictions, y_val_tensor).item()
                    val_losses.append(val_loss)

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.best_model_state = self.model.state_dict()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}")

        # Restore best model if validation was used
        if X_val_features is not None and hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model from validation")

        self.is_trained = True

        # Compute training metrics
        self.model.eval()
        with torch.no_grad():
            y_pred_train = self.model(X_tensor).cpu().numpy()
            train_r2 = r2_score(y_train, y_pred_train)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

        logger.info(f"Training R²: {train_r2:.3f}")
        logger.info(f"Training MAE: {train_mae:.3f}")
        logger.info(f"Training RMSE: {train_rmse:.3f}")

        metrics = {
            'train_r2': train_r2,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'n_train': len(X_train),
            'final_train_loss': train_losses[-1]
        }

        if X_val_features is not None:
            with torch.no_grad():
                y_pred_val = self.model(X_val_tensor).cpu().numpy()
                val_r2 = r2_score(y_val, y_pred_val)
                val_mae = mean_absolute_error(y_val, y_pred_val)

            logger.info(f"Validation R²: {val_r2:.3f}")
            logger.info(f"Validation MAE: {val_mae:.3f}")

            metrics.update({
                'val_r2': val_r2,
                'val_mae': val_mae,
                'best_val_loss': best_val_loss,
                'final_val_loss': val_losses[-1]
            })

        return metrics

    def predict(self, smiles: str, edit) -> Optional[float]:
        """
        Predict delta property for a single (molecule, edit) pair.

        Args:
            smiles: SMILES string
            edit: Edit object

        Returns:
            Predicted change in property or None if failed
        """
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return None

        feat = self.featurize(smiles, edit)
        if feat is None:
            return None

        try:
            if self.model_type == 'neural_network':
                self.model.eval()
                with torch.no_grad():
                    feat_tensor = torch.FloatTensor(feat).unsqueeze(0).to(self.device)
                    pred = self.model(feat_tensor).cpu().item()
            else:
                pred = self.model.predict([feat])[0]

            return float(pred)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return None

    def predict_batch(self, X_test: List[Tuple[str, object]]) -> np.ndarray:
        """
        Predict for multiple (molecule, edit) pairs.

        Args:
            X_test: List of (smiles, edit) tuples

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return np.array([])

        # Featurize
        X_features = []
        for smiles, edit in X_test:
            feat = self.featurize(smiles, edit)
            if feat is not None:
                X_features.append(feat)
            else:
                # Use NaN for failed featurization
                X_features.append(None)

        # Predict
        if self.model_type == 'neural_network':
            # Batch prediction for neural network
            predictions = []
            valid_features = [f for f in X_features if f is not None]

            if len(valid_features) > 0:
                self.model.eval()
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(np.array(valid_features)).to(self.device)
                    preds = self.model(features_tensor).cpu().numpy()

                pred_idx = 0
                for feat in X_features:
                    if feat is not None:
                        predictions.append(preds[pred_idx])
                        pred_idx += 1
                    else:
                        predictions.append(np.nan)
            else:
                predictions = [np.nan] * len(X_features)
        else:
            # Sklearn prediction
            predictions = []
            for feat in X_features:
                if feat is not None:
                    pred = self.model.predict([feat])[0]
                    predictions.append(pred)
                else:
                    predictions.append(np.nan)

        return np.array(predictions)

    def evaluate(self,
                X_test: List[Tuple[str, object]],
                y_test: List[float]) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: List of (smiles, edit) tuples
            y_test: List of true delta values

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {len(X_test)} test examples...")

        y_pred = self.predict_batch(X_test)

        # Remove NaN predictions
        valid_mask = ~np.isnan(y_pred)
        y_pred_valid = y_pred[valid_mask]
        y_test_valid = np.array(y_test)[valid_mask]

        if len(y_pred_valid) == 0:
            logger.error("No valid predictions!")
            return {}

        # Compute metrics
        r2 = r2_score(y_test_valid, y_pred_valid)
        mae = mean_absolute_error(y_test_valid, y_pred_valid)
        rmse = np.sqrt(mean_squared_error(y_test_valid, y_pred_valid))

        logger.info(f"Test R²: {r2:.3f}")
        logger.info(f"Test MAE: {mae:.3f}")
        logger.info(f"Test RMSE: {rmse:.3f}")

        return {
            'test_r2': r2,
            'test_mae': mae,
            'test_rmse': rmse,
            'n_test': len(y_test_valid),
            'n_failed': len(y_test) - len(y_test_valid)
        }

    def get_feature_importance(self, top_k: int = 20) -> Optional[np.ndarray]:
        """
        Get feature importance (for tree-based models).

        Args:
            top_k: Number of top features to return

        Returns:
            Array of feature importance scores
        """
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return None

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-top_k:][::-1]

            logger.info(f"Top {top_k} feature importances:")
            for i, idx in enumerate(top_indices):
                logger.info(f"  {i+1}. Feature {idx}: {importances[idx]:.4f}")

            return importances
        else:
            logger.warning("Model does not support feature importance")
            return None

    def save(self, filepath: str) -> None:
        """
        Save trained model to disk.

        Args:
            filepath: Path to save file
        """
        if not self.is_trained:
            logger.warning("Saving untrained model!")

        save_dict = {
            'model_type': self.model_type,
            'embedding_type': self.embedding_type,
            'is_trained': self.is_trained
        }

        # Handle neural network separately (save state dict)
        if self.model_type == 'neural_network':
            save_dict.update({
                'model_state_dict': self.model.state_dict(),
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'device': self.device
            })
        else:
            save_dict['model'] = self.model

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DeltaPropertyPredictor':
        """
        Load trained model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded DeltaPropertyPredictor
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        model_type = save_dict['model_type']
        embedding_type = save_dict['embedding_type']

        # Reconstruct neural network with saved hyperparameters
        if model_type == 'neural_network':
            predictor = cls(
                model_type=model_type,
                embedding_type=embedding_type,
                hidden_dims=save_dict['hidden_dims'],
                dropout=save_dict['dropout'],
                learning_rate=save_dict['learning_rate'],
                batch_size=save_dict['batch_size'],
                epochs=save_dict['epochs'],
                device=save_dict['device']
            )

            # Get input_dim from state dict
            state_dict = save_dict['model_state_dict']
            input_dim = state_dict['network.0.weight'].shape[1]

            # Initialize model and load weights
            predictor.model = DeltaPropertyMLP(
                input_dim=input_dim,
                hidden_dims=save_dict['hidden_dims'],
                dropout=save_dict['dropout']
            ).to(save_dict['device'])

            predictor.model.load_state_dict(state_dict)
            predictor.model.eval()

        else:
            predictor = cls(
                model_type=model_type,
                embedding_type=embedding_type
            )
            predictor.model = save_dict['model']

        predictor.is_trained = save_dict['is_trained']

        logger.info(f"Model loaded from {filepath}")

        return predictor

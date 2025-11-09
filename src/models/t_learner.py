"""T-Learner for causal effect estimation."""

import logging
import numpy as np
from typing import List, Dict, Optional
import pickle

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

from src.embeddings.fingerprints import FingerprintGenerator

logger = logging.getLogger(__name__)


class TLearner:
    """
    T-Learner for causal effect estimation.

    Trains two separate models:
    - μ₀(x): predicts property for control group (pre-edit molecules)
    - μ₁(x): predicts property for treatment group (post-edit molecules)

    Causal effect estimate: τ(x) = μ₁(x) - μ₀(x)

    This is complementary to DeltaPropertyPredictor:
    - DeltaPropertyPredictor: directly predicts Δ from (molecule, edit)
    - TLearner: predicts absolute values separately, then computes Δ

    The T-Learner approach is more flexible for counterfactual reasoning:
    "What would the property be if we had/hadn't made this edit?"
    """

    def __init__(self,
                 task_type: str = 'regression',
                 model_type: str = 'random_forest',
                 embedding_type: str = 'morgan',
                 **model_kwargs):
        """
        Initialize T-Learner.

        Args:
            task_type: 'regression' or 'classification'
            model_type: 'random_forest' (others can be added)
            embedding_type: Type of molecular fingerprint
            **model_kwargs: Additional arguments for models
        """
        self.task_type = task_type
        self.model_type = model_type
        self.embedding_type = embedding_type

        # Initialize featurizer
        self.featurizer = FingerprintGenerator(fp_type=embedding_type)

        # Initialize models
        default_kwargs = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        default_kwargs.update(model_kwargs)

        if task_type == 'regression':
            self.model_control = RandomForestRegressor(**default_kwargs)
            self.model_treated = RandomForestRegressor(**default_kwargs)
        elif task_type == 'classification':
            self.model_control = RandomForestClassifier(**default_kwargs)
            self.model_treated = RandomForestClassifier(**default_kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        self.is_trained = False

        logger.info(f"Initialized T-Learner ({task_type}, {model_type}, {embedding_type})")

    def featurize(self, smiles: str) -> Optional[np.ndarray]:
        """
        Convert molecule to feature vector.

        Args:
            smiles: SMILES string

        Returns:
            Feature vector or None if failed
        """
        return self.featurizer.generate(smiles)

    def train(self,
              X_control: List[str],
              y_control: List[float],
              X_treated: List[str],
              y_treated: List[float],
              cv_folds: int = 5) -> Dict[str, float]:
        """
        Train both control and treatment models.

        Args:
            X_control: SMILES for control group (pre-edit molecules)
            y_control: Property values for control group
            X_treated: SMILES for treated group (post-edit molecules)
            y_treated: Property values for treated group
            cv_folds: Number of CV folds

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training T-Learner...")
        logger.info(f"  Control group: {len(X_control)} molecules")
        logger.info(f"  Treated group: {len(X_treated)} molecules")

        # Featurize control group
        logger.info("Featurizing control group...")
        X_control_feat = self.featurizer.generate_batch(X_control)
        y_control = np.array(y_control)

        # Featurize treated group
        logger.info("Featurizing treated group...")
        X_treated_feat = self.featurizer.generate_batch(X_treated)
        y_treated = np.array(y_treated)

        # Train control model (μ₀)
        logger.info("Training control model μ₀...")
        cv_control = cross_val_score(self.model_control, X_control_feat, y_control,
                                     cv=cv_folds, scoring='r2', n_jobs=-1)
        self.model_control.fit(X_control_feat, y_control)

        control_r2 = self.model_control.score(X_control_feat, y_control)
        logger.info(f"  Control model - Train R²: {control_r2:.3f}, CV R²: {cv_control.mean():.3f}")

        # Train treated model (μ₁)
        logger.info("Training treated model μ₁...")
        cv_treated = cross_val_score(self.model_treated, X_treated_feat, y_treated,
                                     cv=cv_folds, scoring='r2', n_jobs=-1)
        self.model_treated.fit(X_treated_feat, y_treated)

        treated_r2 = self.model_treated.score(X_treated_feat, y_treated)
        logger.info(f"  Treated model - Train R²: {treated_r2:.3f}, CV R²: {cv_treated.mean():.3f}")

        self.is_trained = True

        return {
            'control_train_r2': control_r2,
            'control_cv_r2': cv_control.mean(),
            'treated_train_r2': treated_r2,
            'treated_cv_r2': cv_treated.mean(),
            'n_control': len(X_control),
            'n_treated': len(X_treated)
        }

    def predict_effect(self, X: List[str]) -> np.ndarray:
        """
        Predict causal effect for molecules.

        For each molecule, predicts:
        τ(x) = μ₁(x) - μ₀(x)

        This is the estimated treatment effect - how much the property
        would change if we applied the edit.

        Args:
            X: List of SMILES strings

        Returns:
            Array of predicted effects
        """
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return np.array([])

        # Featurize
        X_feat = self.featurizer.generate_batch(X)

        # Predict with both models
        pred_control = self.model_control.predict(X_feat)
        pred_treated = self.model_treated.predict(X_feat)

        # Compute effect
        effect = pred_treated - pred_control

        return effect

    def predict_counterfactual(self, X: List[str], treatment: int) -> np.ndarray:
        """
        Predict counterfactual outcome.

        Args:
            X: List of SMILES strings
            treatment: 0 for control, 1 for treated

        Returns:
            Array of predicted property values under the specified treatment
        """
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return np.array([])

        X_feat = self.featurizer.generate_batch(X)

        if treatment == 0:
            return self.model_control.predict(X_feat)
        elif treatment == 1:
            return self.model_treated.predict(X_feat)
        else:
            raise ValueError("treatment must be 0 or 1")

    def evaluate_effect_prediction(self,
                                   X_control_test: List[str],
                                   X_treated_test: List[str],
                                   y_control_test: List[float],
                                   y_treated_test: List[float]) -> Dict[str, float]:
        """
        Evaluate causal effect predictions.

        Assumes X_control_test and X_treated_test are matched pairs
        (same molecules before and after edit).

        Args:
            X_control_test: SMILES for pre-edit molecules
            X_treated_test: SMILES for post-edit molecules
            y_control_test: True property values pre-edit
            y_treated_test: True property values post-edit

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            logger.error("Model not trained yet!")
            return {}

        logger.info(f"Evaluating effect prediction on {len(X_control_test)} pairs...")

        # True effects
        y_control = np.array(y_control_test)
        y_treated = np.array(y_treated_test)
        true_effects = y_treated - y_control

        # Predicted effects (using control molecules as features)
        pred_effects = self.predict_effect(X_control_test)

        # Metrics
        r2 = r2_score(true_effects, pred_effects)
        mae = mean_absolute_error(true_effects, pred_effects)

        logger.info(f"Effect prediction R²: {r2:.3f}")
        logger.info(f"Effect prediction MAE: {mae:.3f}")

        # Also evaluate individual models
        X_control_feat = self.featurizer.generate_batch(X_control_test)
        X_treated_feat = self.featurizer.generate_batch(X_treated_test)

        pred_control = self.model_control.predict(X_control_feat)
        pred_treated = self.model_treated.predict(X_treated_feat)

        control_r2 = r2_score(y_control, pred_control)
        treated_r2 = r2_score(y_treated, pred_treated)

        logger.info(f"Control model R²: {control_r2:.3f}")
        logger.info(f"Treated model R²: {treated_r2:.3f}")

        return {
            'effect_r2': r2,
            'effect_mae': mae,
            'control_r2': control_r2,
            'treated_r2': treated_r2,
            'n_test': len(X_control_test)
        }

    def save(self, filepath: str) -> None:
        """
        Save trained T-Learner to disk.

        Args:
            filepath: Path to save file
        """
        if not self.is_trained:
            logger.warning("Saving untrained model!")

        save_dict = {
            'model_control': self.model_control,
            'model_treated': self.model_treated,
            'task_type': self.task_type,
            'model_type': self.model_type,
            'embedding_type': self.embedding_type,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

        logger.info(f"T-Learner saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'TLearner':
        """
        Load trained T-Learner from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded TLearner
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        learner = cls(
            task_type=save_dict['task_type'],
            model_type=save_dict['model_type'],
            embedding_type=save_dict['embedding_type']
        )

        learner.model_control = save_dict['model_control']
        learner.model_treated = save_dict['model_treated']
        learner.is_trained = save_dict['is_trained']

        logger.info(f"T-Learner loaded from {filepath}")

        return learner

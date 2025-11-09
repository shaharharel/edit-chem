"""Data preprocessing and standardization."""

import logging
import pandas as pd
import numpy as np
from typing import Optional, List
from rdkit import Chem

from src.utils.chemistry import standardize_smiles, is_valid_molecule, get_murcko_scaffold

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocess and standardize molecular data.

    Handles:
    - SMILES standardization
    - Duplicate removal
    - Outlier detection
    - Scaffold annotation
    - Data quality filtering
    """

    def __init__(self,
                 max_mw: float = 800,
                 min_atoms: int = 5,
                 max_atoms: int = 100):
        """
        Initialize preprocessor.

        Args:
            max_mw: Maximum molecular weight
            min_atoms: Minimum number of heavy atoms
            max_atoms: Maximum number of heavy atoms
        """
        self.max_mw = max_mw
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms

    def preprocess_dataframe(self, df: pd.DataFrame, smiles_col: str = 'smiles') -> pd.DataFrame:
        """
        Preprocess a DataFrame of compounds.

        Args:
            df: Input DataFrame
            smiles_col: Name of SMILES column

        Returns:
            Cleaned and standardized DataFrame
        """
        logger.info(f"Preprocessing {len(df)} compounds...")

        initial_count = len(df)

        # 1. Standardize SMILES
        df = self._standardize_smiles_column(df, smiles_col)
        logger.info(f"After standardization: {len(df)} compounds ({len(df)/initial_count*100:.1f}%)")

        # 2. Remove duplicates
        df = self._remove_duplicates(df, smiles_col)
        logger.info(f"After deduplication: {len(df)} compounds ({len(df)/initial_count*100:.1f}%)")

        # 3. Apply validity filters
        df = self._filter_valid_molecules(df, smiles_col)
        logger.info(f"After validity filters: {len(df)} compounds ({len(df)/initial_count*100:.1f}%)")

        # 4. Add scaffold information
        df = self._add_scaffold_info(df, smiles_col)

        logger.info(f"Preprocessing complete: {len(df)} compounds retained")

        return df

    def detect_outliers(self,
                       df: pd.DataFrame,
                       value_col: str,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and optionally remove outliers.

        Args:
            df: Input DataFrame
            value_col: Column containing values to check
            method: 'iqr' (interquartile range) or 'zscore'
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers marked/removed
        """
        logger.info(f"Detecting outliers in '{value_col}' using {method} method...")

        values = df[value_col].dropna()

        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = (df[value_col] < lower_bound) | (df[value_col] > upper_bound)

        elif method == 'zscore':
            mean = values.mean()
            std = values.std()

            z_scores = np.abs((df[value_col] - mean) / std)
            outliers = z_scores > threshold

        else:
            logger.error(f"Unknown outlier detection method: {method}")
            return df

        n_outliers = outliers.sum()
        logger.info(f"Found {n_outliers} outliers ({n_outliers/len(df)*100:.1f}%)")

        # Mark outliers
        df['is_outlier'] = outliers

        return df

    def filter_by_property_range(self,
                                 df: pd.DataFrame,
                                 property_col: str,
                                 min_val: Optional[float] = None,
                                 max_val: Optional[float] = None) -> pd.DataFrame:
        """
        Filter compounds by property value range.

        Args:
            df: Input DataFrame
            property_col: Column name
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)

        if min_val is not None:
            df = df[df[property_col] >= min_val]

        if max_val is not None:
            df = df[df[property_col] <= max_val]

        logger.info(f"Filtered {property_col}: {len(df)} compounds ({len(df)/initial_count*100:.1f}%)")

        return df

    def _standardize_smiles_column(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """Standardize all SMILES in a column."""
        df = df.copy()

        # Apply standardization
        df['standardized_smiles'] = df[smiles_col].apply(standardize_smiles)

        # Remove rows where standardization failed
        df = df.dropna(subset=['standardized_smiles'])

        # Replace original SMILES
        df[smiles_col] = df['standardized_smiles']
        df = df.drop(columns=['standardized_smiles'])

        return df

    def _remove_duplicates(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """Remove duplicate molecules based on SMILES."""
        initial_count = len(df)

        df = df.drop_duplicates(subset=[smiles_col])

        n_removed = initial_count - len(df)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} duplicate molecules")

        return df

    def _filter_valid_molecules(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """Apply validity filters."""
        df = df.copy()

        # Check validity
        df['is_valid'] = df[smiles_col].apply(
            lambda smi: is_valid_molecule(smi, self.max_mw, self.min_atoms, self.max_atoms)
        )

        n_invalid = (~df['is_valid']).sum()
        logger.info(f"Filtering {n_invalid} invalid molecules")

        df = df[df['is_valid']].drop(columns=['is_valid'])

        return df

    def _add_scaffold_info(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """Add Murcko scaffold information."""
        df = df.copy()

        logger.info("Computing Murcko scaffolds...")
        df['scaffold'] = df[smiles_col].apply(get_murcko_scaffold)

        # Count scaffold frequency
        scaffold_counts = df['scaffold'].value_counts()
        df['scaffold_frequency'] = df['scaffold'].map(scaffold_counts)

        logger.info(f"Found {df['scaffold'].nunique()} unique scaffolds")

        return df

    def split_by_scaffold(self,
                         df: pd.DataFrame,
                         test_fraction: float = 0.2,
                         random_state: int = 42) -> tuple:
        """
        Split data by scaffold for train/test.

        Ensures compounds with same scaffold are in same split,
        avoiding data leakage.

        Args:
            df: DataFrame with 'scaffold' column
            test_fraction: Fraction for test set
            random_state: Random seed

        Returns:
            (train_df, test_df) tuple
        """
        if 'scaffold' not in df.columns:
            logger.error("DataFrame must have 'scaffold' column")
            return df, pd.DataFrame()

        np.random.seed(random_state)

        # Get unique scaffolds
        unique_scaffolds = df['scaffold'].unique()
        np.random.shuffle(unique_scaffolds)

        # Split scaffolds
        n_test_scaffolds = int(len(unique_scaffolds) * test_fraction)
        test_scaffolds = set(unique_scaffolds[:n_test_scaffolds])

        # Split data
        test_df = df[df['scaffold'].isin(test_scaffolds)]
        train_df = df[~df['scaffold'].isin(test_scaffolds)]

        logger.info(f"Scaffold split: {len(train_df)} train, {len(test_df)} test")

        return train_df, test_df

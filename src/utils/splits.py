"""
Challenging molecular data splitting strategies for property prediction.

This module implements various splitting strategies designed to test model
generalization in realistic scenarios:
- Scaffold split: Test on novel chemical scaffolds
- Target split: Test on novel biological targets
- Butina clustering: Test on diverse molecular clusters
- Property-stratified: Test across property value ranges
- Temporal split: Simulate chronological discovery
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from collections import defaultdict, Counter
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import train_test_split
import warnings


class MolecularSplitter:
    """Base class for molecular data splitting strategies."""

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42
    ):
        """
        Initialize splitter.

        Args:
            train_size: Fraction for training set
            val_size: Fraction for validation set
            test_size: Fraction for test set
            random_state: Random seed for reproducibility
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            f"Sizes must sum to 1.0, got {train_size + val_size + test_size}"

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'smiles'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train/val/test sets.

        Args:
            df: DataFrame with molecular data
            smiles_col: Column name containing SMILES strings

        Returns:
            train, val, test DataFrames
        """
        raise NotImplementedError("Subclasses must implement split()")

    def _split_indices_to_dataframes(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert indices to train/val/test DataFrames."""
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[val_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True)
        )


class RandomSplitter(MolecularSplitter):
    """Random split - baseline splitting strategy."""

    def split(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'smiles'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Randomly split data."""
        n = len(df)
        indices = np.arange(n)

        np.random.seed(self.random_state)
        np.random.shuffle(indices)

        train_end = int(n * self.train_size)
        val_end = train_end + int(n * self.val_size)

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return self._split_indices_to_dataframes(df, train_idx, val_idx, test_idx)


class ScaffoldSplitter(MolecularSplitter):
    """
    Scaffold split using Bemis-Murcko scaffolds.

    Ensures that molecules with the same core scaffold are in the same split.
    Tests model generalization to novel scaffolds (most challenging split).

    Reference:
        Bemis & Murcko (1996) "The Properties of Known Drugs"
        https://pubs.acs.org/doi/10.1021/jm9602928
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        use_generic: bool = True
    ):
        """
        Initialize scaffold splitter.

        Args:
            use_generic: If True, use generic scaffolds (remove atom types).
                        Generic scaffolds group more molecules together.
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.use_generic = use_generic

    def _get_scaffold(self, smiles: str) -> Optional[str]:
        """Extract Bemis-Murcko scaffold from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            scaffold = MurckoScaffold.GetScaffoldForMol(mol)

            if self.use_generic:
                scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)

            return Chem.MolToSmiles(scaffold)
        except:
            return None

    def split(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'smiles'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by Bemis-Murcko scaffolds.

        Strategy:
        1. Group molecules by scaffold
        2. Sort scaffolds by size (largest first)
        3. Assign scaffolds to splits to balance sizes
        """
        print(f"Computing Bemis-Murcko scaffolds {'(generic)' if self.use_generic else '(specific)'}...")

        # Compute scaffolds
        scaffolds = df[smiles_col].apply(self._get_scaffold)

        # Group indices by scaffold
        scaffold_to_indices = defaultdict(list)
        for idx, scaffold in enumerate(scaffolds):
            if scaffold is not None:
                scaffold_to_indices[scaffold].append(idx)

        # Sort scaffolds by size (descending)
        scaffold_sizes = [(scaffold, len(indices))
                         for scaffold, indices in scaffold_to_indices.items()]
        scaffold_sizes.sort(key=lambda x: x[1], reverse=True)

        print(f"  Found {len(scaffold_sizes)} unique scaffolds")
        print(f"  Largest scaffold: {scaffold_sizes[0][1]} molecules")
        print(f"  Smallest scaffold: {scaffold_sizes[-1][1]} molecules")

        # Allocate scaffolds to splits (greedy algorithm)
        n = len(df)
        target_train = int(n * self.train_size)
        target_val = int(n * self.val_size)

        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count, test_count = 0, 0, 0

        np.random.seed(self.random_state)
        np.random.shuffle(scaffold_sizes)

        for scaffold, size in scaffold_sizes:
            indices = scaffold_to_indices[scaffold]

            # Assign to split with most room
            if train_count < target_train:
                train_idx.extend(indices)
                train_count += size
            elif val_count < target_val:
                val_idx.extend(indices)
                val_count += size
            else:
                test_idx.extend(indices)
                test_count += size

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class TargetSplitter(MolecularSplitter):
    """
    Split by biological target (for ChEMBL bioactivity data).

    Ensures that molecules tested on the same target are in the same split.
    Tests model generalization to novel targets.
    """

    def split(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_id',
        smiles_col: str = 'smiles'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by biological target.

        Args:
            df: DataFrame with columns [smiles, target_id, ...]
            target_col: Column name for target identifier
        """
        if target_col not in df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in DataFrame. "
                f"Available columns: {df.columns.tolist()}"
            )

        print(f"Splitting by biological target ({target_col})...")

        # Group by target
        target_to_indices = defaultdict(list)
        for idx, target in enumerate(df[target_col]):
            target_to_indices[target].append(idx)

        # Sort targets by size
        target_sizes = [(target, len(indices))
                       for target, indices in target_to_indices.items()]
        target_sizes.sort(key=lambda x: x[1], reverse=True)

        print(f"  Found {len(target_sizes)} unique targets")
        print(f"  Largest target: {target_sizes[0][1]} molecules")

        # Allocate targets to splits
        n = len(df)
        target_train = int(n * self.train_size)
        target_val = int(n * self.val_size)

        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count = 0, 0

        np.random.seed(self.random_state)
        np.random.shuffle(target_sizes)

        for target, size in target_sizes:
            indices = target_to_indices[target]

            if train_count < target_train:
                train_idx.extend(indices)
                train_count += size
            elif val_count < target_val:
                val_idx.extend(indices)
                val_count += size
            else:
                test_idx.extend(indices)

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class ButinaSplitter(MolecularSplitter):
    """
    Split using Butina clustering of molecular fingerprints.

    Groups similar molecules into clusters, then assigns clusters to splits.
    Tests generalization to diverse molecular clusters.

    Reference:
        Butina (1999) "Unsupervised Data Base Clustering Based on Daylight's
        Fingerprint and Tanimoto Similarity"
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        cutoff: float = 0.35,
        fp_radius: int = 2,
        fp_size: int = 2048
    ):
        """
        Initialize Butina splitter.

        Args:
            cutoff: Distance cutoff for clustering (0-1)
                   Lower = more clusters (stricter similarity)
            fp_radius: Morgan fingerprint radius
            fp_size: Fingerprint size
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.cutoff = cutoff
        self.fp_radius = fp_radius
        self.fp_size = fp_size

    def _get_fingerprint(self, smiles: str):
        """Compute Morgan fingerprint."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fp_radius, nBits=self.fp_size
            )
        except:
            return None

    def _cluster_fingerprints(self, fps):
        """Cluster fingerprints using Butina algorithm."""
        from rdkit import DataStructs

        # Compute distance matrix
        n = len(fps)
        dists = []
        for i in range(n):
            for j in range(i):
                # Tanimoto similarity -> distance
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dists.append(1 - sim)

        # Cluster
        clusters = Butina.ClusterData(dists, n, self.cutoff, isDistData=True)
        return clusters

    def split(
        self,
        df: pd.DataFrame,
        smiles_col: str = 'smiles'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by Butina clustering."""
        print(f"Computing Morgan fingerprints (r={self.fp_radius}, {self.fp_size} bits)...")

        # Compute fingerprints
        fps = [self._get_fingerprint(smi) for smi in df[smiles_col]]
        valid_mask = [fp is not None for fp in fps]
        fps = [fp for fp in fps if fp is not None]

        if len(fps) < len(df):
            warnings.warn(f"Failed to compute {len(df) - len(fps)} fingerprints")

        print(f"Clustering molecules (cutoff={self.cutoff})...")
        clusters = self._cluster_fingerprints(fps)

        print(f"  Found {len(clusters)} clusters")
        print(f"  Largest cluster: {len(clusters[0])} molecules")

        # Map clusters to original indices
        valid_indices = np.where(valid_mask)[0]
        cluster_to_indices = defaultdict(list)
        for cluster_id, cluster in enumerate(clusters):
            for local_idx in cluster:
                original_idx = valid_indices[local_idx]
                cluster_to_indices[cluster_id].append(original_idx)

        # Sort clusters by size
        cluster_sizes = [(cid, len(indices))
                        for cid, indices in cluster_to_indices.items()]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        # Allocate clusters to splits
        n = len(df)
        target_train = int(n * self.train_size)
        target_val = int(n * self.val_size)

        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count = 0, 0

        np.random.seed(self.random_state)
        np.random.shuffle(cluster_sizes)

        for cluster_id, size in cluster_sizes:
            indices = cluster_to_indices[cluster_id]

            if train_count < target_train:
                train_idx.extend(indices)
                train_count += size
            elif val_count < target_val:
                val_idx.extend(indices)
                val_count += size
            else:
                test_idx.extend(indices)

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class PropertyStratifiedSplitter(MolecularSplitter):
    """
    Stratified split by property value ranges.

    Ensures balanced distribution of property values across train/val/test.
    Useful for regression tasks with skewed property distributions.
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        n_bins: int = 5
    ):
        """
        Initialize property stratified splitter.

        Args:
            n_bins: Number of bins for stratification
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.n_bins = n_bins

    def split(
        self,
        df: pd.DataFrame,
        property_col: str = 'property_value',
        smiles_col: str = 'smiles'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split with stratification by property values.

        Args:
            df: DataFrame with molecular data
            property_col: Column name for property values
        """
        if property_col not in df.columns:
            raise ValueError(
                f"property_col '{property_col}' not found. "
                f"Available: {df.columns.tolist()}"
            )

        print(f"Stratifying by property values ({property_col})...")

        # Create bins
        property_values = df[property_col].values
        bins = np.percentile(property_values, np.linspace(0, 100, self.n_bins + 1))
        bin_labels = np.digitize(property_values, bins[1:-1])

        print(f"  Property range: [{property_values.min():.2f}, {property_values.max():.2f}]")
        print(f"  Created {self.n_bins} bins")

        # Split within each bin
        train_idx, val_idx, test_idx = [], [], []

        for bin_id in range(self.n_bins):
            bin_mask = bin_labels == bin_id
            bin_indices = np.where(bin_mask)[0]

            if len(bin_indices) < 3:
                # Not enough samples, put all in train
                train_idx.extend(bin_indices)
                continue

            # Split this bin
            n_bin = len(bin_indices)
            n_train = int(n_bin * self.train_size)
            n_val = int(n_bin * self.val_size)

            np.random.seed(self.random_state + bin_id)
            np.random.shuffle(bin_indices)

            train_idx.extend(bin_indices[:n_train])
            val_idx.extend(bin_indices[n_train:n_train + n_val])
            test_idx.extend(bin_indices[n_train + n_val:])

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class TemporalSplitter(MolecularSplitter):
    """
    Temporal split (chronological).

    Simulates realistic scenario where model is trained on older data
    and tested on newer data.
    """

    def split(
        self,
        df: pd.DataFrame,
        time_col: str = 'timestamp',
        smiles_col: str = 'smiles'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split chronologically by timestamp.

        Args:
            df: DataFrame with molecular data
            time_col: Column name for timestamp/date
        """
        if time_col not in df.columns:
            raise ValueError(
                f"time_col '{time_col}' not found. "
                f"Available: {df.columns.tolist()}"
            )

        print(f"Splitting chronologically by {time_col}...")

        # Sort by time
        df_sorted = df.sort_values(time_col).reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * self.train_size)
        val_end = train_end + int(n * self.val_size)

        train = df_sorted.iloc[:train_end].copy()
        val = df_sorted.iloc[train_end:val_end].copy()
        test = df_sorted.iloc[val_end:].copy()

        print(f"  Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

        return train, val, test


def get_splitter(
    split_type: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: Optional[int] = 42,
    **kwargs
) -> MolecularSplitter:
    """
    Factory function to get a splitter by name.

    Args:
        split_type: One of ['random', 'scaffold', 'target', 'butina',
                    'stratified', 'temporal']
        train_size, val_size, test_size: Split fractions
        random_state: Random seed
        **kwargs: Additional arguments for specific splitters

    Returns:
        MolecularSplitter instance

    Example:
        >>> splitter = get_splitter('scaffold', use_generic=True)
        >>> train, val, test = splitter.split(df, smiles_col='smiles')
    """
    splitters = {
        'random': RandomSplitter,
        'scaffold': ScaffoldSplitter,
        'target': TargetSplitter,
        'butina': ButinaSplitter,
        'stratified': PropertyStratifiedSplitter,
        'temporal': TemporalSplitter
    }

    if split_type not in splitters:
        raise ValueError(
            f"Unknown split_type '{split_type}'. "
            f"Available: {list(splitters.keys())}"
        )

    splitter_class = splitters[split_type]
    return splitter_class(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        **kwargs
    )

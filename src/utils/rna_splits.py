"""
RNA-specific data splitting strategies for sequence-based prediction.

This module implements splitting strategies designed for RNA sequences,
analogous to the molecular splitters but using sequence-based similarity
and regulatory motif features.

Strategies:
- Random split: Baseline random splitting
- Sequence similarity split: Split by k-mer similarity (analogous to scaffold split)
- Motif-based split: Split by regulatory motif presence (Kozak, uAUG, uORF)
- Edit type split: Split by type of edit (SNV, insertion, deletion)
- GC content stratified: Split with stratification by GC content
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Set
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import warnings


class RNASplitter(ABC):
    """Base class for RNA sequence data splitting strategies."""

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

    @abstractmethod
    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train/val/test sets.

        Args:
            df: DataFrame with RNA sequence data
            seq_col: Column name containing sequences

        Returns:
            train, val, test DataFrames
        """
        pass

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


class RandomRNASplitter(RNASplitter):
    """Random split - baseline splitting strategy for RNA data."""

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
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


class SequenceSimilaritySplitter(RNASplitter):
    """
    Split by sequence similarity using k-mer overlap.

    Analogous to scaffold split for molecules - ensures that similar
    sequences are in the same split to test generalization to novel
    sequence contexts.

    Uses k-mer frequency vectors and clustering to group similar sequences.
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        kmer_size: int = 4,
        similarity_threshold: float = 0.7,
        max_cluster_size: int = 1000
    ):
        """
        Initialize sequence similarity splitter.

        Args:
            kmer_size: K-mer size for computing sequence similarity
            similarity_threshold: Minimum k-mer overlap for clustering (0-1)
            max_cluster_size: Maximum cluster size before breaking up
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.kmer_size = kmer_size
        self.similarity_threshold = similarity_threshold
        self.max_cluster_size = max_cluster_size

    def _get_kmer_set(self, seq: str) -> Set[str]:
        """Extract k-mers from sequence."""
        seq = seq.upper().replace('T', 'U')
        kmers = set()
        for i in range(len(seq) - self.kmer_size + 1):
            kmer = seq[i:i + self.kmer_size]
            if all(c in 'ACGU' for c in kmer):
                kmers.add(kmer)
        return kmers

    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two k-mer sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _cluster_sequences(
        self,
        sequences: List[str],
        kmer_sets: List[Set[str]]
    ) -> List[List[int]]:
        """
        Cluster sequences by k-mer similarity using greedy clustering.

        Returns list of clusters, where each cluster is a list of indices.
        """
        n = len(sequences)
        clustered = [False] * n
        clusters = []

        # Sort by sequence length (longer sequences first as cluster centers)
        sorted_indices = sorted(range(n), key=lambda i: len(sequences[i]), reverse=True)

        for center_idx in sorted_indices:
            if clustered[center_idx]:
                continue

            # Start new cluster
            cluster = [center_idx]
            clustered[center_idx] = True
            center_kmers = kmer_sets[center_idx]

            # Find similar sequences
            for other_idx in range(n):
                if clustered[other_idx]:
                    continue

                similarity = self._jaccard_similarity(center_kmers, kmer_sets[other_idx])

                if similarity >= self.similarity_threshold:
                    cluster.append(other_idx)
                    clustered[other_idx] = True

                    if len(cluster) >= self.max_cluster_size:
                        break

            clusters.append(cluster)

        return clusters

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by sequence similarity clustering.

        Strategy:
        1. Compute k-mer sets for each sequence
        2. Cluster by k-mer overlap
        3. Assign clusters to splits
        """
        print(f"Computing {self.kmer_size}-mer sets for sequence similarity clustering...")

        # Get unique sequences and their indices
        if 'seq_b' in df.columns:
            # For pairs data, consider both sequences
            all_seqs = list(set(df[seq_col].tolist() + df['seq_b'].tolist()))
        else:
            all_seqs = df[seq_col].unique().tolist()

        # Compute k-mer sets
        kmer_sets = [self._get_kmer_set(seq) for seq in all_seqs]

        print(f"  Processing {len(all_seqs)} unique sequences...")

        # Cluster sequences
        clusters = self._cluster_sequences(all_seqs, kmer_sets)

        print(f"  Found {len(clusters)} sequence clusters")
        if clusters:
            print(f"  Largest cluster: {len(clusters[0])} sequences")
            print(f"  Smallest cluster: {len(clusters[-1])} sequences")

        # Map clusters back to DataFrame rows
        seq_to_cluster = {}
        for cluster_id, cluster in enumerate(clusters):
            for seq_idx in cluster:
                seq_to_cluster[all_seqs[seq_idx]] = cluster_id

        # Assign rows to clusters (use seq_a as primary)
        row_to_cluster = []
        for _, row in df.iterrows():
            seq = row[seq_col]
            cluster_id = seq_to_cluster.get(seq, -1)
            row_to_cluster.append(cluster_id)

        # Group rows by cluster
        cluster_to_rows = defaultdict(list)
        for row_idx, cluster_id in enumerate(row_to_cluster):
            cluster_to_rows[cluster_id].append(row_idx)

        # Sort clusters by size
        cluster_sizes = [(cid, len(rows)) for cid, rows in cluster_to_rows.items()]
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
            rows = cluster_to_rows[cluster_id]

            if train_count < target_train:
                train_idx.extend(rows)
                train_count += size
            elif val_count < target_val:
                val_idx.extend(rows)
                val_count += size
            else:
                test_idx.extend(rows)

        print(f"  Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        return self._split_indices_to_dataframes(
            df,
            np.array(train_idx),
            np.array(val_idx),
            np.array(test_idx)
        )


class MotifSplitter(RNASplitter):
    """
    Split by regulatory motif presence.

    Groups sequences by presence/absence of regulatory motifs:
    - Kozak consensus
    - Upstream AUG (uAUG)
    - Upstream open reading frames (uORFs)
    - AU-rich elements (AREs)

    Ensures that sequences with similar regulatory context are in same split.
    """

    # Kozak consensus pattern (simplified)
    KOZAK_PATTERN = 'ACCAUGG'  # Strong Kozak

    # AU-rich element patterns
    ARE_PATTERNS = ['AUUUA', 'UUAUUUAUU']

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        motif_types: List[str] = None
    ):
        """
        Initialize motif splitter.

        Args:
            motif_types: List of motif types to consider.
                        Options: ['kozak', 'uaug', 'uorf', 'are', 'gc_rich']
                        Default: all
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.motif_types = motif_types or ['kozak', 'uaug', 'uorf', 'are']

    def _has_kozak(self, seq: str) -> bool:
        """Check for Kozak consensus."""
        seq = seq.upper().replace('T', 'U')
        # Look for strong Kozak or partial matches
        if 'ACCAUGG' in seq or 'GCCAUGG' in seq:
            return True
        # Weaker check: A/G at -3 and G at +4
        for i in range(len(seq) - 6):
            if seq[i:i+3] == 'AUG':
                if i >= 3 and seq[i-3] in 'AG':
                    return True
        return False

    def _has_uaug(self, seq: str, main_aug_pos: int = None) -> bool:
        """Check for upstream AUG."""
        seq = seq.upper().replace('T', 'U')
        # If main AUG position not specified, assume it's the first one
        if main_aug_pos is None:
            main_aug_pos = seq.find('AUG')
            if main_aug_pos == -1:
                return False

        # Look for AUG before main position
        for i in range(main_aug_pos):
            if seq[i:i+3] == 'AUG':
                return True
        return False

    def _has_uorf(self, seq: str) -> bool:
        """Check for upstream ORF (AUG followed by in-frame stop)."""
        seq = seq.upper().replace('T', 'U')
        stop_codons = {'UAA', 'UAG', 'UGA'}

        for i in range(len(seq) - 5):
            if seq[i:i+3] == 'AUG':
                # Check for in-frame stop codon
                for j in range(i + 3, len(seq) - 2, 3):
                    if seq[j:j+3] in stop_codons:
                        return True
        return False

    def _has_are(self, seq: str) -> bool:
        """Check for AU-rich elements."""
        seq = seq.upper().replace('T', 'U')
        for pattern in self.ARE_PATTERNS:
            if pattern in seq:
                return True
        return False

    def _is_gc_rich(self, seq: str, threshold: float = 0.6) -> bool:
        """Check if sequence is GC-rich."""
        seq = seq.upper().replace('T', 'U')
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq) >= threshold if len(seq) > 0 else False

    def _get_motif_signature(self, seq: str) -> tuple:
        """Get motif signature for sequence."""
        features = []

        if 'kozak' in self.motif_types:
            features.append(self._has_kozak(seq))
        if 'uaug' in self.motif_types:
            features.append(self._has_uaug(seq))
        if 'uorf' in self.motif_types:
            features.append(self._has_uorf(seq))
        if 'are' in self.motif_types:
            features.append(self._has_are(seq))
        if 'gc_rich' in self.motif_types:
            features.append(self._is_gc_rich(seq))

        return tuple(features)

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by motif signature.

        Strategy:
        1. Compute motif signature for each sequence
        2. Group by signature
        3. Assign signature groups to splits
        """
        print(f"Computing motif signatures ({', '.join(self.motif_types)})...")

        # Compute signatures
        signatures = df[seq_col].apply(self._get_motif_signature)

        # Group by signature
        sig_to_indices = defaultdict(list)
        for idx, sig in enumerate(signatures):
            sig_to_indices[sig].append(idx)

        # Sort by group size
        sig_sizes = [(sig, len(indices)) for sig, indices in sig_to_indices.items()]
        sig_sizes.sort(key=lambda x: x[1], reverse=True)

        print(f"  Found {len(sig_sizes)} unique motif signatures")
        print(f"  Largest group: {sig_sizes[0][1]} sequences")

        # Show distribution
        for sig, size in sig_sizes[:5]:
            features = []
            for i, motif_type in enumerate(self.motif_types):
                if sig[i]:
                    features.append(f"+{motif_type}")
                else:
                    features.append(f"-{motif_type}")
            print(f"    {' '.join(features)}: {size} sequences")

        # Allocate groups to splits
        n = len(df)
        target_train = int(n * self.train_size)
        target_val = int(n * self.val_size)

        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count = 0, 0

        np.random.seed(self.random_state)
        np.random.shuffle(sig_sizes)

        for sig, size in sig_sizes:
            indices = sig_to_indices[sig]

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


class EditTypeSplitter(RNASplitter):
    """
    Split by type of edit between sequence pairs.

    For paired data (seq_a → seq_b), groups by:
    - Edit type (SNV, insertion, deletion, multiple)
    - Edit position (5' region, middle, 3' region)
    - Nucleotide change type (transition vs transversion)

    Tests model generalization to different edit types.
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        position_bins: int = 3
    ):
        """
        Initialize edit type splitter.

        Args:
            position_bins: Number of position bins (default: 3 for 5'/mid/3')
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.position_bins = position_bins

    def _classify_edit(self, seq_a: str, seq_b: str) -> tuple:
        """
        Classify the edit between two sequences.

        Returns:
            (edit_type, position_bin, nuc_change_type)
        """
        seq_a = seq_a.upper().replace('T', 'U')
        seq_b = seq_b.upper().replace('T', 'U')

        len_a = len(seq_a)
        len_b = len(seq_b)

        # Determine edit type based on length
        if len_a == len_b:
            # Count differences
            diffs = [(i, seq_a[i], seq_b[i]) for i in range(len_a) if seq_a[i] != seq_b[i]]

            if len(diffs) == 0:
                return ('identical', 0, 'none')
            elif len(diffs) == 1:
                edit_type = 'snv'
                pos = diffs[0][0]
                old_nuc, new_nuc = diffs[0][1], diffs[0][2]
            else:
                edit_type = 'multiple_snv'
                pos = diffs[0][0]  # Use first diff position
                old_nuc, new_nuc = diffs[0][1], diffs[0][2]

        elif len_b > len_a:
            edit_type = 'insertion'
            pos = 0  # Simplified - would need alignment
            old_nuc, new_nuc = 'N', 'N'

        else:
            edit_type = 'deletion'
            pos = 0
            old_nuc, new_nuc = 'N', 'N'

        # Determine position bin
        ref_len = max(len_a, len_b)
        if ref_len > 0:
            rel_pos = pos / ref_len
            position_bin = min(int(rel_pos * self.position_bins), self.position_bins - 1)
        else:
            position_bin = 0

        # Classify nucleotide change (for SNVs)
        purines = {'A', 'G'}
        pyrimidines = {'C', 'U'}

        if edit_type in ['snv', 'multiple_snv']:
            if (old_nuc in purines and new_nuc in purines) or \
               (old_nuc in pyrimidines and new_nuc in pyrimidines):
                nuc_change = 'transition'
            else:
                nuc_change = 'transversion'
        else:
            nuc_change = 'indel'

        return (edit_type, position_bin, nuc_change)

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by edit type classification.

        Requires DataFrame with 'seq_a' and 'seq_b' columns.
        """
        if 'seq_b' not in df.columns:
            raise ValueError("EditTypeSplitter requires 'seq_b' column for pair data")

        print(f"Classifying edits for {len(df)} pairs...")

        # Classify all edits
        edit_classes = []
        for _, row in df.iterrows():
            edit_class = self._classify_edit(row[seq_col], row['seq_b'])
            edit_classes.append(edit_class)

        # Group by edit class
        class_to_indices = defaultdict(list)
        for idx, edit_class in enumerate(edit_classes):
            class_to_indices[edit_class].append(idx)

        # Show distribution
        print(f"  Found {len(class_to_indices)} unique edit classes")
        class_sizes = [(cls, len(indices)) for cls, indices in class_to_indices.items()]
        class_sizes.sort(key=lambda x: x[1], reverse=True)

        for cls, size in class_sizes[:10]:
            edit_type, pos_bin, nuc_change = cls
            pos_label = ['5\'', 'mid', '3\''][pos_bin] if pos_bin < 3 else str(pos_bin)
            print(f"    {edit_type}/{pos_label}/{nuc_change}: {size}")

        # Allocate classes to splits
        n = len(df)
        target_train = int(n * self.train_size)
        target_val = int(n * self.val_size)

        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count = 0, 0

        np.random.seed(self.random_state)
        np.random.shuffle(class_sizes)

        for cls, size in class_sizes:
            indices = class_to_indices[cls]

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


class GCStratifiedSplitter(RNASplitter):
    """
    Stratified split by GC content.

    Ensures balanced GC content distribution across train/val/test.
    Important for avoiding GC-biased models.
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
        Initialize GC stratified splitter.

        Args:
            n_bins: Number of GC content bins
        """
        super().__init__(train_size, val_size, test_size, random_state)
        self.n_bins = n_bins

    def _compute_gc_content(self, seq: str) -> float:
        """Compute GC content of sequence."""
        seq = seq.upper().replace('T', 'U')
        if len(seq) == 0:
            return 0.5
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq)

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split with stratification by GC content."""
        print(f"Stratifying by GC content ({seq_col})...")

        # Compute GC content
        gc_values = df[seq_col].apply(self._compute_gc_content).values

        print(f"  GC range: [{gc_values.min():.2%}, {gc_values.max():.2%}]")
        print(f"  Mean GC: {gc_values.mean():.2%}")

        # Create bins
        bins = np.percentile(gc_values, np.linspace(0, 100, self.n_bins + 1))
        bin_labels = np.digitize(gc_values, bins[1:-1])

        print(f"  Created {self.n_bins} GC bins")

        # Split within each bin
        train_idx, val_idx, test_idx = [], [], []

        for bin_id in range(self.n_bins):
            bin_mask = bin_labels == bin_id
            bin_indices = np.where(bin_mask)[0]

            if len(bin_indices) < 3:
                train_idx.extend(bin_indices)
                continue

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


class LengthStratifiedSplitter(RNASplitter):
    """
    Stratified split by sequence length.

    Ensures balanced length distribution, important when sequences
    have varying lengths (e.g., UTRs of different sizes).
    """

    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: Optional[int] = 42,
        n_bins: int = 5
    ):
        super().__init__(train_size, val_size, test_size, random_state)
        self.n_bins = n_bins

    def split(
        self,
        df: pd.DataFrame,
        seq_col: str = 'seq_a'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split with stratification by sequence length."""
        print(f"Stratifying by sequence length ({seq_col})...")

        lengths = df[seq_col].str.len().values

        print(f"  Length range: [{lengths.min()}, {lengths.max()}]")
        print(f"  Mean length: {lengths.mean():.1f}")

        # Create bins
        bins = np.percentile(lengths, np.linspace(0, 100, self.n_bins + 1))
        bin_labels = np.digitize(lengths, bins[1:-1])

        # Split within each bin
        train_idx, val_idx, test_idx = [], [], []

        for bin_id in range(self.n_bins):
            bin_mask = bin_labels == bin_id
            bin_indices = np.where(bin_mask)[0]

            if len(bin_indices) < 3:
                train_idx.extend(bin_indices)
                continue

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


def get_rna_splitter(
    split_type: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: Optional[int] = 42,
    **kwargs
) -> RNASplitter:
    """
    Factory function to get an RNA splitter by name.

    Args:
        split_type: One of ['random', 'sequence_similarity', 'motif',
                    'edit_type', 'gc_stratified', 'length_stratified']
        train_size, val_size, test_size: Split fractions
        random_state: Random seed
        **kwargs: Additional arguments for specific splitters
                 - sequence_similarity: kmer_size, similarity_threshold
                 - motif: motif_types
                 - edit_type: position_bins
                 - gc_stratified: n_bins
                 - length_stratified: n_bins

    Returns:
        RNASplitter instance

    Example:
        >>> splitter = get_rna_splitter('sequence_similarity', kmer_size=5)
        >>> train, val, test = splitter.split(df, seq_col='seq_a')

        >>> splitter = get_rna_splitter('motif', motif_types=['kozak', 'uaug'])
        >>> train, val, test = splitter.split(df)
    """
    splitters = {
        'random': RandomRNASplitter,
        'sequence_similarity': SequenceSimilaritySplitter,
        'motif': MotifSplitter,
        'edit_type': EditTypeSplitter,
        'gc_stratified': GCStratifiedSplitter,
        'length_stratified': LengthStratifiedSplitter
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

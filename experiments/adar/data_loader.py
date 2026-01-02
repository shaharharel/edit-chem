"""
Data loading and preprocessing for ADAR editing prediction experiments.

Handles:
- Loading the ADAR editing dataset with secondary structure
- Feature extraction from sequence and structure
- Train/val/test splitting with various strategies
- Data sampling for balanced training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ADARDataConfig:
    """Configuration for ADAR data loading."""
    data_path: str = "data/rna/adar/adar_editing_dataset_with_structure.csv"
    n_positives: Optional[int] = None  # None = all positives
    n_negatives: int = 500_000  # Number of negative samples
    random_state: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


def load_adar_data(config: ADARDataConfig) -> pd.DataFrame:
    """
    Load ADAR editing dataset.

    Args:
        config: Data configuration

    Returns:
        DataFrame with sampled data
    """
    logger.info(f"Loading data from {config.data_path}")
    df = pd.read_csv(config.data_path)

    logger.info(f"Full dataset: {len(df):,} samples")
    logger.info(f"  Positives: {(df['is_edited']==1).sum():,}")
    logger.info(f"  Negatives: {(df['is_edited']==0).sum():,}")

    # Sample data
    positives = df[df['is_edited'] == 1]
    negatives = df[df['is_edited'] == 0]

    if config.n_positives is not None:
        positives = positives.sample(
            n=min(config.n_positives, len(positives)),
            random_state=config.random_state
        )

    if config.n_negatives < len(negatives):
        negatives = negatives.sample(
            n=config.n_negatives,
            random_state=config.random_state
        )

    df_sampled = pd.concat([positives, negatives], ignore_index=True)
    df_sampled = df_sampled.sample(frac=1, random_state=config.random_state).reset_index(drop=True)

    logger.info(f"Sampled dataset: {len(df_sampled):,} samples")
    logger.info(f"  Positives: {(df_sampled['is_edited']==1).sum():,}")
    logger.info(f"  Negatives: {(df_sampled['is_edited']==0).sum():,}")

    return df_sampled


def extract_structure_features(
    structures: pd.Series,
    window_sizes: List[int] = [5, 10, 20, 50],
    center_idx: int = 100
) -> pd.DataFrame:
    """
    Extract features from dot-bracket structure notation.

    Features extracted:
    - center_paired: Is center position paired (from data)
    - paired_fraction_*: Fraction of paired positions in window
    - stem_at_center: Is center in a continuous stem
    - stem_length: Length of stem at center (if in stem)
    - loop_distance: Distance to nearest loop
    - local_gc_structure_ratio: Ratio of stems in local region

    Args:
        structures: Series of dot-bracket structures
        window_sizes: Windows around center for feature extraction
        center_idx: Index of center position (default 100 for 201nt)

    Returns:
        DataFrame with structure features
    """
    features = []

    for struct in structures:
        feat = {}

        # Basic center features
        if center_idx < len(struct):
            center_char = struct[center_idx]
            feat['center_is_paired'] = 1 if center_char in '()' else 0
            feat['center_is_opening'] = 1 if center_char == '(' else 0
            feat['center_is_closing'] = 1 if center_char == ')' else 0
        else:
            feat['center_is_paired'] = 0
            feat['center_is_opening'] = 0
            feat['center_is_closing'] = 0

        # Paired fraction in different windows
        for ws in window_sizes:
            start = max(0, center_idx - ws)
            end = min(len(struct), center_idx + ws + 1)
            window = struct[start:end]

            n_paired = sum(1 for c in window if c in '()')
            feat[f'paired_frac_w{ws}'] = n_paired / len(window) if window else 0

            # Asymmetry: opening vs closing
            n_open = sum(1 for c in window if c == '(')
            n_close = sum(1 for c in window if c == ')')
            feat[f'open_close_diff_w{ws}'] = (n_open - n_close) / len(window) if window else 0

        # Stem features around center
        if feat['center_is_paired']:
            # Find stem length
            stem_left = 0
            for i in range(center_idx, -1, -1):
                if struct[i] in '()':
                    stem_left += 1
                else:
                    break

            stem_right = 0
            for i in range(center_idx, len(struct)):
                if struct[i] in '()':
                    stem_right += 1
                else:
                    break

            feat['stem_length'] = stem_left + stem_right - 1
        else:
            feat['stem_length'] = 0

        # Distance to nearest paired position
        dist_to_pair = 0
        for d in range(1, min(center_idx, len(struct) - center_idx)):
            if struct[center_idx - d] in '()' or struct[center_idx + d] in '()':
                dist_to_pair = d
                break
        feat['dist_to_nearest_pair'] = dist_to_pair

        # Global structure stats
        total_paired = sum(1 for c in struct if c in '()')
        feat['global_paired_frac'] = total_paired / len(struct) if struct else 0

        features.append(feat)

    return pd.DataFrame(features)


def extract_sequence_features(
    sequences: pd.Series,
    window_sizes: List[int] = [5, 10, 20, 50],
    center_idx: int = 100,
    kmer_sizes: List[int] = [3, 4]
) -> pd.DataFrame:
    """
    Extract features from RNA sequences.

    Features:
    - Nucleotide composition (global and local)
    - GC content
    - Dinucleotide frequencies
    - K-mer frequencies around center
    - Sequence context at center

    Args:
        sequences: Series of RNA sequences
        window_sizes: Windows for local features
        center_idx: Center position index
        kmer_sizes: K-mer sizes for frequency features

    Returns:
        DataFrame with sequence features
    """
    features = []
    nucleotides = ['A', 'C', 'G', 'U', 'T']

    for seq in sequences:
        seq = seq.upper().replace('T', 'U')
        feat = {}

        # Global composition
        for nuc in ['A', 'C', 'G', 'U']:
            feat[f'global_{nuc}_frac'] = seq.count(nuc) / len(seq) if seq else 0

        feat['global_gc'] = (seq.count('G') + seq.count('C')) / len(seq) if seq else 0
        feat['global_purine'] = (seq.count('A') + seq.count('G')) / len(seq) if seq else 0

        # Local composition in windows
        for ws in window_sizes:
            start = max(0, center_idx - ws)
            end = min(len(seq), center_idx + ws + 1)
            window = seq[start:end]

            for nuc in ['A', 'C', 'G', 'U']:
                feat[f'local_{nuc}_w{ws}'] = window.count(nuc) / len(window) if window else 0

            feat[f'local_gc_w{ws}'] = (window.count('G') + window.count('C')) / len(window) if window else 0

        # Context around center (±3nt)
        context_start = max(0, center_idx - 3)
        context_end = min(len(seq), center_idx + 4)
        context = seq[context_start:context_end]

        # Encode immediate neighbors
        if center_idx > 0:
            feat['left_1_A'] = 1 if seq[center_idx-1] == 'A' else 0
            feat['left_1_C'] = 1 if seq[center_idx-1] == 'C' else 0
            feat['left_1_G'] = 1 if seq[center_idx-1] == 'G' else 0
            feat['left_1_U'] = 1 if seq[center_idx-1] == 'U' else 0
        else:
            feat['left_1_A'] = feat['left_1_C'] = feat['left_1_G'] = feat['left_1_U'] = 0

        if center_idx < len(seq) - 1:
            feat['right_1_A'] = 1 if seq[center_idx+1] == 'A' else 0
            feat['right_1_C'] = 1 if seq[center_idx+1] == 'C' else 0
            feat['right_1_G'] = 1 if seq[center_idx+1] == 'G' else 0
            feat['right_1_U'] = 1 if seq[center_idx+1] == 'U' else 0
        else:
            feat['right_1_A'] = feat['right_1_C'] = feat['right_1_G'] = feat['right_1_U'] = 0

        # Dinucleotide frequencies (global)
        dinucs = ['AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG', 'CU',
                  'GA', 'GC', 'GG', 'GU', 'UA', 'UC', 'UG', 'UU']
        total_dinucs = len(seq) - 1 if len(seq) > 1 else 1
        for dinuc in dinucs:
            count = sum(1 for i in range(len(seq)-1) if seq[i:i+2] == dinuc)
            feat[f'dinuc_{dinuc}'] = count / total_dinucs

        features.append(feat)

    return pd.DataFrame(features)


def create_adar_splits(
    df: pd.DataFrame,
    split_type: str = 'random',
    config: Optional[ADARDataConfig] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test splits for ADAR data.

    Split types:
    - 'random': Random split (baseline)
    - 'chromosome': Hold out chromosomes for testing
    - 'gc_stratified': Stratify by GC content
    - 'structure_stratified': Stratify by structure complexity
    - 'coverage_stratified': Stratify by read coverage
    - 'position': Split by genomic position

    Args:
        df: ADAR DataFrame
        split_type: Type of split
        config: Data configuration (for ratios)
        **kwargs: Additional arguments for specific split types

    Returns:
        train_df, val_df, test_df
    """
    if config is None:
        config = ADARDataConfig()

    np.random.seed(config.random_state)

    if split_type == 'random':
        return _random_split(df, config)
    elif split_type == 'chromosome':
        return _chromosome_split(df, config, **kwargs)
    elif split_type == 'gc_stratified':
        return _gc_stratified_split(df, config)
    elif split_type == 'structure_stratified':
        return _structure_stratified_split(df, config)
    elif split_type == 'coverage_stratified':
        return _coverage_stratified_split(df, config)
    elif split_type == 'editing_rate_stratified':
        return _editing_rate_stratified_split(df, config)
    else:
        raise ValueError(f"Unknown split type: {split_type}")


def _random_split(
    df: pd.DataFrame,
    config: ADARDataConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random split with stratification by label."""
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=config.test_ratio,
        stratify=df['is_edited'],
        random_state=config.random_state
    )

    # Second split: train vs val
    val_ratio_adjusted = config.val_ratio / (config.train_ratio + config.val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=train_val['is_edited'],
        random_state=config.random_state
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _chromosome_split(
    df: pd.DataFrame,
    config: ADARDataConfig,
    test_chroms: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out specific chromosomes for testing."""
    chroms = df['chrom'].unique()

    if test_chroms is None:
        # Hold out ~15% of chromosomes
        np.random.shuffle(chroms)
        n_test = max(1, int(len(chroms) * config.test_ratio))
        test_chroms = chroms[:n_test]

    test_mask = df['chrom'].isin(test_chroms)
    test = df[test_mask]
    train_val = df[~test_mask]

    # Split train/val randomly
    val_ratio_adjusted = config.val_ratio / (config.train_ratio + config.val_ratio)
    train_idx = np.random.choice(
        len(train_val),
        size=int(len(train_val) * (1 - val_ratio_adjusted)),
        replace=False
    )
    val_idx = np.setdiff1d(np.arange(len(train_val)), train_idx)

    train = train_val.iloc[train_idx]
    val = train_val.iloc[val_idx]

    logger.info(f"Chromosome split - test chroms: {test_chroms}")

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _gc_stratified_split(
    df: pd.DataFrame,
    config: ADARDataConfig,
    n_bins: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by GC content."""
    # Compute GC content
    gc_content = df['sequence'].apply(
        lambda s: (s.upper().count('G') + s.upper().count('C')) / len(s) if s else 0
    )

    # Create bins
    gc_bins = pd.qcut(gc_content, q=n_bins, labels=False, duplicates='drop')

    # Combine with label for stratification
    strat_col = gc_bins.astype(str) + '_' + df['is_edited'].astype(str)

    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df,
        test_size=config.test_ratio,
        stratify=strat_col,
        random_state=config.random_state
    )

    strat_train_val = strat_col.loc[train_val.index]
    val_ratio_adjusted = config.val_ratio / (config.train_ratio + config.val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=strat_train_val,
        random_state=config.random_state
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _structure_stratified_split(
    df: pd.DataFrame,
    config: ADARDataConfig,
    n_bins: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by structure complexity (MFE)."""
    mfe_bins = pd.qcut(df['mfe'], q=n_bins, labels=False, duplicates='drop')
    strat_col = mfe_bins.astype(str) + '_' + df['is_edited'].astype(str)

    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df,
        test_size=config.test_ratio,
        stratify=strat_col,
        random_state=config.random_state
    )

    strat_train_val = strat_col.loc[train_val.index]
    val_ratio_adjusted = config.val_ratio / (config.train_ratio + config.val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=strat_train_val,
        random_state=config.random_state
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _coverage_stratified_split(
    df: pd.DataFrame,
    config: ADARDataConfig,
    n_bins: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split by read coverage."""
    cov_bins = pd.qcut(df['coverage'].clip(upper=df['coverage'].quantile(0.99)),
                       q=n_bins, labels=False, duplicates='drop')
    strat_col = cov_bins.astype(str) + '_' + df['is_edited'].astype(str)

    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(
        df,
        test_size=config.test_ratio,
        stratify=strat_col,
        random_state=config.random_state
    )

    strat_train_val = strat_col.loc[train_val.index]
    val_ratio_adjusted = config.val_ratio / (config.train_ratio + config.val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=strat_train_val,
        random_state=config.random_state
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _editing_rate_stratified_split(
    df: pd.DataFrame,
    config: ADARDataConfig,
    n_bins: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by editing rate.
    Tests generalization from low to high editing rates.
    """
    # Only stratify positives by editing rate, negatives have rate 0
    positives = df[df['is_edited'] == 1].copy()
    negatives = df[df['is_edited'] == 0].copy()

    # Bin positives by editing rate
    pos_bins = pd.qcut(positives['editing_rate'], q=n_bins, labels=False, duplicates='drop')

    from sklearn.model_selection import train_test_split

    # Split positives stratified by editing rate
    pos_train_val, pos_test = train_test_split(
        positives,
        test_size=config.test_ratio,
        stratify=pos_bins,
        random_state=config.random_state
    )

    pos_bins_train_val = pos_bins.loc[pos_train_val.index]
    val_ratio_adjusted = config.val_ratio / (config.train_ratio + config.val_ratio)
    pos_train, pos_val = train_test_split(
        pos_train_val,
        test_size=val_ratio_adjusted,
        stratify=pos_bins_train_val,
        random_state=config.random_state
    )

    # Split negatives randomly
    neg_train_val, neg_test = train_test_split(
        negatives,
        test_size=config.test_ratio,
        random_state=config.random_state
    )
    neg_train, neg_val = train_test_split(
        neg_train_val,
        test_size=val_ratio_adjusted,
        random_state=config.random_state
    )

    # Combine
    train = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=config.random_state)
    val = pd.concat([pos_val, neg_val]).sample(frac=1, random_state=config.random_state)
    test = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=config.random_state)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def prepare_features(
    df: pd.DataFrame,
    include_sequence: bool = True,
    include_structure: bool = True,
    include_metadata: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix from ADAR data.

    Args:
        df: ADAR DataFrame
        include_sequence: Include sequence-derived features
        include_structure: Include structure-derived features
        include_metadata: Include coverage, editing_rate, MFE

    Returns:
        X: Feature matrix [N, D]
        y: Labels [N]
        feature_names: List of feature names
    """
    feature_dfs = []

    if include_sequence:
        seq_features = extract_sequence_features(df['sequence'])
        feature_dfs.append(seq_features)

    if include_structure:
        struct_features = extract_structure_features(df['structure'])
        feature_dfs.append(struct_features)

    if include_metadata:
        meta_features = df[['coverage', 'mfe']].copy()
        meta_features['log_coverage'] = np.log1p(meta_features['coverage'])
        # Only include editing_rate for exploration, not for training (leakage)
        feature_dfs.append(meta_features[['log_coverage', 'mfe']])

    X = pd.concat(feature_dfs, axis=1)
    feature_names = X.columns.tolist()

    # Handle any NaN values
    X = X.fillna(0)

    y = df['is_edited'].values

    return X.values, y, feature_names


# ============================================================================
# Pre-computed Data Loading
# ============================================================================

class PrecomputedDataLoader:
    """
    Loader for pre-computed embeddings and graphs.

    Pre-computed data is stored in:
    - graphs.pkl: List of PyG Data objects
    - rnafm_embeddings.npy: [N, 640] RNA-FM embeddings
    - features.npy: [N, D] Hand-crafted features
    - labels.npy: [N] Binary labels
    - metadata.csv: Position metadata
    """

    def __init__(
        self,
        precomputed_dir: str = "data/rna/adar/precomputed_full",
        original_csv: str = "data/rna/adar/adar_editing_dataset_with_structure.csv"
    ):
        self.precomputed_dir = Path(precomputed_dir)
        self.original_csv = Path(original_csv)

        self._graphs = None
        self._rnafm_embeddings = None
        self._features = None
        self._labels = None
        self._metadata = None
        self._original_df = None

    @property
    def graphs(self):
        """Lazy-load graphs from sharded files."""
        if self._graphs is None:
            graphs_dir = self.precomputed_dir / 'graphs'
            if graphs_dir.exists():
                import pickle
                import json

                # Load metadata
                meta_file = graphs_dir / 'metadata.json'
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    logger.info(f"Loading {meta['total_graphs']} graphs from {meta['num_shards']} shards")

                # Load all shards
                self._graphs = []
                shard_files = sorted(graphs_dir.glob('shard_*.pkl'))
                for shard_file in shard_files:
                    with open(shard_file, 'rb') as f:
                        shard_graphs = pickle.load(f)
                        self._graphs.extend(shard_graphs)
                logger.info(f"Loaded {len(self._graphs)} graphs total")
            else:
                # Fallback to single file (old format)
                graph_file = self.precomputed_dir / 'graphs.pkl'
                if graph_file.exists():
                    import pickle
                    logger.info(f"Loading graphs from {graph_file}")
                    with open(graph_file, 'rb') as f:
                        self._graphs = pickle.load(f)
                    logger.info(f"Loaded {len(self._graphs)} graphs")
                else:
                    logger.warning(f"Graphs not found in {self.precomputed_dir}")
        return self._graphs

    def get_graph(self, idx: int):
        """
        Load a single graph by index without loading all graphs.
        More memory efficient for large datasets.
        """
        graphs_dir = self.precomputed_dir / 'graphs'
        if not graphs_dir.exists():
            # Fallback to loading all
            return self.graphs[idx] if self.graphs else None

        import pickle
        import json

        meta_file = graphs_dir / 'metadata.json'
        if not meta_file.exists():
            return self.graphs[idx] if self.graphs else None

        with open(meta_file) as f:
            meta = json.load(f)

        shard_size = meta['shard_size']
        shard_idx = idx // shard_size
        local_idx = idx % shard_size

        shard_file = graphs_dir / f'shard_{shard_idx:04d}.pkl'
        if not shard_file.exists():
            return None

        with open(shard_file, 'rb') as f:
            shard_graphs = pickle.load(f)

        return shard_graphs[local_idx]

    def get_graphs_for_indices(self, indices: np.ndarray) -> List:
        """
        Load graphs for specific indices efficiently.
        Groups by shard to minimize file reads.
        """
        graphs_dir = self.precomputed_dir / 'graphs'
        if not graphs_dir.exists():
            # Fallback
            all_graphs = self.graphs
            return [all_graphs[i] for i in indices] if all_graphs else []

        import pickle
        import json

        meta_file = graphs_dir / 'metadata.json'
        if not meta_file.exists():
            all_graphs = self.graphs
            return [all_graphs[i] for i in indices] if all_graphs else []

        with open(meta_file) as f:
            meta = json.load(f)

        shard_size = meta['shard_size']

        # Group indices by shard
        shard_to_indices = {}
        for i, idx in enumerate(indices):
            shard_idx = idx // shard_size
            if shard_idx not in shard_to_indices:
                shard_to_indices[shard_idx] = []
            shard_to_indices[shard_idx].append((i, idx % shard_size))

        # Load shards and extract graphs
        result = [None] * len(indices)
        for shard_idx, items in shard_to_indices.items():
            shard_file = graphs_dir / f'shard_{shard_idx:04d}.pkl'
            with open(shard_file, 'rb') as f:
                shard_graphs = pickle.load(f)
            for result_idx, local_idx in items:
                result[result_idx] = shard_graphs[local_idx]

        return result

    @property
    def rnafm_embeddings(self):
        """Lazy-load RNA-FM embeddings."""
        if self._rnafm_embeddings is None:
            emb_file = self.precomputed_dir / 'rnafm_embeddings.npy'
            if emb_file.exists():
                logger.info(f"Loading RNA-FM embeddings from {emb_file}")
                self._rnafm_embeddings = np.load(emb_file)
                logger.info(f"Loaded embeddings: {self._rnafm_embeddings.shape}")
            else:
                logger.warning(f"Embeddings file not found: {emb_file}")
        return self._rnafm_embeddings

    @property
    def features(self):
        """Lazy-load hand-crafted features."""
        if self._features is None:
            feat_file = self.precomputed_dir / 'features.npy'
            if feat_file.exists():
                logger.info(f"Loading features from {feat_file}")
                self._features = np.load(feat_file)
                logger.info(f"Loaded features: {self._features.shape}")
            else:
                logger.warning(f"Features file not found: {feat_file}")
        return self._features

    @property
    def labels(self):
        """Lazy-load labels."""
        if self._labels is None:
            label_file = self.precomputed_dir / 'labels.npy'
            if label_file.exists():
                self._labels = np.load(label_file)
            else:
                logger.warning(f"Labels file not found: {label_file}")
        return self._labels

    @property
    def metadata(self):
        """Lazy-load metadata."""
        if self._metadata is None:
            meta_file = self.precomputed_dir / 'metadata.csv'
            if meta_file.exists():
                self._metadata = pd.read_csv(meta_file)
            else:
                logger.warning(f"Metadata file not found: {meta_file}")
        return self._metadata

    @property
    def original_df(self):
        """Lazy-load original DataFrame (for splitting)."""
        if self._original_df is None:
            if self.original_csv.exists():
                logger.info(f"Loading original data from {self.original_csv}")
                self._original_df = pd.read_csv(self.original_csv)
            else:
                logger.warning(f"Original CSV not found: {self.original_csv}")
        return self._original_df

    def get_sampled_indices(
        self,
        n_positives: Optional[int] = None,
        n_negatives: int = 500_000,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Get indices for sampled subset of data.

        Args:
            n_positives: Number of positive samples (None = all)
            n_negatives: Number of negative samples
            random_state: Random seed

        Returns:
            Array of indices into the full precomputed data
        """
        labels = self.labels
        if labels is None:
            raise ValueError("Labels not available")

        np.random.seed(random_state)

        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]

        if n_positives is not None and n_positives < len(pos_idx):
            pos_idx = np.random.choice(pos_idx, size=n_positives, replace=False)

        if n_negatives < len(neg_idx):
            neg_idx = np.random.choice(neg_idx, size=n_negatives, replace=False)

        all_idx = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(all_idx)

        return all_idx

    def get_split_indices(
        self,
        indices: np.ndarray,
        split_type: str = 'random',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/val/test split indices.

        Args:
            indices: Indices to split (from get_sampled_indices)
            split_type: Type of split
            train_ratio, val_ratio, test_ratio: Split ratios
            random_state: Random seed

        Returns:
            train_indices, val_indices, test_indices
        """
        from sklearn.model_selection import train_test_split

        labels = self.labels[indices]

        if split_type == 'random':
            # Simple stratified split
            train_val_idx, test_idx = train_test_split(
                np.arange(len(indices)),
                test_size=test_ratio,
                stratify=labels,
                random_state=random_state
            )

            val_ratio_adj = val_ratio / (train_ratio + val_ratio)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio_adj,
                stratify=labels[train_val_idx],
                random_state=random_state
            )

            return indices[train_idx], indices[val_idx], indices[test_idx]

        elif split_type == 'gc_stratified':
            # Need to load original data for GC content
            df = self.original_df
            gc_content = df['sequence'].iloc[indices].apply(
                lambda s: (s.upper().count('G') + s.upper().count('C')) / len(s)
            )
            gc_bins = pd.qcut(gc_content, q=5, labels=False, duplicates='drop')
            strat = gc_bins.values * 2 + labels  # Combine GC and label

            train_val_idx, test_idx = train_test_split(
                np.arange(len(indices)),
                test_size=test_ratio,
                stratify=strat,
                random_state=random_state
            )

            val_ratio_adj = val_ratio / (train_ratio + val_ratio)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio_adj,
                stratify=strat[train_val_idx],
                random_state=random_state
            )

            return indices[train_idx], indices[val_idx], indices[test_idx]

        elif split_type == 'structure_stratified':
            # Stratify by MFE
            metadata = self.metadata
            if metadata is not None:
                mfe = metadata['mfe'].iloc[indices].values
                mfe_bins = pd.qcut(mfe, q=5, labels=False, duplicates='drop')
                strat = mfe_bins * 2 + labels
            else:
                strat = labels

            train_val_idx, test_idx = train_test_split(
                np.arange(len(indices)),
                test_size=test_ratio,
                stratify=strat,
                random_state=random_state
            )

            val_ratio_adj = val_ratio / (train_ratio + val_ratio)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio_adj,
                stratify=strat[train_val_idx],
                random_state=random_state
            )

            return indices[train_idx], indices[val_idx], indices[test_idx]

        elif split_type == 'chromosome':
            # Hold out chromosomes
            metadata = self.metadata
            if metadata is not None:
                chroms = metadata['chrom'].iloc[indices].values
                unique_chroms = np.unique(chroms)
                np.random.seed(random_state)
                np.random.shuffle(unique_chroms)
                n_test = max(1, int(len(unique_chroms) * test_ratio))
                test_chroms = unique_chroms[:n_test]

                test_mask = np.isin(chroms, test_chroms)
                test_idx = np.where(test_mask)[0]
                train_val_idx = np.where(~test_mask)[0]

                val_ratio_adj = val_ratio / (train_ratio + val_ratio)
                n_val = int(len(train_val_idx) * val_ratio_adj)
                val_idx = np.random.choice(train_val_idx, size=n_val, replace=False)
                train_idx = np.setdiff1d(train_val_idx, val_idx)

                return indices[train_idx], indices[val_idx], indices[test_idx]
            else:
                # Fallback to random
                return self.get_split_indices(indices, 'random', train_ratio, val_ratio, test_ratio, random_state)

        elif split_type == 'coverage_stratified':
            metadata = self.metadata
            if metadata is not None:
                coverage = metadata['coverage'].iloc[indices].values
                cov_clipped = np.clip(coverage, 0, np.percentile(coverage, 99))
                cov_bins = pd.qcut(cov_clipped, q=5, labels=False, duplicates='drop')
                strat = cov_bins * 2 + labels
            else:
                strat = labels

            train_val_idx, test_idx = train_test_split(
                np.arange(len(indices)),
                test_size=test_ratio,
                stratify=strat,
                random_state=random_state
            )

            val_ratio_adj = val_ratio / (train_ratio + val_ratio)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_ratio_adj,
                stratify=strat[train_val_idx],
                random_state=random_state
            )

            return indices[train_idx], indices[val_idx], indices[test_idx]

        elif split_type == 'editing_rate_stratified':
            metadata = self.metadata
            if metadata is not None:
                # Only stratify positives by editing rate
                pos_mask = labels == 1
                pos_indices_local = np.where(pos_mask)[0]
                neg_indices_local = np.where(~pos_mask)[0]

                if len(pos_indices_local) > 0:
                    edit_rates = metadata['editing_rate'].iloc[indices[pos_indices_local]].values
                    rate_bins = pd.qcut(edit_rates, q=5, labels=False, duplicates='drop')

                    # Split positives
                    pos_train_val, pos_test = train_test_split(
                        pos_indices_local,
                        test_size=test_ratio,
                        stratify=rate_bins,
                        random_state=random_state
                    )

                    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
                    pos_train, pos_val = train_test_split(
                        pos_train_val,
                        test_size=val_ratio_adj,
                        stratify=rate_bins[np.isin(pos_indices_local, pos_train_val)],
                        random_state=random_state
                    )
                else:
                    pos_train = pos_val = pos_test = np.array([], dtype=int)

                # Split negatives randomly
                if len(neg_indices_local) > 0:
                    neg_train_val, neg_test = train_test_split(
                        neg_indices_local,
                        test_size=test_ratio,
                        random_state=random_state
                    )
                    neg_train, neg_val = train_test_split(
                        neg_train_val,
                        test_size=val_ratio_adj,
                        random_state=random_state
                    )
                else:
                    neg_train = neg_val = neg_test = np.array([], dtype=int)

                train_idx = np.concatenate([pos_train, neg_train])
                val_idx = np.concatenate([pos_val, neg_val])
                test_idx = np.concatenate([pos_test, neg_test])

                return indices[train_idx], indices[val_idx], indices[test_idx]
            else:
                return self.get_split_indices(indices, 'random', train_ratio, val_ratio, test_ratio, random_state)

        else:
            raise ValueError(f"Unknown split type: {split_type}")

    def get_data_for_indices(
        self,
        indices: np.ndarray,
        include_graphs: bool = False,
        include_rnafm: bool = False,
        include_features: bool = True
    ) -> Dict:
        """
        Get data for specific indices.

        Args:
            indices: Array of indices
            include_graphs: Include PyG graphs
            include_rnafm: Include RNA-FM embeddings
            include_features: Include hand-crafted features

        Returns:
            Dict with requested data
        """
        result = {
            'labels': self.labels[indices]
        }

        if include_features and self.features is not None:
            result['features'] = self.features[indices]

        if include_rnafm and self.rnafm_embeddings is not None:
            result['rnafm_embeddings'] = self.rnafm_embeddings[indices]

        if include_graphs:
            # Use efficient shard-based loading
            result['graphs'] = self.get_graphs_for_indices(indices)

        if self.metadata is not None:
            result['metadata'] = self.metadata.iloc[indices]

        return result

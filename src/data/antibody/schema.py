"""
Data schema for antibody mutation datasets.

Defines the unified AbEditPairs format for antibody mutation data,
compatible with AbAgym, AbBiBench, and other mutation datasets.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import pandas as pd
import json


class AssayType(Enum):
    """Types of binding/activity assays."""
    BINDING_AFFINITY = "binding_affinity"  # Kd, Ka
    DDG = "ddg"  # ΔΔG of binding
    ENRICHMENT = "enrichment"  # DMS enrichment score
    EXPRESSION = "expression"  # Surface display, expression level
    STABILITY = "stability"  # Tm, aggregation
    IC50 = "ic50"  # Half-maximal inhibitory concentration
    EC50 = "ec50"  # Half-maximal effective concentration
    OTHER = "other"


class ChainType(Enum):
    """Antibody chain types."""
    HEAVY = "H"
    LIGHT = "L"
    KAPPA = "K"
    LAMBDA = "lambda"


@dataclass
class AbMutation:
    """
    Represents a single amino acid mutation in an antibody.

    Attributes:
        chain: Chain type ('H' for heavy, 'L' for light)
        position: 0-indexed position in the sequence
        from_aa: Original amino acid (single letter)
        to_aa: Mutated amino acid (single letter)
        imgt_position: Optional IMGT-numbered position
        region: Optional CDR/FR region (e.g., 'CDR3', 'FR1')
    """
    chain: str
    position: int
    from_aa: str
    to_aa: str
    imgt_position: Optional[int] = None
    region: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.chain}{self.from_aa}{self.position + 1}{self.to_aa}"

    @classmethod
    def from_string(cls, mutation_str: str) -> 'AbMutation':
        """
        Parse mutation from string format.

        Supported formats:
        - "HA50K" (chain H, A->K at position 50)
        - "H:A50K" (with colon separator)
        - "A50K" (assumes heavy chain)
        """
        mutation_str = mutation_str.strip()

        # Parse chain
        if mutation_str[0] in ['H', 'L', 'K'] and mutation_str[1] in 'ACDEFGHIKLMNPQRSTVWY':
            chain = mutation_str[0]
            rest = mutation_str[1:]
        elif ':' in mutation_str:
            chain, rest = mutation_str.split(':', 1)
        else:
            chain = 'H'  # Default to heavy
            rest = mutation_str

        # Parse from_aa, position, to_aa
        from_aa = rest[0]
        to_aa = rest[-1]
        position = int(rest[1:-1]) - 1  # Convert to 0-indexed

        return cls(chain=chain, position=position, from_aa=from_aa, to_aa=to_aa)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chain': self.chain,
            'position': self.position,
            'from_aa': self.from_aa,
            'to_aa': self.to_aa,
            'imgt_position': self.imgt_position,
            'region': self.region,
        }


@dataclass
class AbEditPair:
    """
    A single antibody mutation example with associated measurements.

    This is the unified format for antibody mutation data, supporting:
    - Single and multiple mutations
    - Multiple measurement types (binding, expression, stability)
    - Optional structural information

    Attributes:
        antibody_id: Unique identifier for the antibody
        antigen_id: Optional antigen identifier
        heavy_wt: Wild-type heavy chain sequence
        light_wt: Wild-type light chain sequence
        mutations: List of mutations
        assay_type: Type of assay used
        delta_value: Primary Δ measurement (e.g., ΔΔG, Δ-enrichment)
        raw_wt_value: Optional raw value for wild-type
        raw_mut_value: Optional raw value for mutant
        delta_binding: Optional Δ-binding measurement
        delta_expression: Optional Δ-expression measurement
        delta_stability: Optional Δ-stability measurement
        structure_id: Optional PDB or structure ID
        source_dataset: Original dataset name
        metadata: Additional metadata
    """
    antibody_id: str
    heavy_wt: str
    light_wt: str
    mutations: List[AbMutation]

    # Measurements
    delta_value: float
    assay_type: AssayType = AssayType.DDG

    # Optional
    antigen_id: Optional[str] = None
    raw_wt_value: Optional[float] = None
    raw_mut_value: Optional[float] = None
    delta_binding: Optional[float] = None
    delta_expression: Optional[float] = None
    delta_stability: Optional[float] = None

    # Structure
    structure_id: Optional[str] = None

    # Metadata
    source_dataset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and clean fields after initialization."""
        import pandas as pd
        # Ensure light_wt is a string (handle NaN from pandas)
        if self.light_wt is None or (isinstance(self.light_wt, float) and pd.isna(self.light_wt)):
            object.__setattr__(self, 'light_wt', '')

    @property
    def heavy_mut(self) -> str:
        """Get mutant heavy chain sequence."""
        seq = list(self.heavy_wt)
        for mut in self.mutations:
            if mut.chain.upper() == 'H' and mut.position < len(seq):
                seq[mut.position] = mut.to_aa
        return ''.join(seq)

    @property
    def light_mut(self) -> str:
        """Get mutant light chain sequence."""
        if not self.light_wt:
            return ''
        seq = list(self.light_wt)
        for mut in self.mutations:
            if mut.chain.upper() in ['L', 'K'] and mut.position < len(seq):
                seq[mut.position] = mut.to_aa
        return ''.join(seq)

    @property
    def num_mutations(self) -> int:
        """Number of mutations."""
        return len(self.mutations)

    @property
    def mutation_string(self) -> str:
        """Comma-separated mutation string."""
        return ','.join(str(m) for m in self.mutations)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'antibody_id': self.antibody_id,
            'antigen_id': self.antigen_id,
            'heavy_wt': self.heavy_wt,
            'light_wt': self.light_wt,
            'heavy_mut': self.heavy_mut,
            'light_mut': self.light_mut,
            'mutations': [m.to_dict() for m in self.mutations],
            'mutation_string': self.mutation_string,
            'num_mutations': self.num_mutations,
            'assay_type': self.assay_type.value,
            'delta_value': self.delta_value,
            'raw_wt_value': self.raw_wt_value,
            'raw_mut_value': self.raw_mut_value,
            'delta_binding': self.delta_binding,
            'delta_expression': self.delta_expression,
            'delta_stability': self.delta_stability,
            'structure_id': self.structure_id,
            'source_dataset': self.source_dataset,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AbEditPair':
        """Create from dictionary."""
        mutations = [
            AbMutation(**m) if isinstance(m, dict) else m
            for m in d.get('mutations', [])
        ]

        assay_type = d.get('assay_type', 'ddg')
        if isinstance(assay_type, str):
            assay_type = AssayType(assay_type)

        return cls(
            antibody_id=d['antibody_id'],
            antigen_id=d.get('antigen_id'),
            heavy_wt=d['heavy_wt'],
            light_wt=d['light_wt'],
            mutations=mutations,
            assay_type=assay_type,
            delta_value=d['delta_value'],
            raw_wt_value=d.get('raw_wt_value'),
            raw_mut_value=d.get('raw_mut_value'),
            delta_binding=d.get('delta_binding'),
            delta_expression=d.get('delta_expression'),
            delta_stability=d.get('delta_stability'),
            structure_id=d.get('structure_id'),
            source_dataset=d.get('source_dataset'),
            metadata=d.get('metadata', {}),
        )


class AbEditPairsDataset:
    """
    Collection of AbEditPair examples.

    Provides utilities for loading, filtering, and converting data.
    """

    def __init__(self, pairs: List[AbEditPair] = None):
        self.pairs = pairs or []

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx) -> AbEditPair:
        return self.pairs[idx]

    def __iter__(self):
        return iter(self.pairs)

    def add(self, pair: AbEditPair):
        """Add a pair to the dataset."""
        self.pairs.append(pair)

    def extend(self, pairs: List[AbEditPair]):
        """Add multiple pairs."""
        self.pairs.extend(pairs)

    def filter(
        self,
        antibody_ids: Optional[List[str]] = None,
        antigen_ids: Optional[List[str]] = None,
        max_mutations: Optional[int] = None,
        min_mutations: int = 1,
        assay_types: Optional[List[AssayType]] = None,
        source_datasets: Optional[List[str]] = None,
    ) -> 'AbEditPairsDataset':
        """
        Filter dataset by criteria.

        Returns a new filtered dataset.
        """
        filtered = []

        for pair in self.pairs:
            # Check antibody ID
            if antibody_ids is not None and pair.antibody_id not in antibody_ids:
                continue

            # Check antigen ID
            if antigen_ids is not None and pair.antigen_id not in antigen_ids:
                continue

            # Check mutation count
            if pair.num_mutations < min_mutations:
                continue
            if max_mutations is not None and pair.num_mutations > max_mutations:
                continue

            # Check assay type
            if assay_types is not None and pair.assay_type not in assay_types:
                continue

            # Check source dataset
            if source_datasets is not None and pair.source_dataset not in source_datasets:
                continue

            filtered.append(pair)

        return AbEditPairsDataset(filtered)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        records = [p.to_dict() for p in self.pairs]
        return pd.DataFrame(records)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'AbEditPairsDataset':
        """Create from pandas DataFrame."""
        pairs = []
        for _, row in df.iterrows():
            d = row.to_dict()

            # Parse mutations if string
            if 'mutations' in d and isinstance(d['mutations'], str):
                d['mutations'] = json.loads(d['mutations'])

            pairs.append(AbEditPair.from_dict(d))

        return cls(pairs)

    def save(self, path: str):
        """Save to file (CSV or JSON)."""
        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump([p.to_dict() for p in self.pairs], f, indent=2)
        else:
            self.to_dataframe().to_csv(path, index=False)

    @classmethod
    def load(cls, path: str) -> 'AbEditPairsDataset':
        """Load from file."""
        if path.endswith('.json'):
            with open(path, 'r') as f:
                data = json.load(f)
            pairs = [AbEditPair.from_dict(d) for d in data]
            return cls(pairs)
        else:
            df = pd.read_csv(path)
            return cls.from_dataframe(df)

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        by_antibody: bool = True,
    ) -> tuple:
        """
        Split dataset into train/val/test.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed
            by_antibody: If True, split by antibody ID to prevent leakage

        Returns:
            Tuple of (train, val, test) AbEditPairsDataset
        """
        import numpy as np
        np.random.seed(seed)

        if by_antibody:
            # Get unique antibody IDs
            antibody_ids = list(set(p.antibody_id for p in self.pairs))
            np.random.shuffle(antibody_ids)

            n_train = int(len(antibody_ids) * train_ratio)
            n_val = int(len(antibody_ids) * val_ratio)

            train_ids = set(antibody_ids[:n_train])
            val_ids = set(antibody_ids[n_train:n_train + n_val])
            test_ids = set(antibody_ids[n_train + n_val:])

            train_pairs = [p for p in self.pairs if p.antibody_id in train_ids]
            val_pairs = [p for p in self.pairs if p.antibody_id in val_ids]
            test_pairs = [p for p in self.pairs if p.antibody_id in test_ids]

        else:
            # Random split
            indices = np.random.permutation(len(self.pairs))

            n_train = int(len(indices) * train_ratio)
            n_val = int(len(indices) * val_ratio)

            train_pairs = [self.pairs[i] for i in indices[:n_train]]
            val_pairs = [self.pairs[i] for i in indices[n_train:n_train + n_val]]
            test_pairs = [self.pairs[i] for i in indices[n_train + n_val:]]

        return (
            AbEditPairsDataset(train_pairs),
            AbEditPairsDataset(val_pairs),
            AbEditPairsDataset(test_pairs),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        import numpy as np

        delta_values = [p.delta_value for p in self.pairs]
        num_mutations = [p.num_mutations for p in self.pairs]

        return {
            'n_pairs': len(self.pairs),
            'n_antibodies': len(set(p.antibody_id for p in self.pairs)),
            'n_antigens': len(set(p.antigen_id for p in self.pairs if p.antigen_id)),
            'delta_mean': np.mean(delta_values),
            'delta_std': np.std(delta_values),
            'delta_min': np.min(delta_values),
            'delta_max': np.max(delta_values),
            'mutations_mean': np.mean(num_mutations),
            'mutations_max': np.max(num_mutations),
            'single_mutations': sum(1 for n in num_mutations if n == 1),
            'multi_mutations': sum(1 for n in num_mutations if n > 1),
            'assay_types': dict(pd.Series([p.assay_type.value for p in self.pairs]).value_counts()),
        }

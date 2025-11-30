"""
Configuration for RNA MPRA experiments.

Parallel to experiments/small_molecules/experiment_config.py but
adapted for RNA sequences and MPRA data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class RNAExperimentConfig:
    """
    Configuration for RNA edit effect prediction experiments.

    This configuration is designed to work with:
    - MPRA (Massively Parallel Reporter Assay) data
    - Pretrained RNA embedders (RNA-FM, RNABERT, Nucleotide)
    - The same model architectures as small molecules

    Example:
        config = RNAExperimentConfig(
            experiment_name="mpra_5utr_mrl",
            data_file="data/rna/pairs/mpra_5utr_pairs_long.csv",
            embedder_type='nucleotide',
            methods=[
                {'name': 'Edit Framework', 'type': 'edit_framework', ...},
                {'name': 'Baseline', 'type': 'baseline_property', ...}
            ]
        )
    """

    # Required parameters
    experiment_name: str
    data_file: str

    # RNA embedder settings
    embedder_type: str = 'nucleotide'  # 'nucleotide', 'rnafm', 'rnabert'
    trainable_embedder: bool = False
    embedder_device: str = 'auto'

    # RNA-specific embedder options
    rnafm_model: str = 'rna_fm_t12'
    rnafm_pooling: str = 'mean'
    rnabert_model: str = 'multimolecule/rnabert'
    rnabert_pooling: str = 'mean'
    nucleotide_kmer_sizes: List[int] = field(default_factory=lambda: [3, 4])
    nucleotide_include_structure: bool = False

    # Edit embedder settings
    use_local_context: bool = False
    context_window: int = 25
    include_edit_features: bool = False

    # Data settings
    num_tasks: int = 1
    min_pairs_per_property: int = 0
    property_filter: Optional[List[str]] = None  # e.g., ['MRL_5UTR']

    # Splitting
    splitter_type: str = 'random'  # 'random', 'sequence_similarity', 'motif', 'edit_type'
    splitter_params: Dict = field(default_factory=dict)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42

    # Model methods to run
    methods: List[Dict] = field(default_factory=list)

    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: [
        'mae', 'rmse', 'r2', 'pearson_r', 'spearman_r', 'direction_accuracy'
    ])

    # Output
    output_dir: str = 'results'
    save_models: bool = False
    models_dir: str = 'models'

    # Analysis
    include_motif_analysis: bool = True
    include_edit_embedding_comparison: bool = True

    # Additional test sets
    additional_test_files: Dict[str, str] = field(default_factory=dict)


# Default configurations for common experiments

DEFAULT_NUCLEOTIDE_CONFIG = {
    'embedder_type': 'nucleotide',
    'nucleotide_kmer_sizes': [3, 4],
    'nucleotide_include_structure': False,
    'methods': [
        {
            'name': 'Edit Framework - Nucleotide',
            'type': 'edit_framework',
            'use_edit_fragments': False,
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 0.001,
            'batch_size': 64,
            'max_epochs': 50
        },
        {
            'name': 'Baseline Property Predictor',
            'type': 'baseline_property',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 0.001,
            'batch_size': 64,
            'max_epochs': 50
        }
    ]
}

DEFAULT_RNAFM_CONFIG = {
    'embedder_type': 'rnafm',
    'rnafm_model': 'rna_fm_t12',
    'rnafm_pooling': 'mean',
    'trainable_embedder': False,
    'methods': [
        {
            'name': 'Edit Framework - RNA-FM',
            'type': 'edit_framework',
            'use_edit_fragments': False,
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 30
        },
        {
            'name': 'Baseline Property Predictor',
            'type': 'baseline_property',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 30
        }
    ]
}

DEFAULT_RNABERT_CONFIG = {
    'embedder_type': 'rnabert',
    'rnabert_model': 'multimolecule/rnabert',
    'rnabert_pooling': 'mean',
    'trainable_embedder': False,
    'methods': [
        {
            'name': 'Edit Framework - RNABERT',
            'type': 'edit_framework',
            'use_edit_fragments': False,
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 30
        },
        {
            'name': 'Baseline Property Predictor',
            'type': 'baseline_property',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 30
        }
    ]
}

# Structured Edit Embedder configuration (uses RNA-FM internally)
DEFAULT_STRUCTURED_EDIT_CONFIG = {
    'embedder_type': 'nucleotide',  # Base embedder (not used for structured)
    'methods': [
        {
            'name': 'Structured Edit Framework',
            'type': 'structured_edit_framework',
            # RNA-FM settings (used internally)
            'rnafm_model': 'rna_fm_t12',
            'rnafm_pooling': 'mean',
            'device': 'auto',
            # StructuredRNAEditEmbedder dimensions
            'mutation_type_dim': 64,
            'mutation_effect_dim': 256,
            'position_dim': 64,
            'local_context_dim': 256,
            'attention_context_dim': 128,
            'fusion_hidden_dims': [512, 384],
            'output_dim': 256,
            'window_size': 10,
            'embedder_dropout': 0.1,
            # Prediction head
            'head_hidden_dims': [128, 64],
            'dropout': 0.1,
            # Training
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        {
            'name': 'Edit Framework - RNA-FM',
            'type': 'edit_framework',
            'use_edit_fragments': False,
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        {
            'name': 'Baseline Property Predictor',
            'type': 'baseline_property',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.1,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        }
    ]
}

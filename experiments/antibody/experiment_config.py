"""
Configuration for antibody mutation effect prediction experiments.

Parallel to experiments/rna/experiment_config.py and experiments/small_molecules/experiment_config.py
but adapted for antibody sequences with paired heavy-light chains and structured mutation data.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class AntibodyExperimentConfig:
    """
    Configuration for antibody mutation effect prediction experiments.

    This configuration is designed to work with:
    - AbBiBench, AbAgym, and similar antibody mutation datasets
    - Pretrained antibody embedders (IgT5, IgBert, AntiBERTa2, AbLang2, BALM)
    - Optional structural encoders (GVP-GNN, SE3-Transformer, Equiformer)
    - The same model architectures as small molecules and RNA

    Example:
        config = AntibodyExperimentConfig(
            experiment_name="abbibench_binding",
            data_dir="data/antibody/abbibench",
            embedder_type='igt5',
            methods=[
                {'name': 'Edit Framework', 'type': 'edit_framework', ...},
                {'name': 'Structured Edit', 'type': 'structured_edit', ...}
            ]
        )
    """

    # Required parameters
    experiment_name: str
    data_dir: str

    # Data source
    dataset_type: str = 'abbibench'  # 'abbibench', 'abagym', 'ab_bind', 'skempi2'
    dataset_kwargs: Dict = field(default_factory=dict)

    # Antibody embedder settings
    embedder_type: str = 'igt5'  # 'igt5', 'igbert', 'antiberta2', 'ablang2', 'balm', 'balm_paired'
    trainable_embedder: bool = False
    embedder_device: str = 'auto'

    # Embedder-specific options
    igt5_model: str = 'Exscientia/IgT5'
    igbert_model: str = 'Exscientia/IgBert'
    antiberta2_model: str = 'alchemab/antiberta2'
    ablang2_model: str = 'qilowoq/AbLang2'
    balm_model: str = 'beam-labs/BALM'
    balm_paired_model: str = 'briney/BALM-paired'
    pooling: str = 'mean'  # 'mean', 'cls', 'max'

    # Structural encoder settings (optional)
    use_structure: bool = False
    structural_encoder: str = 'gvp'  # 'gvp', 'se3', 'equiformer'
    structure_dir: Optional[str] = None  # Directory with PDB files

    # Edit embedder settings
    edit_embedder_type: str = 'simple'  # 'simple', 'structured'
    use_identity_features: bool = True  # BLOSUM, hydropathy, charge, volume
    use_location_features: bool = True  # Chain, IMGT position, CDR region
    use_local_context: bool = True
    context_window: int = 5
    aggregation: str = 'mean'  # 'mean', 'attention'

    # Multi-task settings
    task_names: Optional[List[str]] = None  # e.g., ['delta_binding', 'delta_expression']
    task_weights: Optional[Dict[str, float]] = None
    num_tasks: int = 1

    # Data settings
    include_multi_mutations: bool = True
    max_mutations: int = 10
    min_pairs: int = 0

    # Splitting
    splitter_type: str = 'antibody'  # 'random', 'antibody' (by antibody ID)
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
    output_dir: str = 'results/antibody'
    save_models: bool = False
    models_dir: str = 'models/antibody'

    # Additional test sets
    additional_test_files: Dict[str, str] = field(default_factory=dict)


# Default configurations for common experiments

DEFAULT_IGT5_CONFIG = {
    'embedder_type': 'igt5',
    'igt5_model': 'Exscientia/IgT5',
    'pooling': 'mean',
    'trainable_embedder': False,
    'methods': [
        {
            'name': 'Edit Framework - IgT5',
            'type': 'edit_framework',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        {
            'name': 'Baseline Property Predictor',
            'type': 'baseline_property',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        }
    ]
}

DEFAULT_IGBERT_CONFIG = {
    'embedder_type': 'igbert',
    'igbert_model': 'Exscientia/IgBert',
    'pooling': 'mean',
    'trainable_embedder': False,
    'methods': [
        {
            'name': 'Edit Framework - IgBert',
            'type': 'edit_framework',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        {
            'name': 'Baseline Property Predictor',
            'type': 'baseline_property',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        }
    ]
}

DEFAULT_STRUCTURED_EDIT_CONFIG = {
    'embedder_type': 'igt5',
    'edit_embedder_type': 'structured',
    'use_identity_features': True,
    'use_location_features': True,
    'use_local_context': True,
    'context_window': 5,
    'aggregation': 'attention',
    'methods': [
        {
            'name': 'Structured Edit Framework - IgT5',
            'type': 'structured_edit_framework',
            # StructuredAntibodyEditEmbedder dimensions
            'identity_dim': 64,
            'location_dim': 64,
            'local_context_dim': 128,
            'global_context_dim': 128,
            'structure_dim': 128,
            'fusion_hidden_dims': [512, 384],
            'output_dim': 320,
            'window_size': 5,
            'aggregation': 'attention',
            'embedder_dropout': 0.1,
            # Prediction head
            'head_hidden_dims': [256, 128, 64],
            'dropout': 0.2,
            # Training
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        {
            'name': 'Edit Framework - IgT5',
            'type': 'edit_framework',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        {
            'name': 'Baseline Property Predictor',
            'type': 'baseline_property',
            'hidden_dims': [512, 256, 128],
            'head_hidden_dims': [128, 64],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        }
    ]
}

DEFAULT_STRUCTURE_AWARE_CONFIG = {
    'embedder_type': 'igt5',
    'edit_embedder_type': 'structured',
    'use_structure': True,
    'structural_encoder': 'gvp',
    'use_identity_features': True,
    'use_location_features': True,
    'methods': [
        {
            'name': 'Structure-aware Structured Edit - IgT5 + GVP',
            'type': 'structured_edit_framework',
            'use_structure': True,
            'structural_encoder': 'gvp',
            # Dimensions
            'identity_dim': 64,
            'location_dim': 64,
            'local_context_dim': 128,
            'global_context_dim': 128,
            'structure_dim': 128,
            'fusion_hidden_dims': [512, 384],
            'output_dim': 320,
            # Training
            'head_hidden_dims': [256, 128, 64],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 16,  # Smaller due to structure
            'max_epochs': 50
        },
        {
            'name': 'Structured Edit Framework - IgT5',
            'type': 'structured_edit_framework',
            'use_structure': False,
            'identity_dim': 64,
            'location_dim': 64,
            'local_context_dim': 128,
            'global_context_dim': 128,
            'fusion_hidden_dims': [512, 384],
            'output_dim': 320,
            'head_hidden_dims': [256, 128, 64],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        }
    ]
}

# Multi-task configuration for predicting multiple properties
MULTITASK_CONFIG = {
    'embedder_type': 'igt5',
    'edit_embedder_type': 'structured',
    'task_names': ['delta_binding', 'delta_expression', 'delta_stability'],
    'task_weights': {'delta_binding': 1.0, 'delta_expression': 0.5, 'delta_stability': 0.5},
    'num_tasks': 3,
    'methods': [
        {
            'name': 'Multi-task Structured Edit',
            'type': 'structured_edit_framework',
            'n_tasks': 3,
            'identity_dim': 64,
            'location_dim': 64,
            'local_context_dim': 128,
            'global_context_dim': 128,
            'fusion_hidden_dims': [512, 384],
            'output_dim': 320,
            'head_hidden_dims': [256, 128, 64],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 100
        }
    ]
}

# Embedder comparison configuration
EMBEDDER_COMPARISON_CONFIG = {
    'methods': [
        # IgT5
        {
            'name': 'Edit Framework - IgT5',
            'type': 'edit_framework',
            'embedder_type': 'igt5',
            'hidden_dims': [512, 256, 128],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        # IgBert
        {
            'name': 'Edit Framework - IgBert',
            'type': 'edit_framework',
            'embedder_type': 'igbert',
            'hidden_dims': [512, 256, 128],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        # AntiBERTa2
        {
            'name': 'Edit Framework - AntiBERTa2',
            'type': 'edit_framework',
            'embedder_type': 'antiberta2',
            'hidden_dims': [512, 256, 128],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        # AbLang2
        {
            'name': 'Edit Framework - AbLang2',
            'type': 'edit_framework',
            'embedder_type': 'ablang2',
            'hidden_dims': [512, 256, 128],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        # BALM
        {
            'name': 'Edit Framework - BALM',
            'type': 'edit_framework',
            'embedder_type': 'balm',
            'hidden_dims': [512, 256, 128],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
        # BALM-paired
        {
            'name': 'Edit Framework - BALM-paired',
            'type': 'edit_framework',
            'embedder_type': 'balm_paired',
            'hidden_dims': [512, 256, 128],
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'max_epochs': 50
        },
    ]
}

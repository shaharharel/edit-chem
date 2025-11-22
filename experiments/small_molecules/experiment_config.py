from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ExperimentConfig:
    experiment_name: str

    data_file: str
    min_pairs_per_property: int
    num_tasks: int

    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int

    splitter_type: str
    splitter_params: Dict

    methods: List[Dict]
    metrics: List[str]
    test_datasets: List[str]

    save_models: bool
    models_dir: str
    output_dir: str

    embedder_type: str = 'chemprop'

    include_cluster_analysis: bool = True
    n_clusters: int = 4

    include_edit_embedding_comparison: bool = True

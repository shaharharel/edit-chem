import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gc
from experiment_config import ExperimentConfig
from run_experiment import run_experiment


def main():
    # Splitters to run ( need to optimize 'butina')
    splitters = ['random', 'scaffold', 'stratified', 'target']

    for splitter_type in splitters:
        print(f"\n{'='*80}")
        print(f"Running experiment with {splitter_type.upper()} split")
        print(f"{'='*80}\n")

        # Configure splitter-specific parameters
        splitter_params = {}
        if splitter_type == 'target':
            splitter_params[splitter_type] = {'target_col': 'target_chembl_id'}
        elif splitter_type == 'stratified':
            splitter_params[splitter_type] = {'property_col': 'delta'}

        config = ExperimentConfig(
            experiment_name=f"small_molecule_edit_prediction_{splitter_type}",

            data_file="../../data/small_molecules/pairs/chembl_pairs_long_sample.csv",

            splitter_type=splitter_type,
            splitter_params=splitter_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,

            num_tasks=10,

            methods=[
                {
                    'name': 'Edit Framework - ChemProp',
                    'type': 'edit_framework',
                    'use_edit_fragments': False,
                    'hidden_dims': [512, 256, 128],
                    'dropout': 0.1,
                    'lr': 0.001,
                    'batch_size': 128,
                    'max_epochs': 10
                },
                {
                    'name': 'Edit Framework - ChemProp with Fragments',
                    'type': 'edit_framework',
                    'use_edit_fragments': True,
                    'hidden_dims': [512, 256, 128],
                    'dropout': 0.1,
                    'lr': 0.001,
                    'batch_size': 128,
                    'max_epochs': 10
                },
                {
                    'name': 'Baseline Property Predictor',
                    'type': 'baseline_property',
                    'hidden_dims': [512, 256, 128],
                    'dropout': 0.1,
                    'lr': 0.001,
                    'batch_size': 128,
                    'max_epochs': 10
                }
            ],

            embedder_type='chemprop',

            metrics=['mae', 'rmse', 'r2', 'pearson_r', 'spearman_r'],

            output_dir=f'results/{splitter_type}',

            additional_test_files={},

            include_cluster_analysis=True,
            n_clusters=4,

            include_edit_embedding_comparison=True
        )

        results, report_path = run_experiment(config)

        print(f"\n{'='*80}")
        print(f"Experiment '{config.experiment_name}' completed successfully!")
        print(f"Report saved to: {report_path}")
        print(f"{'='*80}\n")

        # Free memory after each splitter to avoid accumulation
        del results, report_path, config
        gc.collect()
        print(f"Memory freed after {splitter_type} splitter\n")

    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print("\nResults directories:")
    for splitter_type in splitters:
        print(f"  - experiments/small_molecules/results/{splitter_type}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiment_config import ExperimentConfig
from data_loader import load_datasets
from model_factory import create_models, create_embedder
from embedding_prep import prepare_embeddings
from trainer import train_all_models
from evaluator import evaluate_all_models
from report_generator import generate_report


def run_experiment(config: ExperimentConfig):
    print(f"Running experiment: {config.experiment_name}")

    train_data, test_datasets = load_datasets(config)

    embedder = create_embedder(config.embedder_type)

    models = create_models(config, train_data, embedder)

    use_fragments = any(m.get('use_edit_fragments', False) for m in config.methods)

    embeddings = prepare_embeddings(
        train_data=train_data,
        embedder=embedder,
        use_fragments=use_fragments,
        cache_dir='.embeddings_cache'
    )

    trained_models = train_all_models(models, train_data, config, embeddings)

    results = evaluate_all_models(trained_models, train_data, test_datasets, config, embeddings)

    report_path = generate_report(results, config, trained_models, embeddings, train_data)

    print(f"\nExperiment complete! Report saved to: {report_path}")
    return results, report_path

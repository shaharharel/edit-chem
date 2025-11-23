import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict
from src.models import PropertyPredictor, EditEffectPredictor
from src.embedding.small_molecule import ChemBERTaEmbedder, ChemPropEmbedder, EditEmbedder


def create_embedder(embedder_type: str):
    if embedder_type == 'chemberta':
        return ChemBERTaEmbedder()
    elif embedder_type == 'chemprop':
        return ChemPropEmbedder()
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


def create_models(config, train_data: Dict, embedder) -> Dict:
    models = {}
    task_names = list(train_data.keys())

    for method_config in config.methods:
        method_name = method_config['name']
        method_type = method_config['type']

        if method_type == 'baseline_property':
            model = PropertyPredictor(
                embedder=embedder,
                task_names=task_names,
                hidden_dims=method_config.get('hidden_dims'),
                dropout=method_config.get('dropout', 0.2),
                learning_rate=method_config.get('lr', 0.001),
                batch_size=method_config.get('batch_size', 32),
                max_epochs=method_config.get('max_epochs', method_config.get('epochs', 50))
            )

            models[method_name] = {
                'type': 'baseline_property',
                'embedder': embedder,
                'model': model,
                'config': method_config
            }

        elif method_type == 'edit_framework':
            edit_embedder = EditEmbedder(embedder)

            model = EditEffectPredictor(
                mol_embedder=embedder,
                edit_embedder=edit_embedder,
                task_names=task_names,
                hidden_dims=method_config.get('hidden_dims'),
                dropout=method_config.get('dropout', 0.2),
                learning_rate=method_config.get('lr', 0.001),
                batch_size=method_config.get('batch_size', 32),
                max_epochs=method_config.get('max_epochs', method_config.get('epochs', 50)),
                trainable_edit_embeddings=method_config.get('trainable_edit_embeddings', True),
                trainable_edit_hidden_dims=method_config.get('trainable_edit_dims', [512, 256]),
                trainable_edit_use_fragments=method_config.get('use_edit_fragments', False)
            )

            models[method_name] = {
                'type': 'edit_framework',
                'mol_embedder': embedder,
                'edit_embedder': edit_embedder,
                'model': model,
                'config': method_config
            }

    return models

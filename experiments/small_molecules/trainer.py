import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict
import numpy as np
import pandas as pd
from src.models import PropertyPredictor, EditEffectPredictor


def train_all_models(models: Dict, train_data: Dict, config, embeddings: Dict = None) -> Dict:
    trained_models = {}
    task_names = list(train_data.keys())
    num_tasks = len(task_names)

    for method_name, method_info in models.items():
        print(f"\nTraining {method_name}...")

        method_type = method_info['type']
        model = method_info['model']

        split_name = config.splitter_type
        checkpoint_path = Path(config.models_dir) / f"{method_name}_{split_name}.pt"

        load_checkpoint = method_info['config'].get('load_checkpoint')
        if load_checkpoint and Path(load_checkpoint).exists():
            print(f"Loading from {load_checkpoint}")
            if method_type == 'baseline_property':
                model = PropertyPredictor.load_checkpoint(
                    load_checkpoint,
                    embedder=method_info['embedder']
                )
            elif method_type == 'edit_framework':
                model = EditEffectPredictor.load_checkpoint(
                    load_checkpoint,
                    mol_embedder=method_info['mol_embedder']
                )
            method_info['model'] = model
            print("✓ Model loaded successfully")
        elif method_type == 'baseline_property':
            train_df_list = []
            val_df_list = []
            for prop in task_names:
                train_df_list.append(train_data[prop]['train'])
                val_df_list.append(train_data[prop]['val'])

            train_combined = pd.concat(train_df_list, ignore_index=True)
            val_combined = pd.concat(val_df_list, ignore_index=True)

            y_train = np.full((len(train_combined), num_tasks), np.nan, dtype=np.float32)
            y_val = np.full((len(val_combined), num_tasks), np.nan, dtype=np.float32)

            train_idx = 0
            val_idx = 0
            for i, prop in enumerate(task_names):
                train_prop = train_data[prop]['train']
                val_prop = train_data[prop]['val']

                y_train[train_idx:train_idx+len(train_prop), i] = train_prop['delta'].values
                y_val[val_idx:val_idx+len(val_prop), i] = val_prop['delta'].values

                train_idx += len(train_prop)
                val_idx += len(val_prop)

            if embeddings:
                mol_emb_train_list = []
                mol_emb_val_list = []
                for prop in task_names:
                    mol_emb_train_list.append(embeddings[prop]['train']['mol_b'])
                    mol_emb_val_list.append(embeddings[prop]['val']['mol_b'])

                mol_emb_train = np.vstack(mol_emb_train_list)
                mol_emb_val = np.vstack(mol_emb_val_list)

                model.fit(
                    mol_emb_train=mol_emb_train,
                    y_train=y_train,
                    mol_emb_val=mol_emb_val,
                    y_val=y_val,
                    verbose=True
                )

        elif method_type == 'edit_framework':
            train_df_list = []
            val_df_list = []
            for prop in task_names:
                train_df_list.append(train_data[prop]['train'])
                val_df_list.append(train_data[prop]['val'])

            train_combined = pd.concat(train_df_list, ignore_index=True)
            val_combined = pd.concat(val_df_list, ignore_index=True)

            delta_train = np.full((len(train_combined), num_tasks), np.nan, dtype=np.float32)
            delta_val = np.full((len(val_combined), num_tasks), np.nan, dtype=np.float32)

            train_idx = 0
            val_idx = 0
            for i, prop in enumerate(task_names):
                train_prop = train_data[prop]['train']
                val_prop = train_data[prop]['val']

                delta_train[train_idx:train_idx+len(train_prop), i] = train_prop['delta'].values
                delta_val[val_idx:val_idx+len(val_prop), i] = val_prop['delta'].values

                train_idx += len(train_prop)
                val_idx += len(val_prop)

            use_fragments = method_info['config'].get('use_edit_fragments', False)

            if embeddings:
                mol_emb_a_train_list = []
                mol_emb_b_train_list = []
                mol_emb_a_val_list = []
                mol_emb_b_val_list = []

                for prop in task_names:
                    mol_emb_a_train_list.append(embeddings[prop]['train']['mol_a'])
                    mol_emb_b_train_list.append(embeddings[prop]['train']['mol_b'])
                    mol_emb_a_val_list.append(embeddings[prop]['val']['mol_a'])
                    mol_emb_b_val_list.append(embeddings[prop]['val']['mol_b'])

                mol_emb_a_train = np.vstack(mol_emb_a_train_list)
                mol_emb_b_train = np.vstack(mol_emb_b_train_list)
                mol_emb_a_val = np.vstack(mol_emb_a_val_list)
                mol_emb_b_val = np.vstack(mol_emb_b_val_list)

                if use_fragments:
                    edit_emb_a_train_list = []
                    edit_emb_b_train_list = []
                    edit_emb_a_val_list = []
                    edit_emb_b_val_list = []

                    for prop in task_names:
                        edit_emb_a_train_list.append(embeddings[prop]['train']['edit_frag_a'])
                        edit_emb_b_train_list.append(embeddings[prop]['train']['edit_frag_b'])
                        edit_emb_a_val_list.append(embeddings[prop]['val']['edit_frag_a'])
                        edit_emb_b_val_list.append(embeddings[prop]['val']['edit_frag_b'])

                    edit_emb_a_train = np.vstack(edit_emb_a_train_list)
                    edit_emb_b_train = np.vstack(edit_emb_b_train_list)
                    edit_emb_a_val = np.vstack(edit_emb_a_val_list)
                    edit_emb_b_val = np.vstack(edit_emb_b_val_list)

                    model.fit(
                        mol_emb_a=mol_emb_a_train,
                        mol_emb_b=mol_emb_b_train,
                        delta_y=delta_train,
                        mol_emb_a_val=mol_emb_a_val,
                        mol_emb_b_val=mol_emb_b_val,
                        delta_y_val=delta_val,
                        edit_emb_a=edit_emb_a_train,
                        edit_emb_b=edit_emb_b_train,
                        edit_emb_a_val=edit_emb_a_val,
                        edit_emb_b_val=edit_emb_b_val,
                        verbose=True
                    )
                else:
                    model.fit(
                        mol_emb_a=mol_emb_a_train,
                        mol_emb_b=mol_emb_b_train,
                        delta_y=delta_train,
                        mol_emb_a_val=mol_emb_a_val,
                        mol_emb_b_val=mol_emb_b_val,
                        delta_y_val=delta_val,
                        verbose=True
                    )

        if config.save_models:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(str(checkpoint_path))
            print(f"Saved to {checkpoint_path}")

        trained_models[method_name] = {
            'type': method_type,
            'model': model,
            'config': method_info['config']
        }
        if method_type == 'baseline_property':
            trained_models[method_name]['embedder'] = method_info['embedder']
        elif method_type == 'edit_framework':
            trained_models[method_name]['mol_embedder'] = method_info['mol_embedder']
            trained_models[method_name]['edit_embedder'] = method_info['edit_embedder']

    return trained_models

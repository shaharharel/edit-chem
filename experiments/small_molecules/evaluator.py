import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict
import numpy as np
from src.utils import RegressionMetrics


def evaluate_all_models(trained_models: Dict, train_data: Dict, test_datasets: Dict, config) -> Dict:
    """
    Evaluate all models. Each method uses its own embedder to compute embeddings on-the-fly.
    """
    results = {
        'train': {},
        'test': {},
        'additional_test': {}
    }

    task_names = list(train_data.keys())

    for method_name, method_info in trained_models.items():
        print(f"\nEvaluating {method_name}...")

        method_type = method_info['type']
        model = method_info['model']

        # Get the embedder for this method
        if method_type in ['baseline_property', 'trainable_baseline_property']:
            embedder = method_info.get('embedder')
        else:
            embedder = method_info.get('mol_embedder')

        results['test'][method_name] = {}

        for prop in task_names:
            splits = train_data[prop]
            test_df = splits['test']

            if method_type == 'baseline_property':
                # PropertyPredictor predicts absolute values, so we need:
                # 1. Predict value_a for mol_a
                # 2. Predict value_b for mol_b
                # 3. Compute predicted delta = value_b - value_a

                smiles_a = test_df['mol_a'].tolist()
                smiles_b = test_df['mol_b'].tolist()
                y_true = test_df['delta'].values

                # Compute embeddings on-the-fly
                mol_emb_a = np.array(embedder.encode(smiles_a), dtype=np.float32)
                mol_emb_b = np.array(embedder.encode(smiles_b), dtype=np.float32)

                preds_a = model.predict(smiles_a, mol_emb=mol_emb_a)
                preds_b = model.predict(smiles_b, mol_emb=mol_emb_b)

                # Compute predicted delta
                y_pred = preds_b[prop] - preds_a[prop]

            elif method_type == 'trainable_baseline_property':
                # TrainablePropertyPredictor: End-to-end prediction with raw SMILES
                smiles_a = test_df['mol_a'].tolist()
                smiles_b = test_df['mol_b'].tolist()
                y_true = test_df['delta'].values

                preds_a = model.predict(smiles_a)
                preds_b = model.predict(smiles_b)

                # Compute predicted delta
                y_pred = preds_b[prop] - preds_a[prop]

            elif method_type == 'edit_framework':
                # EditEffectPredictor with frozen embedder
                smiles_a_test = test_df['mol_a'].tolist()
                smiles_b_test = test_df['mol_b'].tolist()
                y_true = test_df['delta'].values

                # Compute embeddings on-the-fly
                mol_emb_a_test = np.array(embedder.encode(smiles_a_test), dtype=np.float32)
                mol_emb_b_test = np.array(embedder.encode(smiles_b_test), dtype=np.float32)

                preds_all = model.predict(
                    smiles_a_test, smiles_b_test,
                    mol_emb_a=mol_emb_a_test,
                    mol_emb_b=mol_emb_b_test
                )
                y_pred = preds_all[prop]

            elif method_type == 'trainable_edit_framework':
                # TrainableEditEffectPredictor: End-to-end prediction with raw SMILES
                smiles_a_test = test_df['mol_a'].tolist()
                smiles_b_test = test_df['mol_b'].tolist()
                y_true = test_df['delta'].values

                preds_all = model.predict(smiles_a_test, smiles_b_test)
                y_pred = preds_all[prop]

            elif method_type == 'edit_framework_structured':
                # StructuredEditEffectPredictor: uses MMP structural info
                from src.data.small_molecule.mmp_parser import parse_mmp_batch

                smiles_a_test = test_df['mol_a'].tolist()
                smiles_b_test = test_df['mol_b'].tolist()
                y_true = test_df['delta'].values

                test_mmp = parse_mmp_batch(test_df)

                preds_all = model.predict(
                    smiles_A=smiles_a_test,
                    smiles_B=smiles_b_test,
                    removed_atoms_A=test_mmp['removed_atom_indices_A'],
                    added_atoms_B=test_mmp['added_atom_indices_B'],
                    attach_atoms_A=test_mmp['attach_atom_indices_A'],
                    mapped_pairs=test_mmp['mapped_atom_pairs']
                )
                y_pred = preds_all[prop]

            else:
                print(f"  Warning: Unknown method type {method_type} for {method_name}")
                continue

            metrics = RegressionMetrics.compute_all(y_true, y_pred)

            results['test'][method_name][prop] = {
                'metrics': metrics,
                'y_true': y_true,
                'y_pred': y_pred
            }

    # Evaluate on additional test datasets
    for test_dataset_name, test_df in test_datasets.items():
        results['additional_test'][test_dataset_name] = {}

        for method_name, method_info in trained_models.items():
            method_type = method_info['type']
            model = method_info['model']

            # Get the embedder for this method
            if method_type in ['baseline_property', 'trainable_baseline_property']:
                embedder = method_info.get('embedder')
            else:
                embedder = method_info.get('mol_embedder')

            results['additional_test'][test_dataset_name][method_name] = {}

            for prop in task_names:
                if 'property_name' in test_df.columns and prop not in test_df['property_name'].unique():
                    continue

                prop_test_df = test_df[test_df['property_name'] == prop] if 'property_name' in test_df.columns else test_df

                if len(prop_test_df) == 0:
                    continue

                if method_type == 'baseline_property':
                    smiles_a = prop_test_df['mol_a'].tolist()
                    smiles_b = prop_test_df['mol_b'].tolist()
                    y_true = prop_test_df['delta'].values

                    mol_emb_a = np.array(embedder.encode(smiles_a), dtype=np.float32)
                    mol_emb_b = np.array(embedder.encode(smiles_b), dtype=np.float32)

                    preds_a = model.predict(smiles_a, mol_emb=mol_emb_a)
                    preds_b = model.predict(smiles_b, mol_emb=mol_emb_b)
                    y_pred = preds_b[prop] - preds_a[prop]

                elif method_type == 'trainable_baseline_property':
                    smiles_a = prop_test_df['mol_a'].tolist()
                    smiles_b = prop_test_df['mol_b'].tolist()
                    y_true = prop_test_df['delta'].values

                    preds_a = model.predict(smiles_a)
                    preds_b = model.predict(smiles_b)
                    y_pred = preds_b[prop] - preds_a[prop]

                elif method_type == 'edit_framework':
                    smiles_a_test = prop_test_df['mol_a'].tolist()
                    smiles_b_test = prop_test_df['mol_b'].tolist()
                    y_true = prop_test_df['delta'].values

                    mol_emb_a_test = np.array(embedder.encode(smiles_a_test), dtype=np.float32)
                    mol_emb_b_test = np.array(embedder.encode(smiles_b_test), dtype=np.float32)

                    preds_all = model.predict(
                        smiles_a_test, smiles_b_test,
                        mol_emb_a=mol_emb_a_test,
                        mol_emb_b=mol_emb_b_test
                    )
                    y_pred = preds_all[prop]

                elif method_type == 'trainable_edit_framework':
                    smiles_a_test = prop_test_df['mol_a'].tolist()
                    smiles_b_test = prop_test_df['mol_b'].tolist()
                    y_true = prop_test_df['delta'].values

                    preds_all = model.predict(smiles_a_test, smiles_b_test)
                    y_pred = preds_all[prop]

                elif method_type == 'edit_framework_structured':
                    from src.data.small_molecule.mmp_parser import parse_mmp_batch

                    smiles_a_test = prop_test_df['mol_a'].tolist()
                    smiles_b_test = prop_test_df['mol_b'].tolist()
                    y_true = prop_test_df['delta'].values

                    test_mmp = parse_mmp_batch(prop_test_df)

                    preds_all = model.predict(
                        smiles_A=smiles_a_test,
                        smiles_B=smiles_b_test,
                        removed_atoms_A=test_mmp['removed_atom_indices_A'],
                        added_atoms_B=test_mmp['added_atom_indices_B'],
                        attach_atoms_A=test_mmp['attach_atom_indices_A'],
                        mapped_pairs=test_mmp['mapped_atom_pairs']
                    )
                    y_pred = preds_all[prop]

                else:
                    continue

                metrics = RegressionMetrics.compute_all(y_true, y_pred)

                results['additional_test'][test_dataset_name][method_name][prop] = {
                    'metrics': metrics,
                    'y_true': y_true,
                    'y_pred': y_pred
                }

    return results

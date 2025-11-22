import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict
from src.utils import RegressionMetrics


def evaluate_all_models(trained_models: Dict, train_data: Dict, test_datasets: Dict, config, embeddings: Dict = None) -> Dict:
    results = {
        'train': {},
        'test': {},
        'additional_test': {}
    }

    task_names = list(train_data.keys())

    for method_name, method_info in trained_models.items():
        method_type = method_info['type']
        model = method_info['model']

        results['test'][method_name] = {}

        for prop in task_names:
            splits = train_data[prop]
            test_df = splits['test']

            if method_type == 'baseline_property':
                smiles_test = test_df['mol_b'].tolist()
                y_true = test_df['delta'].values

                if embeddings and prop in embeddings and 'test' in embeddings[prop]:
                    mol_emb_test = embeddings[prop]['test']['mol_b']
                    preds_all = model.predict(smiles_test, mol_emb=mol_emb_test)
                else:
                    preds_all = model.predict(smiles_test)
                y_pred = preds_all[prop]

            elif method_type == 'edit_framework':
                smiles_a_test = test_df['mol_a'].tolist()
                smiles_b_test = test_df['mol_b'].tolist()
                y_true = test_df['delta'].values

                use_fragments = method_info['config'].get('use_edit_fragments', False)

                if embeddings and prop in embeddings and 'test' in embeddings[prop]:
                    mol_emb_a_test = embeddings[prop]['test']['mol_a']
                    mol_emb_b_test = embeddings[prop]['test']['mol_b']

                    if use_fragments:
                        edit_frag_a_test = embeddings[prop]['test']['edit_frag_a']
                        edit_frag_b_test = embeddings[prop]['test']['edit_frag_b']
                        preds_all = model.predict(
                            smiles_a_test, smiles_b_test,
                            mol_emb_a=mol_emb_a_test,
                            mol_emb_b=mol_emb_b_test,
                            edit_frag_a_emb=edit_frag_a_test,
                            edit_frag_b_emb=edit_frag_b_test
                        )
                    else:
                        preds_all = model.predict(
                            smiles_a_test, smiles_b_test,
                            mol_emb_a=mol_emb_a_test,
                            mol_emb_b=mol_emb_b_test
                        )
                else:
                    preds_all = model.predict(smiles_a_test, smiles_b_test)
                y_pred = preds_all[prop]

            metrics = RegressionMetrics.compute_all(y_true, y_pred)

            results['test'][method_name][prop] = {
                'metrics': metrics,
                'y_true': y_true,
                'y_pred': y_pred
            }

    for test_dataset_name, test_df in test_datasets.items():
        results['additional_test'][test_dataset_name] = {}

        for method_name, method_info in trained_models.items():
            method_type = method_info['type']
            model = method_info['model']

            results['additional_test'][test_dataset_name][method_name] = {}

            for prop in task_names:
                if 'property_name' in test_df.columns and prop not in test_df['property_name'].unique():
                    continue

                prop_test_df = test_df[test_df['property_name'] == prop] if 'property_name' in test_df.columns else test_df

                if len(prop_test_df) == 0:
                    continue

                if method_type == 'baseline_property':
                    smiles_test = prop_test_df['mol_b'].tolist()
                    y_true = prop_test_df['delta'].values
                    preds_all = model.predict(smiles_test)
                    y_pred = preds_all[prop]

                elif method_type == 'edit_framework':
                    smiles_a_test = prop_test_df['mol_a'].tolist()
                    smiles_b_test = prop_test_df['mol_b'].tolist()
                    y_true = prop_test_df['delta'].values
                    preds_all = model.predict(smiles_a_test, smiles_b_test)
                    y_pred = preds_all[prop]

                metrics = RegressionMetrics.compute_all(y_true, y_pred)

                results['additional_test'][test_dataset_name][method_name][prop] = {
                    'metrics': metrics,
                    'y_true': y_true,
                    'y_pred': y_pred
                }

    return results

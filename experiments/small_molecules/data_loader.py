import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from typing import Dict, List, Tuple
from src.utils import get_splitter


def load_datasets(config) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    df = pd.read_csv(config.data_file)

    train_data = {}
    test_datasets = {}

    df = df[df['edit_smiles'].notna() & df['mol_a'].notna() & df['mol_b'].notna() & df['delta'].notna()]

    property_counts = df.groupby('property_name').size()
    valid_properties = property_counts[property_counts >= config.min_pairs_per_property].index.tolist()
    df = df[df['property_name'].isin(valid_properties)]

    properties = df['property_name'].unique()
    selected_properties = properties[:config.num_tasks] if len(properties) > config.num_tasks else properties

    df = df[df['property_name'].isin(selected_properties)]

    edit_property_counts = df.groupby('edit_name')['property_name'].nunique()
    multi_property_edits = edit_property_counts[edit_property_counts > 1].index
    df = df[df['edit_name'].isin(multi_property_edits)].copy()

    print(f"Filtered to {len(df):,} pairs with edits appearing in >1 property")

    splitter = get_splitter(
        config.splitter_type,
        train_size=config.train_ratio,
        val_size=config.val_ratio,
        test_size=config.test_ratio,
        random_state=config.random_seed,
        **config.splitter_params.get(config.splitter_type, {})
    )

    for prop in selected_properties:
        prop_data = df[df['property_name'] == prop].copy()

        train, val, test = splitter.split(prop_data, smiles_col='mol_a')

        train_data[prop] = {
            'train': train,
            'val': val,
            'test': test
        }

    for test_dataset_name in config.test_datasets:
        try:
            test_df = pd.read_csv(test_dataset_name)
            test_datasets[test_dataset_name] = test_df
        except:
            pass

    return train_data, test_datasets

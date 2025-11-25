import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from typing import Dict, List, Tuple
from src.utils import get_splitter


def load_datasets(config) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    df = pd.read_csv(config.data_file)
    print(f"\nInitial dataset: {len(df):,} pairs")

    train_data = {}
    test_datasets = {}

    df = df[df['edit_smiles'].notna() & df['mol_a'].notna() & df['mol_b'].notna() & df['delta'].notna()]
    print(f"After removing missing values: {len(df):,} pairs")

    property_counts = df.groupby('property_name').size()
    valid_properties = property_counts[property_counts >= config.min_pairs_per_property].index.tolist()
    df = df[df['property_name'].isin(valid_properties)]
    print(f"After filtering properties with >={config.min_pairs_per_property} pairs: {len(df):,} pairs across {len(valid_properties)} properties")

    properties = df['property_name'].unique()
    selected_properties = properties[:config.num_tasks] if len(properties) > config.num_tasks else properties

    df = df[df['property_name'].isin(selected_properties)]
    print(f"After selecting top {config.num_tasks} properties: {len(df):,} pairs")

    edit_property_counts = df.groupby('edit_smiles')['property_name'].nunique()
    multi_property_edits = edit_property_counts[edit_property_counts >= config.min_properties_per_edit].index
    df = df[df['edit_smiles'].isin(multi_property_edits)].copy()

    print(f"After filtering edits appearing in >={config.min_properties_per_edit} properties: {len(df):,} pairs")
    print(f"Final dataset: {len(df):,} pairs with {len(df['edit_smiles'].unique()):,} unique edits\n")

    # Get splitter params, separating split() method args from __init__() args
    splitter_params = config.splitter_params.get(config.splitter_type, {})

    # Parameters that go to split() method, not __init__()
    split_method_params = {}
    init_params = {}

    # Known split() method parameters (vary by splitter type)
    split_param_names = {'property_col', 'smiles_col', 'target_col', 'time_col', 'core_col'}

    for key, value in splitter_params.items():
        if key in split_param_names:
            split_method_params[key] = value
        else:
            init_params[key] = value

    splitter = get_splitter(
        config.splitter_type,
        train_size=config.train_ratio,
        val_size=config.val_ratio,
        test_size=config.test_ratio,
        random_state=config.random_seed,
        **init_params
    )

    for prop in selected_properties:
        prop_data = df[df['property_name'] == prop].copy()

        train, val, test = splitter.split(prop_data, smiles_col='mol_a', **split_method_params)

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

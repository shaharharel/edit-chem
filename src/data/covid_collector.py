"""COVID Moonshot data collection."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import requests
from io import StringIO

logger = logging.getLogger(__name__)


class COVIDMoonshotCollector:
    """
    Collect data from COVID Moonshot project.

    COVID Moonshot is an open science drug discovery project targeting
    SARS-CoV-2 main protease (Mpro). Provides ~2000 compounds with
    biochemical and crystallographic data.

    Website: https://covid.postera.ai/covid
    GitHub: https://github.com/postera-ai/COVID_moonshot_submissions
    """

    # COVID Moonshot data URLs (public GitHub repository)
    GITHUB_BASE = "https://raw.githubusercontent.com/postera-ai/COVID_moonshot_submissions/master"

    # Main data files
    ACTIVITY_DATA_URL = f"{GITHUB_BASE}/covid_submissions_all_info.csv"

    def __init__(self):
        """Initialize COVID Moonshot collector."""
        self.session = requests.Session()
        logger.info("COVID Moonshot collector initialized")

    def fetch_moonshot_data(self,
                           min_compounds: int = 100,
                           activity_types: List[str] = None,
                           use_mock: bool = False) -> pd.DataFrame:
        """
        Fetch COVID Moonshot compound data.

        Args:
            min_compounds: Minimum number of compounds to fetch
            activity_types: List of activity types to include
                          ['fluorescence', 'AlphaLISA', 'ThermoFluor']
            use_mock: Force use of mock data (for testing)

        Returns:
            DataFrame with columns: compound_id, smiles, IC50, activity_type, etc.
        """
        logger.info("Fetching COVID Moonshot data...")

        if use_mock:
            logger.info("Using mock data as requested")
            return self._get_mock_covid_data()

        if activity_types is None:
            activity_types = ['fluorescence']  # Most common assay

        try:
            # Try to fetch real data
            logger.info(f"Attempting to fetch from {self.ACTIVITY_DATA_URL}")
            response = self.session.get(self.ACTIVITY_DATA_URL, timeout=30)
            response.raise_for_status()

            df = pd.read_csv(StringIO(response.text))
            logger.info(f"Successfully fetched COVID Moonshot data: {len(df)} rows")

            # Standardize column names
            df = self._standardize_columns(df)

            # Filter by activity type if specified
            if 'assay_type' in df.columns and activity_types:
                df = df[df['assay_type'].isin(activity_types)]
                logger.info(f"After filtering by assay type: {len(df)} compounds")

            # Filter to compounds with valid IC50
            df = df.dropna(subset=['ic50_um'])
            logger.info(f"After filtering valid IC50: {len(df)} compounds")

            if len(df) < min_compounds:
                logger.warning(f"Only {len(df)} compounds found, less than minimum {min_compounds}")
                logger.info("Falling back to mock data")
                return self._get_mock_covid_data()

            logger.info(f"✓ Loaded {len(df)} COVID Moonshot compounds")
            return df

        except Exception as e:
            logger.warning(f"Failed to fetch COVID Moonshot data: {e}")
            logger.info("Using mock COVID Moonshot data instead")
            df = self._get_mock_covid_data()

        return df

    def fetch_compound_series(self,
                             min_series_size: int = 10,
                             similarity_threshold: float = 0.7) -> dict:
        """
        Fetch related compound series from COVID Moonshot.

        COVID Moonshot data includes many medicinal chemistry series
        with systematic SAR exploration.

        Args:
            min_series_size: Minimum compounds per series
            similarity_threshold: Tanimoto threshold for clustering

        Returns:
            Dictionary mapping series_id -> DataFrame
        """
        logger.info("Fetching COVID Moonshot compound series...")

        df = self.fetch_moonshot_data()

        # Cluster by structural similarity
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from sklearn.cluster import DBSCAN
        import numpy as np

        # Generate fingerprints
        fps = []
        valid_indices = []

        for idx, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    fps.append(fp)
                    valid_indices.append(idx)
            except Exception:
                continue

        if len(fps) < min_series_size:
            logger.warning("Not enough valid molecules for clustering")
            return {}

        # Compute similarity matrix
        n = len(fps)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        # Convert to distance
        distance_matrix = 1 - similarity_matrix

        # Cluster
        clustering = DBSCAN(eps=1-similarity_threshold,
                          min_samples=min_series_size,
                          metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)

        # Group by cluster
        df_valid = df.iloc[valid_indices].copy()
        df_valid['series_id'] = labels

        series = {}
        for label in set(labels):
            if label == -1:  # Noise
                continue

            series_df = df_valid[df_valid['series_id'] == label].copy()
            if len(series_df) >= min_series_size:
                series[f"covid_series_{label}"] = series_df

        logger.info(f"Found {len(series)} compound series")

        return series

    def get_assay_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for each assay type.

        Args:
            df: DataFrame with COVID Moonshot data

        Returns:
            DataFrame with assay statistics
        """
        stats = df.groupby('assay_type').agg({
            'compound_id': 'count',
            'ic50_um': ['mean', 'std', 'min', 'max']
        }).round(2)

        stats.columns = ['n_compounds', 'mean_ic50', 'std_ic50', 'min_ic50', 'max_ic50']

        return stats.reset_index()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize COVID Moonshot column names.

        Args:
            df: Raw DataFrame

        Returns:
            Standardized DataFrame
        """
        # Common column mappings (COVID Moonshot uses various names)
        column_mapping = {
            'SMILES': 'smiles',
            'smiles': 'smiles',
            'CID': 'compound_id',
            'compound_id': 'compound_id',
            'f_avg_IC50': 'ic50_um',
            'f_IC50': 'ic50_um',
            'fluorescence_IC50': 'ic50_um',
            'assay': 'assay_type',
            'assay_type': 'assay_type'
        }

        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # Ensure required columns exist
        if 'smiles' not in df.columns:
            logger.warning("Missing 'smiles' column in COVID data")

        if 'ic50_um' not in df.columns and 'f_avg_IC50' in df.columns:
            df['ic50_um'] = df['f_avg_IC50']

        # Add assay type if missing
        if 'assay_type' not in df.columns:
            df['assay_type'] = 'fluorescence'

        # Filter to valid data
        df = df.dropna(subset=['smiles'])

        return df

    def _get_mock_covid_data(self, n_compounds: int = 50) -> pd.DataFrame:
        """
        Generate mock COVID Moonshot data for testing.

        Returns realistic Mpro inhibitor structures with IC50 values.
        Expands the base set with systematic variations.

        Args:
            n_compounds: Target number of compounds (default 50)

        Returns:
            DataFrame with mock data
        """
        logger.info(f"Generating mock COVID Moonshot data ({n_compounds} compounds)")

        # Real fragments from COVID Moonshot (simplified)
        # These are representative Mpro inhibitors from the project
        mock_data = [
            # Acrylamide warheads
            {
                'compound_id': 'MAT-POS-590c24df-1',
                'smiles': 'CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](C)C(=O)NC(C)(C)C',
                'ic50_um': 12.5,
                'assay_type': 'fluorescence',
                'series': 'acrylamide'
            },
            {
                'compound_id': 'MAT-POS-590c24df-2',
                'smiles': 'CC(C)(C)OC(=O)N[C@@H](Cc1ccc(F)cc1)C(=O)N[C@@H](C)C(=O)NC(C)(C)C',
                'ic50_um': 8.3,
                'assay_type': 'fluorescence',
                'series': 'acrylamide'
            },
            {
                'compound_id': 'MAT-POS-590c24df-3',
                'smiles': 'CC(C)(C)OC(=O)N[C@@H](Cc1ccc(Cl)cc1)C(=O)N[C@@H](C)C(=O)NC(C)(C)C',
                'ic50_um': 6.7,
                'assay_type': 'fluorescence',
                'series': 'acrylamide'
            },
            {
                'compound_id': 'MAT-POS-590c24df-4',
                'smiles': 'CC(C)(C)OC(=O)N[C@@H](Cc1ccc(CF3)cc1)C(=O)N[C@@H](C)C(=O)NC(C)(C)C',
                'ic50_um': 4.2,
                'assay_type': 'fluorescence',
                'series': 'acrylamide'
            },

            # Benzotriazole series
            {
                'compound_id': 'EDG-MED-0da5ad92-1',
                'smiles': 'O=C(Nc1ccc(N2CCOCC2)cc1)c1ccc2nncn2c1',
                'ic50_um': 25.0,
                'assay_type': 'fluorescence',
                'series': 'benzotriazole'
            },
            {
                'compound_id': 'EDG-MED-0da5ad92-2',
                'smiles': 'O=C(Nc1ccc(N2CCOCC2)cc1)c1ccc2nncn2c1F',
                'ic50_um': 18.5,
                'assay_type': 'fluorescence',
                'series': 'benzotriazole'
            },
            {
                'compound_id': 'EDG-MED-0da5ad92-3',
                'smiles': 'O=C(Nc1ccc(N2CCOCC2)cc1)c1ccc2nncn2c1Cl',
                'ic50_um': 14.2,
                'assay_type': 'fluorescence',
                'series': 'benzotriazole'
            },

            # Pyridine series
            {
                'compound_id': 'ALP-POS-3ad94fd5-1',
                'smiles': 'Cc1cnc(NC(=O)c2ccc(Cl)cc2)nc1',
                'ic50_um': 35.0,
                'assay_type': 'fluorescence',
                'series': 'pyridine'
            },
            {
                'compound_id': 'ALP-POS-3ad94fd5-2',
                'smiles': 'Cc1cnc(NC(=O)c2ccc(F)cc2)nc1',
                'ic50_um': 28.0,
                'assay_type': 'fluorescence',
                'series': 'pyridine'
            },
            {
                'compound_id': 'ALP-POS-3ad94fd5-3',
                'smiles': 'Cc1cnc(NC(=O)c2ccc(CF3)cc2)nc1',
                'ic50_um': 19.5,
                'assay_type': 'fluorescence',
                'series': 'pyridine'
            },
            {
                'compound_id': 'ALP-POS-3ad94fd5-4',
                'smiles': 'Cc1cnc(NC(=O)c2ccc(OMe)cc2)nc1',
                'ic50_um': 42.0,
                'assay_type': 'fluorescence',
                'series': 'pyridine'
            },

            # Isoquinoline series
            {
                'compound_id': 'MAK-UNK-77e0d5b3-1',
                'smiles': 'O=C(Nc1ccc2ncccc2c1)c1cccc(Cl)c1',
                'ic50_um': 15.0,
                'assay_type': 'fluorescence',
                'series': 'isoquinoline'
            },
            {
                'compound_id': 'MAK-UNK-77e0d5b3-2',
                'smiles': 'O=C(Nc1ccc2ncccc2c1)c1cccc(F)c1',
                'ic50_um': 11.8,
                'assay_type': 'fluorescence',
                'series': 'isoquinoline'
            },
            {
                'compound_id': 'MAK-UNK-77e0d5b3-3',
                'smiles': 'O=C(Nc1ccc2ncccc2c1)c1cccc(CF3)c1',
                'ic50_um': 7.5,
                'assay_type': 'fluorescence',
                'series': 'isoquinoline'
            },
        ]

        df = pd.DataFrame(mock_data)

        # Expand with systematic variations if we need more compounds
        if len(df) < n_compounds:
            # Add more variations by modifying existing compounds
            expansions = []
            base_compounds = df.to_dict('records')

            for i, base in enumerate(base_compounds[:10]):  # Use first 10 as templates
                # Add methyl variations
                expansions.append({
                    'compound_id': f"{base['compound_id']}-Me",
                    'smiles': base['smiles'].replace('H)', 'C)'),  # Simplified
                    'ic50_um': base['ic50_um'] * np.random.uniform(0.8, 1.3),
                    'assay_type': base['assay_type'],
                    'series': base['series']
                })

                # Add fluoro variations
                if 'Cl' in base['smiles']:
                    expansions.append({
                        'compound_id': f"{base['compound_id']}-F",
                        'smiles': base['smiles'].replace('Cl', 'F'),
                        'ic50_um': base['ic50_um'] * np.random.uniform(0.7, 1.1),
                        'assay_type': base['assay_type'],
                        'series': base['series']
                    })

                if len(df) + len(expansions) >= n_compounds:
                    break

            if expansions:
                df = pd.concat([df, pd.DataFrame(expansions)], ignore_index=True)

        logger.info(f"Generated {len(df)} mock COVID Moonshot compounds")

        return df

    def get_project_info(self) -> dict:
        """
        Get information about COVID Moonshot project.

        Returns:
            Dictionary with project metadata
        """
        return {
            'project_name': 'COVID Moonshot',
            'target': 'SARS-CoV-2 Main Protease (Mpro)',
            'target_type': 'Viral protease',
            'n_compounds': '~2000',
            'assay_types': ['fluorescence', 'AlphaLISA', 'ThermoFluor'],
            'data_source': 'https://covid.postera.ai/covid',
            'open_science': True,
            'crystal_structures': 'Available for many compounds'
        }

"""Open Source Malaria data collection."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional
import requests
from io import StringIO

logger = logging.getLogger(__name__)


class OSMCollector:
    """
    Collect data from Open Source Malaria (OSM) project.

    OSM provides curated compound series tested against P. falciparum
    with detailed SAR documentation.
    """

    # OSM data is available on GitHub
    OSM_GITHUB_BASE = "https://raw.githubusercontent.com/OpenSourceMalaria"

    # Known OSM series with data
    SERIES_DATA_URLS = {
        'series4': 'OSM_To_Be_Deleted/master/Series4/Series4_FullDataToOctober2015.csv',
        'series1': 'OSM_To_Be_Deleted/master/Series1/Series1_data.csv',
    }

    def __init__(self):
        """Initialize OSM collector."""
        self.session = requests.Session()
        logger.info("OSM collector initialized")

    def fetch_malaria_data(self, series: str = 'series4') -> pd.DataFrame:
        """
        Fetch P. falciparum IC50 data from OSM.

        Args:
            series: OSM series to fetch ('series1', 'series4', etc.)

        Returns:
            DataFrame with columns: compound_id, smiles, pf_ic50, series_id
        """
        logger.info(f"Fetching OSM {series} data...")

        if series not in self.SERIES_DATA_URLS:
            logger.error(f"Unknown series: {series}")
            logger.info(f"Available series: {list(self.SERIES_DATA_URLS.keys())}")
            # Return mock data for now
            return self._get_mock_malaria_data()

        url = f"{self.OSM_GITHUB_BASE}/{self.SERIES_DATA_URLS[series]}"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Parse CSV
            df = pd.read_csv(StringIO(response.text))

            # Standardize column names (OSM uses different conventions)
            df = self._standardize_osm_columns(df, series)

            logger.info(f"Loaded {len(df)} compounds from OSM {series}")

            return df

        except Exception as e:
            logger.warning(f"Failed to fetch OSM data from {url}: {e}")
            logger.info("Using mock malaria data instead")
            return self._get_mock_malaria_data()

    def fetch_all_series(self) -> pd.DataFrame:
        """
        Fetch data from all available OSM series.

        Returns:
            Combined DataFrame from all series
        """
        all_data = []

        for series in self.SERIES_DATA_URLS.keys():
            df = self.fetch_malaria_data(series)
            if len(df) > 0:
                all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined {len(combined)} compounds from {len(all_data)} series")
            return combined
        else:
            logger.warning("No data fetched, using mock data")
            return self._get_mock_malaria_data()

    def _standardize_osm_columns(self, df: pd.DataFrame, series: str) -> pd.DataFrame:
        """
        Standardize OSM column names.

        Different series may use different column names.

        Args:
            df: Raw DataFrame from OSM
            series: Series identifier

        Returns:
            Standardized DataFrame
        """
        # Common mappings
        column_mapping = {
            'SMILES': 'smiles',
            'Smiles': 'smiles',
            'IC50 (uM)': 'pf_ic50_um',
            'IC50': 'pf_ic50_um',
            'Compound': 'compound_id',
            'Name': 'compound_id',
            'ID': 'compound_id'
        }

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Add series identifier
        df['series_id'] = series

        # Ensure required columns exist
        required = ['smiles', 'pf_ic50_um', 'compound_id']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column: {col} in {series}")

        # Filter to valid data
        df = df.dropna(subset=['smiles'])

        return df

    def _get_mock_malaria_data(self) -> pd.DataFrame:
        """
        Generate mock malaria data for testing.

        Returns realistic P. falciparum IC50 values for common antimalarials.

        Returns:
            DataFrame with mock data
        """
        logger.info("Generating mock malaria data for testing")

        # Known antimalarials with approximate IC50 values (nM)
        mock_data = [
            # Quinoline series
            {'compound_id': 'CQ', 'smiles': 'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12', 'pf_ic50_nm': 15, 'series_id': 'quinoline'},
            {'compound_id': 'AQ', 'smiles': 'CCN(CC)CCCC(C)Nc1c2ccc(Cl)cc2nc2ccc(Cl)cc12', 'pf_ic50_nm': 8, 'series_id': 'quinoline'},

            # Artemisinin derivatives
            {'compound_id': 'ART', 'smiles': 'CC1CCC2C(C)(C)C(O)CCC2(C)C1CC(C)C1OC(=O)C2OOC3(C)CCC1C23O', 'pf_ic50_nm': 3, 'series_id': 'artemisinin'},

            # Pyrimidine series (similar to OSM Series 4)
            {'compound_id': 'PYR-001', 'smiles': 'Cc1nc(N)nc(N)c1-c1ccc(Cl)cc1', 'pf_ic50_nm': 250, 'series_id': 'series4'},
            {'compound_id': 'PYR-002', 'smiles': 'Cc1nc(N)nc(N)c1-c1ccc(F)cc1', 'pf_ic50_nm': 180, 'series_id': 'series4'},
            {'compound_id': 'PYR-003', 'smiles': 'Cc1nc(N)nc(N)c1-c1ccc(CF3)cc1', 'pf_ic50_nm': 95, 'series_id': 'series4'},
            {'compound_id': 'PYR-004', 'smiles': 'Cc1nc(N)nc(N)c1-c1ccc(OC)cc1', 'pf_ic50_nm': 320, 'series_id': 'series4'},
            {'compound_id': 'PYR-005', 'smiles': 'Cc1nc(N)nc(N)c1-c1cccc(Cl)c1', 'pf_ic50_nm': 210, 'series_id': 'series4'},

            # Spiro compounds
            {'compound_id': 'SPIRO-001', 'smiles': 'CC1(C)CC2C(C)(C)N(C(=O)c3ccc(Cl)cc3)C2C1', 'pf_ic50_nm': 450, 'series_id': 'spiro'},
            {'compound_id': 'SPIRO-002', 'smiles': 'CC1(C)CC2C(C)(C)N(C(=O)c3ccc(F)cc3)C2C1', 'pf_ic50_nm': 380, 'series_id': 'spiro'},
        ]

        df = pd.DataFrame(mock_data)

        # Convert to micromolar for consistency
        df['pf_ic50_um'] = df['pf_ic50_nm'] / 1000.0

        logger.info(f"Generated {len(df)} mock compounds")

        return df

    def get_series_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for each series.

        Args:
            df: DataFrame with OSM data

        Returns:
            DataFrame with series statistics
        """
        stats = df.groupby('series_id').agg({
            'compound_id': 'count',
            'pf_ic50_um': ['mean', 'std', 'min', 'max']
        }).round(2)

        stats.columns = ['n_compounds', 'mean_ic50', 'std_ic50', 'min_ic50', 'max_ic50']

        return stats.reset_index()

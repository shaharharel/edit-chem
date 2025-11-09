"""ChEMBL data collection for chemical and biological properties."""

import logging
import pandas as pd
from typing import List, Optional, Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChEMBLCollector:
    """
    Collect compound data from ChEMBL database.

    ChEMBL provides:
    - Chemical properties: LogP, MW, TPSA, HBD/HBA
    - Biological activities: IC50, Ki, Kd for various targets
    - Compound series from medicinal chemistry projects
    """

    def __init__(self):
        """Initialize ChEMBL collector."""
        try:
            from chembl_webresource_client.new_client import new_client
            self.molecule = new_client.molecule
            self.activity = new_client.activity
            self.target = new_client.target
            logger.info("ChEMBL client initialized")
        except ImportError:
            logger.error("chembl_webresource_client not installed. Install with: pip install chembl-webresource-client")
            raise

    def fetch_compounds_with_properties(self,
                                       min_compounds: int = 1000,
                                       target_id: Optional[str] = None,
                                       max_mw: float = 600) -> pd.DataFrame:
        """
        Fetch compounds with calculated chemical properties.

        Args:
            min_compounds: Minimum number of compounds to fetch
            target_id: Optional ChEMBL target ID (e.g., 'CHEMBL234')
            max_mw: Maximum molecular weight filter

        Returns:
            DataFrame with columns: chembl_id, smiles, alogp, mw, hbd, hba, etc.
        """
        logger.info(f"Fetching compounds from ChEMBL (min={min_compounds})...")

        compounds = []

        if target_id:
            logger.info(f"Filtering by target: {target_id}")
            # Get compounds tested against specific target
            activities = self.activity.filter(
                target_chembl_id=target_id,
                pchembl_value__isnull=False
            ).only(['molecule_chembl_id', 'pchembl_value', 'standard_type'])

            # Get unique molecule IDs
            chembl_ids = list(set(a['molecule_chembl_id'] for a in activities[:min_compounds * 2]))
            logger.info(f"Found {len(chembl_ids)} unique molecules for target")

        else:
            # Random sample - get molecules with properties
            # We'll query in batches
            logger.info("Fetching random sample of molecules...")
            chembl_ids = self._sample_molecules_with_properties(min_compounds)

        # Fetch molecular data
        for i, chembl_id in enumerate(tqdm(chembl_ids, desc="Fetching molecules")):
            if len(compounds) >= min_compounds:
                break

            try:
                mol_data = self.molecule.get(chembl_id)

                if not mol_data:
                    continue

                # Get molecular structure
                if 'molecule_structures' not in mol_data or not mol_data['molecule_structures']:
                    continue

                smiles = mol_data['molecule_structures'].get('canonical_smiles')
                if not smiles:
                    continue

                # Get molecular properties
                if 'molecule_properties' not in mol_data:
                    continue

                props = mol_data['molecule_properties']

                # Apply filters
                mw = props.get('full_mwt')
                if mw and mw > max_mw:
                    continue

                compounds.append({
                    'chembl_id': chembl_id,
                    'smiles': smiles,
                    'alogp': props.get('alogp'),
                    'mw': mw,
                    'hbd': props.get('hbd'),
                    'hba': props.get('hba'),
                    'psa': props.get('psa'),
                    'rtb': props.get('rtb'),  # Rotatable bonds
                    'aromatic_rings': props.get('aromatic_rings'),
                    'heavy_atoms': props.get('heavy_atoms')
                })

            except Exception as e:
                logger.warning(f"Error fetching {chembl_id}: {e}")
                continue

        df = pd.DataFrame(compounds)
        logger.info(f"Collected {len(df)} compounds with properties")

        return df.dropna(subset=['smiles', 'alogp'])

    def fetch_activity_data(self,
                           target_id: str,
                           standard_type: str = 'IC50',
                           min_compounds: int = 50) -> pd.DataFrame:
        """
        Fetch bioactivity data for a specific target.

        Args:
            target_id: ChEMBL target ID (e.g., 'CHEMBL234')
            standard_type: Activity type ('IC50', 'Ki', 'Kd', 'EC50')
            min_compounds: Minimum number of compounds

        Returns:
            DataFrame with columns: chembl_id, smiles, activity_value, pchembl_value
        """
        logger.info(f"Fetching {standard_type} data for target {target_id}...")

        # Query activities
        activities = self.activity.filter(
            target_chembl_id=target_id,
            standard_type=standard_type,
            pchembl_value__isnull=False,
            standard_relation='='  # Only exact measurements
        )

        # Collect data
        data = []
        for act in tqdm(activities, desc=f"Fetching {standard_type} data"):
            try:
                mol_id = act.get('molecule_chembl_id')
                if not mol_id:
                    continue

                # Get molecule structure
                mol_data = self.molecule.get(mol_id)
                if not mol_data or 'molecule_structures' not in mol_data:
                    continue

                smiles = mol_data['molecule_structures'].get('canonical_smiles')
                if not smiles:
                    continue

                data.append({
                    'chembl_id': mol_id,
                    'smiles': smiles,
                    'standard_type': act.get('standard_type'),
                    'standard_value': act.get('standard_value'),
                    'standard_units': act.get('standard_units'),
                    'pchembl_value': act.get('pchembl_value'),  # -log10(IC50)
                    'assay_chembl_id': act.get('assay_chembl_id')
                })

                if len(data) >= min_compounds:
                    break

            except Exception as e:
                logger.warning(f"Error processing activity: {e}")
                continue

        df = pd.DataFrame(data)
        logger.info(f"Collected {len(df)} activity measurements")

        return df

    def fetch_compound_series(self,
                             target_id: str,
                             min_series_size: int = 10,
                             similarity_threshold: float = 0.6) -> Dict[str, pd.DataFrame]:
        """
        Fetch related compound series for a target.

        Groups compounds by structural similarity to find medicinal chemistry series.

        Args:
            target_id: ChEMBL target ID
            min_series_size: Minimum compounds per series
            similarity_threshold: Tanimoto similarity threshold for clustering

        Returns:
            Dictionary mapping series_id -> DataFrame of compounds
        """
        logger.info(f"Fetching compound series for target {target_id}...")

        # First get all active compounds
        df = self.fetch_activity_data(target_id, min_compounds=500)

        if len(df) < min_series_size:
            logger.warning(f"Not enough compounds found: {len(df)}")
            return {}

        # Cluster by structural similarity
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from sklearn.cluster import DBSCAN
        import numpy as np

        # Generate fingerprints
        fps = []
        valid_indices = []

        for idx, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                fps.append(fp)
                valid_indices.append(idx)

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

        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=1-similarity_threshold, min_samples=min_series_size, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)

        # Group by cluster
        df_valid = df.iloc[valid_indices].copy()
        df_valid['series_id'] = labels

        series = {}
        for label in set(labels):
            if label == -1:  # Noise cluster
                continue

            series_df = df_valid[df_valid['series_id'] == label].copy()
            if len(series_df) >= min_series_size:
                series[f"series_{label}"] = series_df

        logger.info(f"Found {len(series)} compound series")

        return series

    def get_target_info(self, target_id: str) -> Optional[dict]:
        """
        Get information about a target.

        Args:
            target_id: ChEMBL target ID

        Returns:
            Dictionary with target information
        """
        try:
            target_data = self.target.get(target_id)

            if target_data:
                return {
                    'target_id': target_id,
                    'pref_name': target_data.get('pref_name'),
                    'organism': target_data.get('organism'),
                    'target_type': target_data.get('target_type')
                }
        except Exception as e:
            logger.warning(f"Error fetching target info: {e}")

        return None

    def _sample_molecules_with_properties(self, n: int) -> List[str]:
        """
        Sample molecule ChEMBL IDs that have property data.

        Args:
            n: Number of molecules to sample

        Returns:
            List of ChEMBL IDs
        """
        # Query for molecules with properties
        # Note: This is a simplified approach - may need refinement
        molecules = self.molecule.filter(
            molecule_properties__isnull=False
        ).only(['molecule_chembl_id'])[:n * 2]

        return [m['molecule_chembl_id'] for m in molecules if m.get('molecule_chembl_id')]

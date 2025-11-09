"""
ChEMBL downloader optimized for long-format pairs generation.

This approach:
1. Gets ALL molecules with ANY bioactivity (no waste!)
2. Gets ALL properties for each molecule (computed + experimental)
3. Stores in long format ready for pairs

Long format structure:
    mol_a, mol_b, edit_id, core, from_smarts, to_smarts,
    property_name, value_a, value_b, delta

Benefits:
- No NaN/missing value issues
- Only store what exists
- Efficient storage
- Easy filtering by property
- SQL-friendly
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import os

logger = logging.getLogger(__name__)


class ChEMBLLongFormat:
    """
    Download ChEMBL molecules and bioactivity optimized for long-format pairs.

    Instead of wide format with NaNs, we store molecules with ALL their properties
    (computed + experimental) and convert to long format during pair generation.
    """

    def __init__(self, output_dir: str = "data/chembl_bulk", db_dir: str = "data/chembl_db"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        os.environ['PYSTOW_HOME'] = str(self.db_dir.absolute())

        try:
            import chembl_downloader
            self.chembl = chembl_downloader
            logger.info("✓ chembl-downloader loaded")
        except ImportError:
            logger.error("✗ chembl-downloader not installed!")
            logger.error("  Install with: pip install chembl-downloader")
            raise

    def get_top_targets(
        self,
        top_n: int = 100,
        activity_types: List[str] = ['IC50', 'Ki', 'EC50', 'Kd'],
        min_molecules: int = 1000
    ) -> pd.DataFrame:
        """
        Find the most-tested targets in ChEMBL.

        Args:
            top_n: Number of top targets to return
            activity_types: Which activity types to consider
            min_molecules: Minimum molecules tested per target

        Returns:
            DataFrame with target info (chembl_id, name, count)
        """
        logger.info(f"Finding top {top_n} most-tested targets...")

        activity_filter = "', '".join(activity_types)

        query = f"""
        SELECT
            td.chembl_id as target_chembl_id,
            td.pref_name as target_name,
            td.target_type,
            COUNT(DISTINCT md.molregno) as n_molecules,
            COUNT(DISTINCT act.activity_id) as n_measurements

        FROM target_dictionary td
        JOIN assays ass ON td.tid = ass.tid
        JOIN activities act ON ass.assay_id = act.assay_id
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno

        WHERE
            act.standard_type IN ('{activity_filter}')
            AND act.pchembl_value IS NOT NULL
            AND act.standard_relation = '='
            AND cs.canonical_smiles IS NOT NULL
            AND td.target_type IN ('SINGLE PROTEIN', 'PROTEIN COMPLEX')

        GROUP BY td.chembl_id, td.pref_name, td.target_type
        HAVING COUNT(DISTINCT md.molregno) >= {min_molecules}

        ORDER BY n_molecules DESC
        LIMIT {top_n};
        """

        df = self.chembl.query(query)
        logger.info(f"  ✓ Found {len(df)} targets")

        for i, row in df.head(20).iterrows():
            logger.info(f"    {i+1}. {row['target_name']}: {row['n_molecules']:,} molecules")

        if len(df) > 20:
            logger.info(f"    ... and {len(df) - 20} more")

        logger.info("")

        return df

    def download_all_with_bioactivity(
        self,
        max_molecules: int = 1000000,
        max_mw: float = 800,
        min_mw: float = 100,
        activity_types: List[str] = ['IC50', 'Ki', 'EC50', 'Kd'],
        top_targets: Optional[int] = None,
        computed_properties: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Download molecules with bioactivity data.

        Two modes:
        1. top_targets=None: Get ALL molecules with ANY bioactivity (default)
        2. top_targets=N: Get molecules tested on top N targets (OPTIMIZED FOR PAIRS!)

        Args:
            max_molecules: Maximum molecules to download
            max_mw: Maximum molecular weight
            min_mw: Minimum molecular weight
            activity_types: Activity types to include
            top_targets: If set, only get molecules tested on top N targets
                        (RECOMMENDED: 100-200 for maximum pair density!)

        Returns:
            DataFrame with molecules + all computed properties
        """
        logger.info("=" * 70)
        logger.info(" CHEMBL LONG-FORMAT DOWNLOAD")
        logger.info("=" * 70)

        if top_targets:
            logger.info(f" Strategy: Get molecules from TOP {top_targets} targets (OPTIMIZED FOR PAIRS!)")
        else:
            logger.info(f" Strategy: Get ALL molecules with ANY bioactivity")

        logger.info(f" Max molecules: {max_molecules:,}")
        logger.info(f" MW range: {min_mw}-{max_mw}")
        logger.info(f" Activity types: {', '.join(activity_types)}")
        logger.info("=" * 70)
        logger.info("")

        activity_filter = "', '".join(activity_types)

        # Optional: Get top targets first
        target_filter = None
        if top_targets:
            logger.info("Step 0: Finding top targets...")
            targets_df = self.get_top_targets(
                top_n=top_targets,
                activity_types=activity_types,
                min_molecules=1000
            )

            if len(targets_df) == 0:
                raise ValueError("No targets found!")

            target_ids = targets_df['target_chembl_id'].tolist()
            target_filter = "', '".join(target_ids)

            logger.info(f"  ✓ Will download molecules from {len(target_ids)} targets")
            logger.info(f"  Total molecules across these targets: {targets_df['n_molecules'].sum():,}")
            logger.info("")

        # Step 1: Get molecules with bioactivity
        if top_targets:
            logger.info(f"Step 1: Downloading molecules from top {top_targets} targets...")
        else:
            logger.info("Step 1: Downloading molecules with ANY bioactivity...")

        # Build computed properties SELECT clause
        if computed_properties is None:
            # Default: all properties
            computed_props = [
                'alogp', 'mw', 'mw_freebase', 'hbd', 'hba', 'psa', 'rtb',
                'aromatic_rings', 'heavy_atoms', 'qed_weighted',
                'num_ro5_violations', 'ro3_pass', 'full_molformula', 'np_likeness_score'
            ]
        else:
            computed_props = computed_properties

        # Map property names to SQL columns
        prop_to_sql = {
            'mw': 'cp.full_mwt AS mw',
            'alogp': 'cp.alogp',
            'mw_freebase': 'cp.mw_freebase',
            'hbd': 'cp.hbd',
            'hba': 'cp.hba',
            'psa': 'cp.psa',
            'rtb': 'cp.rtb',
            'aromatic_rings': 'cp.aromatic_rings',
            'heavy_atoms': 'cp.heavy_atoms',
            'qed_weighted': 'cp.qed_weighted',
            'num_ro5_violations': 'cp.num_ro5_violations',
            'ro3_pass': 'cp.ro3_pass',
            'full_molformula': 'cp.full_molformula',
            'np_likeness_score': 'cp.np_likeness_score',
            # Drug metadata (from molecule_dictionary)
            'max_phase': 'md.max_phase',
            'oral': 'md.oral',
            'parenteral': 'md.parenteral',
            'topical': 'md.topical'
        }

        # Build SQL columns
        sql_columns = []
        for prop in computed_props:
            if prop in prop_to_sql:
                sql_columns.append(prop_to_sql[prop])

        computed_props_sql = ',\n            '.join(sql_columns)

        logger.info(f"  Downloading {len(computed_props)} computed properties: {computed_props}")

        query_molecules = f"""
        SELECT DISTINCT
            md.chembl_id,
            cs.canonical_smiles AS smiles,
            cs.standard_inchi_key AS inchi_key,

            -- Computed properties (dynamic based on user selection)
            {computed_props_sql},

            -- Always include molecule_type
            md.molecule_type

        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN compound_properties cp ON md.molregno = cp.molregno

        WHERE
            cp.full_mwt >= {min_mw}
            AND cp.full_mwt <= {max_mw}
            AND cs.canonical_smiles IS NOT NULL

            -- Has bioactivity data
            AND EXISTS (
                SELECT 1
                FROM activities act
                """

        # Add target filter if specified
        if target_filter:
            query_molecules += f"""
                JOIN assays ass ON act.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                WHERE act.molregno = md.molregno
                AND act.standard_type IN ('{activity_filter}')
                AND act.pchembl_value IS NOT NULL
                AND act.standard_relation = '='
                AND td.chembl_id IN ('{target_filter}')
                LIMIT 1
            )

        LIMIT {max_molecules};
        """
        else:
            query_molecules += f"""
                WHERE act.molregno = md.molregno
                AND act.standard_type IN ('{activity_filter}')
                AND act.pchembl_value IS NOT NULL
                AND act.standard_relation = '='
                LIMIT 1
            )

        LIMIT {max_molecules};
        """

        df_molecules = self.chembl.query(query_molecules)
        logger.info(f"  ✓ Retrieved {len(df_molecules):,} molecules with bioactivity")
        logger.info("")

        # Save molecules
        molecules_file = self.output_dir / f"chembl_molecules_{len(df_molecules)}.csv"
        df_molecules.to_csv(molecules_file, index=False)
        logger.info(f"  Saved molecules to: {molecules_file}")
        logger.info("")

        return df_molecules

    def download_all_bioactivity(
        self,
        molecules_df: pd.DataFrame,
        activity_types: List[str] = ['IC50', 'Ki', 'EC50', 'Kd'],
        target_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Download ALL bioactivity data for given molecules.

        Returns long-format data: one row per molecule-target-assay combination.

        Args:
            molecules_df: DataFrame with molecules (must have 'chembl_id' column)
            activity_types: Activity types to include
            target_filter: Optional list of target ChEMBL IDs to filter by

        Returns:
            Long-format DataFrame with bioactivity
        """
        logger.info("Step 2: Downloading bioactivity data...")
        logger.info(f"  Fetching bioactivity for {len(molecules_df):,} molecules...")
        if target_filter:
            logger.info(f"  Filtering to {len(target_filter)} targets")
        logger.info("")

        activity_filter = "', '".join(activity_types)

        # Get chembl_ids
        chembl_ids = molecules_df['chembl_id'].tolist()

        # Process in chunks to avoid SQL query length limits
        chunk_size = 5000
        chunks = [chembl_ids[i:i+chunk_size] for i in range(0, len(chembl_ids), chunk_size)]

        all_bioactivity = []

        for i, chunk in enumerate(chunks):
            logger.info(f"  Processing chunk {i+1}/{len(chunks)} ({len(chunk)} molecules)...")

            chembl_filter = "', '".join(chunk)

            # Build WHERE clause with optional target filter
            where_clause = f"""
            WHERE
                md.chembl_id IN ('{chembl_filter}')
                AND act.standard_type IN ('{activity_filter}')
                AND act.pchembl_value IS NOT NULL
                AND act.standard_relation = '='
                AND td.target_type IN ('SINGLE PROTEIN', 'PROTEIN COMPLEX')
            """

            if target_filter:
                target_filter_str = "', '".join(target_filter)
                where_clause += f"\n                AND td.chembl_id IN ('{target_filter_str}')"

            query_bioactivity = f"""
            SELECT
                md.chembl_id,
                td.chembl_id as target_chembl_id,
                td.pref_name as target_name,
                td.target_type,
                act.standard_type as activity_type,
                act.pchembl_value,
                act.standard_value,
                act.standard_units

            FROM activities act
            JOIN assays ass ON act.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN molecule_dictionary md ON act.molregno = md.molregno

            {where_clause}
            """

            df_chunk = self.chembl.query(query_bioactivity)
            all_bioactivity.append(df_chunk)

        df_bioactivity = pd.concat(all_bioactivity, ignore_index=True)
        logger.info(f"  ✓ Retrieved {len(df_bioactivity):,} bioactivity measurements")
        logger.info("")

        # Create property_name column (target + activity type)
        df_bioactivity['property_name'] = (
            df_bioactivity['target_name'].str.lower().str.replace(' ', '_').str.replace('-', '_').str[:20] +
            '_' +
            df_bioactivity['activity_type'].str.lower()
        )

        # For duplicates (same molecule-target-assay), take median
        df_bioactivity_agg = df_bioactivity.groupby(['chembl_id', 'property_name']).agg({
            'pchembl_value': 'median',
            'target_chembl_id': 'first',
            'target_name': 'first',
            'activity_type': 'first'
        }).reset_index()

        # Save bioactivity in long format
        bioactivity_file = self.output_dir / f"chembl_bioactivity_long_{len(molecules_df)}.csv"
        df_bioactivity_agg.to_csv(bioactivity_file, index=False)
        logger.info(f"  Saved bioactivity to: {bioactivity_file}")
        logger.info("")

        # Statistics
        n_unique_properties = df_bioactivity_agg['property_name'].nunique()
        n_unique_targets = df_bioactivity_agg['target_chembl_id'].nunique()
        avg_properties_per_mol = len(df_bioactivity_agg) / len(molecules_df)

        logger.info("  Statistics:")
        logger.info(f"    Unique targets: {n_unique_targets:,}")
        logger.info(f"    Unique properties: {n_unique_properties:,}")
        logger.info(f"    Avg properties/molecule: {avg_properties_per_mol:.1f}")
        logger.info("")

        return df_bioactivity_agg

    def download_target_sequences(
        self,
        bioactivity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Download protein sequences for all targets in bioactivity data.

        Args:
            bioactivity_df: DataFrame with bioactivity (must have 'target_chembl_id' column)

        Returns:
            DataFrame with target sequences (chembl_id, name, organism, sequence, uniprot_id)
        """
        logger.info("Step 3: Downloading target sequences...")

        # Get unique targets
        unique_targets = bioactivity_df['target_chembl_id'].unique()
        logger.info(f"  Found {len(unique_targets)} unique targets")

        # Download sequences in chunks
        chunk_size = 100
        chunks = [unique_targets[i:i+chunk_size] for i in range(0, len(unique_targets), chunk_size)]

        all_targets = []

        for i, chunk in enumerate(chunks):
            logger.info(f"  Processing chunk {i+1}/{len(chunks)} ({len(chunk)} targets)...")

            target_filter = "', '".join(chunk)

            query_targets = f"""
            SELECT DISTINCT
                td.chembl_id,
                td.pref_name as target_name,
                td.organism,
                td.target_type,
                cs.accession as uniprot_id,
                cs.sequence,
                LENGTH(cs.sequence) as seq_length
            FROM target_dictionary td
            JOIN target_components tc ON td.tid = tc.tid
            JOIN component_sequences cs ON tc.component_id = cs.component_id
            WHERE td.chembl_id IN ('{target_filter}')
            """

            df_chunk = self.chembl.query(query_targets)
            all_targets.append(df_chunk)

        df_targets = pd.concat(all_targets, ignore_index=True)
        logger.info(f"  ✓ Retrieved sequences for {len(df_targets)} targets")
        logger.info("")

        # Save targets
        targets_file = self.output_dir / f"chembl_target_sequences_{len(df_targets)}.csv"
        df_targets.to_csv(targets_file, index=False)
        logger.info(f"  Saved target sequences to: {targets_file}")
        logger.info("")

        # Show some examples
        logger.info("  Example targets:")
        for _, row in df_targets.head(5).iterrows():
            logger.info(f"    {row['chembl_id']}: {row['target_name']} ({row['seq_length']} aa)")
        logger.info("")

        return df_targets

    def create_property_lookup(
        self,
        molecules_df: pd.DataFrame,
        bioactivity_df: pd.DataFrame,
        computed_properties: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Create efficient lookup: {chembl_id: {property_name: value}}

        This includes both computed properties and bioactivity.

        Args:
            molecules_df: DataFrame with computed properties
            bioactivity_df: DataFrame with bioactivity (long format)
            computed_properties: List of computed properties to include (default: all)

        Returns:
            Nested dict for fast property lookup
        """
        logger.info("Step 3: Creating property lookup...")
        from tqdm import tqdm

        lookup = {}

        # Create chembl_id -> SMILES mapping (O(N) instead of O(N²))
        logger.info("  Building ChEMBL ID index...")
        chembl_to_smiles = dict(zip(molecules_df['chembl_id'], molecules_df['smiles']))
        logger.info(f"  ✓ Indexed {len(chembl_to_smiles):,} molecules")

        # Add computed properties
        if computed_properties is None:
            # Default: all properties
            computed_props = [
                'alogp', 'mw', 'mw_freebase', 'hbd', 'hba', 'psa', 'rtb',
                'aromatic_rings', 'heavy_atoms', 'qed_weighted',
                'num_ro5_violations', 'np_likeness_score'
            ]
        else:
            # Use specified properties
            computed_props = computed_properties

        logger.info(f"  Adding {len(computed_props)} computed properties: {computed_props}")
        for _, row in tqdm(molecules_df.iterrows(), total=len(molecules_df), desc="  Molecules"):
            chembl_id = row['chembl_id']
            smiles = row['smiles']

            lookup[smiles] = {
                'chembl_id': chembl_id,
                'properties': {}
            }

            # Add computed properties
            for prop in computed_props:
                if prop in row and pd.notna(row[prop]):
                    lookup[smiles]['properties'][prop] = row[prop]

        # Add bioactivity properties (now O(N) with dict lookup!)
        logger.info("  Adding bioactivity properties...")
        for _, row in tqdm(bioactivity_df.iterrows(), total=len(bioactivity_df), desc="  Bioactivity"):
            chembl_id = row['chembl_id']

            # Fast dict lookup instead of slow DataFrame filter!
            smiles = chembl_to_smiles.get(chembl_id)
            if smiles is None:
                continue

            if smiles in lookup:
                prop_name = row['property_name']
                value = row['pchembl_value']

                if pd.notna(value):
                    lookup[smiles]['properties'][prop_name] = value

        logger.info(f"  ✓ Created lookup for {len(lookup):,} molecules")
        logger.info("")

        return lookup

    def download_complete(
        self,
        max_molecules: int = 1000000,
        max_mw: float = 800,
        min_mw: float = 100,
        activity_types: List[str] = ['IC50', 'Ki', 'EC50', 'Kd'],
        top_targets: Optional[int] = None,
        computed_properties: Optional[List[str]] = None,
        download_target_sequences: bool = False
    ) -> tuple:
        """
        Complete download pipeline.

        Args:
            computed_properties: List of computed properties to include (default: all)
            download_target_sequences: If True, also download protein sequences for all targets (default: False)

        Returns:
            (molecules_df, bioactivity_df, property_lookup) if download_target_sequences=False
            (molecules_df, bioactivity_df, property_lookup, targets_df) if download_target_sequences=True
        """
        # Get top targets if specified
        top_targets_df = None
        target_filter = None
        if top_targets:
            top_targets_df = self.get_top_targets(
                top_n=top_targets,
                activity_types=activity_types
            )
            target_filter = top_targets_df['target_chembl_id'].tolist()

        # Download molecules
        molecules_df = self.download_all_with_bioactivity(
            max_molecules=max_molecules,
            max_mw=max_mw,
            min_mw=min_mw,
            activity_types=activity_types,
            top_targets=top_targets,
            computed_properties=computed_properties
        )

        # Download bioactivity (filtered by targets if specified)
        bioactivity_df = self.download_all_bioactivity(
            molecules_df=molecules_df,
            activity_types=activity_types,
            target_filter=target_filter
        )

        # Download target sequences if requested
        targets_df = None
        if download_target_sequences:
            targets_df = self.download_target_sequences(bioactivity_df)

        # Create lookup
        property_lookup = self.create_property_lookup(
            molecules_df=molecules_df,
            bioactivity_df=bioactivity_df,
            computed_properties=computed_properties
        )

        logger.info("=" * 70)
        logger.info(" SUCCESS!")
        logger.info("=" * 70)
        logger.info(f" Molecules: {len(molecules_df):,}")
        logger.info(f" Bioactivity measurements: {len(bioactivity_df):,}")
        logger.info(f" Unique properties: {bioactivity_df['property_name'].nunique():,}")
        if targets_df is not None:
            logger.info(f" Target sequences: {len(targets_df)}")
        logger.info("=" * 70)
        logger.info("")

        if download_target_sequences:
            return molecules_df, bioactivity_df, property_lookup, targets_df
        else:
            return molecules_df, bioactivity_df, property_lookup


def main():
    """Main script for long-format download."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download ChEMBL optimized for long-format pairs"
    )

    parser.add_argument('--max-molecules', type=int, default=1000000,
                       help='Maximum molecules to download (default: 1000000)')
    parser.add_argument('--max-mw', type=float, default=800,
                       help='Maximum molecular weight (default: 800)')
    parser.add_argument('--min-mw', type=float, default=100,
                       help='Minimum molecular weight (default: 100)')
    parser.add_argument('--activity-types', nargs='+',
                       default=['IC50', 'Ki', 'EC50', 'Kd'],
                       help='Activity types (default: IC50 Ki EC50 Kd)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Create downloader
    downloader = ChEMBLLongFormat()

    # Download everything
    molecules_df, bioactivity_df, property_lookup = downloader.download_complete(
        max_molecules=args.max_molecules,
        max_mw=args.max_mw,
        min_mw=args.min_mw,
        activity_types=args.activity_types
    )

    print()
    print("=" * 70)
    print(" FILES CREATED")
    print("=" * 70)
    print(f" 1. data/chembl_bulk/chembl_molecules_{len(molecules_df)}.csv")
    print(f"    - {len(molecules_df):,} molecules with computed properties")
    print()
    print(f" 2. data/chembl_bulk/chembl_bioactivity_long_{len(molecules_df)}.csv")
    print(f"    - {len(bioactivity_df):,} bioactivity measurements")
    print(f"    - {bioactivity_df['property_name'].nunique():,} unique properties")
    print()
    print("=" * 70)
    print()
    print(" NEXT STEP: Generate pairs")
    print()
    print("   python build_pairs_long_format.py \\")
    print(f"       --molecules-file data/chembl_bulk/chembl_molecules_{len(molecules_df)}.csv \\")
    print(f"       --bioactivity-file data/chembl_bulk/chembl_bioactivity_long_{len(molecules_df)}.csv")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()

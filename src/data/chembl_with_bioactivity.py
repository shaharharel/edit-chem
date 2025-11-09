"""
Enhanced ChEMBL downloader that includes experimental bioactivity data.

This adds experimental properties (IC50, Ki, EC50, etc.) to molecules
in addition to the computed properties.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import os

logger = logging.getLogger(__name__)


class ChEMBLWithBioactivity:
    """
    Download ChEMBL molecules with both computed AND experimental properties.

    Experimental properties include:
    - IC50, Ki, Kd, EC50 measurements
    - Target information
    - Assay descriptions
    - pChEMBL values (standardized activity)
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
            logger.info("chembl-downloader loaded successfully")
        except ImportError:
            logger.error("chembl-downloader not installed!")
            raise

    def download_with_aggregated_bioactivity(
        self,
        n_molecules: int = 100000,
        max_mw: float = 600,
        min_mw: float = 100,
        activity_types: List[str] = ['IC50', 'Ki', 'EC50', 'Kd'],
        min_activities_per_molecule: int = 1
    ) -> pd.DataFrame:
        """
        Download molecules with aggregated experimental bioactivity data.

        For each molecule, this computes:
        - Number of experimental measurements
        - Median pChEMBL value across all assays
        - Number of unique targets tested
        - Activity types available

        Args:
            n_molecules: Number of molecules
            max_mw: Max molecular weight
            min_mw: Min molecular weight
            activity_types: Which activity types to include
            min_activities_per_molecule: Minimum experimental measurements required

        Returns:
            DataFrame with molecules + aggregated bioactivity stats
        """
        logger.info("=" * 60)
        logger.info("CHEMBL WITH EXPERIMENTAL BIOACTIVITY")
        logger.info("=" * 60)
        logger.info(f"Target: {n_molecules:,} molecules")
        logger.info(f"Activity types: {', '.join(activity_types)}")
        logger.info(f"Min activities: {min_activities_per_molecule}")
        logger.info("=" * 60)
        logger.info("")

        # Build query for molecules with bioactivity
        activity_filter = "', '".join(activity_types)

        query = f"""
        WITH molecule_activities AS (
            SELECT
                md.molregno,
                md.chembl_id,
                cs.canonical_smiles AS smiles,
                cs.standard_inchi_key AS inchi_key,

                -- Computed properties
                cp.alogp,
                cp.full_mwt AS mw,
                cp.mw_freebase,
                cp.hbd,
                cp.hba,
                cp.psa,
                cp.rtb,
                cp.aromatic_rings,
                cp.heavy_atoms,
                cp.qed_weighted,
                cp.num_ro5_violations,
                cp.ro3_pass,
                cp.full_molformula,
                cp.np_likeness_score,

                -- Metadata
                md.molecule_type,
                md.max_phase,
                md.oral,
                md.parenteral,
                md.topical,

                -- Experimental bioactivity aggregates
                COUNT(DISTINCT act.activity_id) as n_activities,
                COUNT(DISTINCT act.standard_type) as n_activity_types,
                COUNT(DISTINCT td.chembl_id) as n_targets,
                AVG(act.pchembl_value) as mean_pchembl,
                MEDIAN(act.pchembl_value) as median_pchembl,
                MIN(act.pchembl_value) as min_pchembl,
                MAX(act.pchembl_value) as max_pchembl,
                GROUP_CONCAT(DISTINCT act.standard_type) as activity_types_tested,
                GROUP_CONCAT(DISTINCT td.pref_name) as target_names

            FROM molecule_dictionary md
            JOIN compound_structures cs ON md.molregno = cs.molregno
            JOIN compound_properties cp ON md.molregno = cp.molregno
            JOIN activities act ON md.molregno = act.molregno
            JOIN assays ass ON act.assay_id = ass.assay_id
            LEFT JOIN target_dictionary td ON ass.tid = td.tid

            WHERE
                cp.full_mwt >= {min_mw}
                AND cp.full_mwt <= {max_mw}
                AND cs.canonical_smiles IS NOT NULL
                AND act.standard_type IN ('{activity_filter}')
                AND act.pchembl_value IS NOT NULL
                AND act.standard_relation = '='

            GROUP BY md.molregno
            HAVING COUNT(DISTINCT act.activity_id) >= {min_activities_per_molecule}
        )

        SELECT * FROM molecule_activities
        LIMIT {n_molecules};
        """

        logger.info("Querying database...")
        df = self.chembl.query(query)

        logger.info(f"Retrieved {len(df):,} molecules with bioactivity data")

        # Save
        output_file = self.output_dir / f"chembl_with_bioactivity_{len(df)}.csv"
        df.to_csv(output_file, index=False)

        logger.info("=" * 60)
        logger.info(f"SUCCESS: {len(df):,} molecules with experimental data")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 60)

        return df

    def download_with_specific_target(
        self,
        target_chembl_id: str,
        activity_type: str = 'IC50',
        n_molecules: int = 10000,
        max_mw: float = 600,
        min_mw: float = 100
    ) -> pd.DataFrame:
        """
        Download molecules tested against a specific target.

        Example targets:
        - CHEMBL203: EGFR
        - CHEMBL1862: Acetylcholinesterase
        - CHEMBL204: Thrombin
        - CHEMBL1824: Dopamine D2 receptor

        Args:
            target_chembl_id: ChEMBL target ID
            activity_type: IC50, Ki, EC50, etc.
            n_molecules: Max molecules to retrieve
            max_mw: Max molecular weight
            min_mw: Min molecular weight

        Returns:
            DataFrame with molecules and their experimental values for this target
        """
        logger.info(f"Downloading molecules tested against {target_chembl_id}")
        logger.info(f"Activity type: {activity_type}")

        query = f"""
        SELECT
            md.chembl_id,
            cs.canonical_smiles AS smiles,
            cs.standard_inchi_key AS inchi_key,

            -- Computed properties
            cp.alogp,
            cp.full_mwt AS mw,
            cp.hbd,
            cp.hba,
            cp.psa,
            cp.qed_weighted,

            -- Experimental data for this target
            act.standard_type,
            act.standard_value,
            act.standard_units,
            act.pchembl_value,
            act.standard_relation,
            td.pref_name as target_name,
            td.target_type,
            ass.assay_type,
            ass.assay_organism,

            -- Metadata
            md.max_phase

        FROM activities act
        JOIN molecule_dictionary md ON act.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN compound_properties cp ON md.molregno = cp.molregno
        JOIN assays ass ON act.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid

        WHERE
            td.chembl_id = '{target_chembl_id}'
            AND act.standard_type = '{activity_type}'
            AND act.pchembl_value IS NOT NULL
            AND act.standard_relation = '='
            AND cp.full_mwt >= {min_mw}
            AND cp.full_mwt <= {max_mw}
            AND cs.canonical_smiles IS NOT NULL

        ORDER BY act.pchembl_value DESC
        LIMIT {n_molecules};
        """

        df = self.chembl.query(query)

        logger.info(f"Retrieved {len(df):,} molecules tested against {target_chembl_id}")

        # Save
        output_file = self.output_dir / f"chembl_{target_chembl_id}_{activity_type}_{len(df)}.csv"
        df.to_csv(output_file, index=False)

        return df

    def download_diverse_with_bioactivity(
        self,
        n_molecules: int = 100000,
        max_mw: float = 600,
        min_mw: float = 100
    ) -> pd.DataFrame:
        """
        Download diverse molecules that have SOME experimental bioactivity data.

        This uses a fast query with subquery to check for bioactivity.

        Args:
            n_molecules: Number of molecules
            max_mw: Max molecular weight
            min_mw: Min molecular weight

        Returns:
            DataFrame with molecules that have bioactivity data
        """
        logger.info("Downloading molecules with experimental data (fast query)...")

        # Much faster: use subquery with EXISTS instead of JOIN + DISTINCT
        query = f"""
        SELECT
            md.chembl_id,
            cs.canonical_smiles AS smiles,
            cs.standard_inchi_key AS inchi_key,

            -- Computed properties
            cp.alogp,
            cp.full_mwt AS mw,
            cp.mw_freebase,
            cp.hbd,
            cp.hba,
            cp.psa,
            cp.rtb,
            cp.aromatic_rings,
            cp.heavy_atoms,
            cp.qed_weighted,
            cp.num_ro5_violations,
            cp.ro3_pass,
            cp.full_molformula,
            cp.np_likeness_score,

            -- Metadata
            md.molecule_type,
            md.max_phase,
            md.oral,
            md.parenteral,
            md.topical

        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN compound_properties cp ON md.molregno = cp.molregno

        WHERE
            cp.full_mwt >= {min_mw}
            AND cp.full_mwt <= {max_mw}
            AND cs.canonical_smiles IS NOT NULL
            AND cp.alogp IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM activities act
                WHERE act.molregno = md.molregno
                AND act.pchembl_value IS NOT NULL
                LIMIT 1
            )

        LIMIT {n_molecules};
        """

        df = self.chembl.query(query)

        logger.info(f"Retrieved {len(df):,} molecules with experimental data")

        # Save
        output_file = self.output_dir / f"chembl_with_any_bioactivity_{len(df)}.csv"
        df.to_csv(output_file, index=False)

        return df


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Download ChEMBL with experimental bioactivity")
    parser.add_argument('--mode', choices=['aggregated', 'target', 'diverse'], default='diverse',
                       help='Download mode')
    parser.add_argument('--target', type=int, default=100000,
                       help='Number of molecules')
    parser.add_argument('--target-id', type=str,
                       help='ChEMBL target ID (for target mode)')
    parser.add_argument('--activity-type', type=str, default='IC50',
                       help='Activity type (IC50, Ki, etc.)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    downloader = ChEMBLWithBioactivity()

    if args.mode == 'aggregated':
        df = downloader.download_with_aggregated_bioactivity(
            n_molecules=args.target,
            activity_types=['IC50', 'Ki', 'EC50', 'Kd']
        )
    elif args.mode == 'target':
        if not args.target_id:
            print("Error: --target-id required for target mode")
            return
        df = downloader.download_with_specific_target(
            target_chembl_id=args.target_id,
            activity_type=args.activity_type,
            n_molecules=args.target
        )
    else:  # diverse
        df = downloader.download_diverse_with_bioactivity(
            n_molecules=args.target
        )

    print()
    print(f"Downloaded {len(df)} molecules")
    print(f"Columns: {list(df.columns)}")
    print()
    print("Sample:")
    print(df.head())


if __name__ == '__main__':
    main()

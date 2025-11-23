"""
Fast ChEMBL bulk data download using chembl-downloader.

This is MUCH faster than API calls:
- API approach: 50,000 molecules = 27 hours
- Bulk download: 2.5 million molecules = 1-2 hours

Uses chembl-downloader to get SQLite database, then queries for properties.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import os

logger = logging.getLogger(__name__)


class ChEMBLBulkDownloader:
    """
    Fast bulk download of ChEMBL data using chembl-downloader package.

    This downloads the entire ChEMBL SQLite database (~5 GB) once,
    then queries it locally - MUCH faster than API calls.

    Database location: data/chembl_db/ (in project folder)
    """

    def __init__(self, output_dir: str = "data/chembl_bulk", db_dir: str = "data/chembl_db"):
        """
        Initialize bulk downloader.

        Args:
            output_dir: Directory for output CSV data
            db_dir: Directory for ChEMBL database (default: data/chembl_db)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Database directory in project folder
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variable to store database in project
        # This tells chembl-downloader where to cache the database
        os.environ['PYSTOW_HOME'] = str(self.db_dir.absolute())

        # Try to import chembl-downloader
        try:
            import chembl_downloader
            self.chembl = chembl_downloader
            logger.info("chembl-downloader loaded successfully")
            logger.info(f"Database will be stored in: {self.db_dir.absolute()}")
        except ImportError:
            logger.error("chembl-downloader not installed!")
            logger.error("Install with: pip install chembl-downloader")
            raise

    def check_database_cached(self, version: Optional[str] = None) -> bool:
        """
        Check if ChEMBL database is already downloaded.

        Args:
            version: ChEMBL version (None = latest)

        Returns:
            True if database exists, False otherwise
        """
        if version is None:
            version = self.chembl.latest()

        # Database is stored in PYSTOW_HOME/chembl/{version}/
        db_path = self.db_dir / 'chembl' / str(version) / f'chembl_{version}.db'
        return db_path.exists()

    def download_molecules_with_properties(self,
                                          n_molecules: int = 50000,
                                          max_mw: float = 600,
                                          min_mw: float = 100,
                                          require_alogp: bool = True,
                                          version: Optional[str] = None) -> pd.DataFrame:
        """
        Download molecules with properties from ChEMBL.

        This method:
        1. Downloads ChEMBL SQLite database (~5 GB, one-time download)
        2. Queries for molecules with desired properties
        3. Returns DataFrame with SMILES and properties

        Args:
            n_molecules: Number of molecules to fetch
            max_mw: Maximum molecular weight
            min_mw: Minimum molecular weight
            require_alogp: Require ALogP values
            version: ChEMBL version (None = latest)

        Returns:
            DataFrame with molecules and properties
        """
        logger.info("=" * 60)
        logger.info("FAST CHEMBL BULK DOWNLOAD")
        logger.info("=" * 60)
        logger.info(f"Target: {n_molecules:,} molecules")
        logger.info(f"MW range: {min_mw}-{max_mw}")
        logger.info(f"Require ALogP: {require_alogp}")
        logger.info("=" * 60)
        logger.info("")

        # Check if database is cached
        if self.check_database_cached(version):
            logger.info("✓ ChEMBL database found in cache")
            logger.info(f"  Location: {self.db_dir.absolute()}")
            logger.info("  Skipping download...")
        else:
            logger.info("Step 1: Downloading ChEMBL database...")
            logger.info("(This is a one-time download, ~5 GB)")
            logger.info(f"Database will be saved to: {self.db_dir.absolute()}")
            logger.info("Future runs will use cached database")

        logger.info("")

        # Build SQL query
        query = self._build_query(
            n_molecules=n_molecules,
            max_mw=max_mw,
            min_mw=min_mw,
            require_alogp=require_alogp
        )

        logger.info("Step 2: Querying database...")
        logger.info(f"Running SQL query for {n_molecules:,} molecules")

        # Execute query using chembl-downloader
        df = self.chembl.query(query, version=version)

        logger.info(f"Retrieved {len(df):,} molecules")

        # Save to CSV
        output_file = self.output_dir / f"chembl_molecules_{len(df)}.csv"
        df.to_csv(output_file, index=False)

        logger.info("=" * 60)
        logger.info(f"SUCCESS: Downloaded {len(df):,} molecules")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 60)

        return df

    def _build_query(self,
                    n_molecules: int,
                    max_mw: float,
                    min_mw: float,
                    require_alogp: bool) -> str:
        """
        Build SQL query for ChEMBL database.

        ChEMBL schema:
        - molecule_dictionary: molecule info
        - compound_structures: SMILES, InChI
        - compound_properties: calculated properties

        Available columns in compound_properties (ChEMBL 36):
        - molregno, mw_freebase, alogp, hba, hbd, psa, rtb
        - ro3_pass, num_ro5_violations, full_mwt, aromatic_rings
        - heavy_atoms, qed_weighted, full_molformula, np_likeness_score

        Args:
            n_molecules: Number of molecules
            max_mw: Max molecular weight
            min_mw: Min molecular weight
            require_alogp: Require ALogP

        Returns:
            SQL query string
        """
        query = f"""
        SELECT
            md.chembl_id,
            cs.canonical_smiles AS smiles,
            cs.standard_inchi_key AS inchi_key,

            -- Chemical properties (calculated)
            cp.alogp,
            cp.full_mwt AS mw,
            cp.mw_freebase,

            -- H-bonding
            cp.hbd,
            cp.hba,

            -- Other descriptors
            cp.psa,
            cp.rtb,
            cp.aromatic_rings,
            cp.heavy_atoms,

            -- Drug-likeness
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
        """

        if require_alogp:
            query += "    AND cp.alogp IS NOT NULL\n"

        query += f"LIMIT {n_molecules};"

        return query

    def get_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get property statistics.

        Args:
            df: DataFrame with molecules

        Returns:
            Statistics DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numeric_cols].describe()

        logger.info("\nProperty Statistics:")
        print(stats)

        return stats

    def download_specific_targets(self,
                                  target_chembl_id: str,
                                  activity_type: str = 'IC50',
                                  min_activities: int = 1000) -> pd.DataFrame:
        """
        Download molecules with activity data for specific target.

        Args:
            target_chembl_id: ChEMBL target ID (e.g., 'CHEMBL203' for EGFR)
            activity_type: Activity type (IC50, Ki, Kd, EC50)
            min_activities: Minimum activities to fetch

        Returns:
            DataFrame with molecules and activities
        """
        logger.info(f"Downloading {activity_type} data for {target_chembl_id}...")

        query = f"""
        SELECT
            md.chembl_id,
            cs.canonical_smiles AS smiles,
            act.standard_type,
            act.standard_value,
            act.standard_units,
            act.pchembl_value,
            cp.alogp,
            cp.full_mwt AS mw,
            cp.hbd,
            cp.hba,
            cp.psa

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
            AND cs.canonical_smiles IS NOT NULL

        LIMIT {min_activities};
        """

        df = self.chembl.query(query)

        logger.info(f"Retrieved {len(df):,} {activity_type} measurements")

        return df

    def iterate_all_smiles(self, limit: Optional[int] = None):
        """
        Iterate through all SMILES in ChEMBL.

        This is a generator for memory-efficient processing.

        Args:
            limit: Optional limit on number of molecules

        Yields:
            (chembl_id, smiles) tuples
        """
        logger.info("Iterating through ChEMBL SMILES...")

        # Use chembl-downloader's iterate_smiles
        count = 0
        for chembl_id, smiles in self.chembl.iterate_smiles():
            yield chembl_id, smiles

            count += 1
            if limit and count >= limit:
                break

        logger.info(f"Processed {count:,} SMILES")


def main():
    """
    Main script for fast bulk download.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Fast ChEMBL bulk download")
    parser.add_argument('--target', type=int, default=50000,
                       help='Number of molecules (default: 50000)')
    parser.add_argument('--max-mw', type=float, default=600,
                       help='Max molecular weight (default: 600)')
    parser.add_argument('--min-mw', type=float, default=100,
                       help='Min molecular weight (default: 100)')
    parser.add_argument('--output', type=str, default='data/chembl_bulk',
                       help='Output directory')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create downloader
    downloader = ChEMBLBulkDownloader(output_dir=args.output)

    # Download data
    df = downloader.download_molecules_with_properties(
        n_molecules=args.target,
        max_mw=args.max_mw,
        min_mw=args.min_mw
    )

    # Show statistics
    downloader.get_statistics(df)


if __name__ == '__main__':
    main()

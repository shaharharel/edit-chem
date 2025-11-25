#!/usr/bin/env python
"""
Production ChEMBL MMP pair extraction for SQLite databases.

This script provides the same functionality as run_chembl_pair_extraction.py
but works with SQLite databases instead of PostgreSQL.

Features:
- Direct SQLite database extraction
- Target selection (top N, specific targets, or sample)
- Fast atom mapping (485x speedup!)
- Checkpointing and resuming
- All 18 columns including atom mapping
- Progress tracking

Usage:
    # Extract from SQLite with top 5 targets
    python scripts/extraction/run_chembl_sqlite_extraction.py \\
        --db-path data/small_molecules/chembl_db/chembl/36/chembl_36.db \\
        --top-n-targets 5 \\
        --output-dir data/pairs

    # Extract only CSV files (skip MMP pair extraction)
    python scripts/extraction/run_chembl_sqlite_extraction.py \\
        --db-path data/small_molecules/chembl_db/chembl/36/chembl_36.db \\
        --top-n-targets 5 \\
        --output-dir data/chembl \\
        --csv-only

    # Use existing CSV files
    python scripts/extraction/run_chembl_sqlite_extraction.py \\
        --molecules-file data/chembl/molecules.csv \\
        --bioactivity-file data/chembl/bioactivity.csv \\
        --output-dir data/pairs

    # Extract specific targets
    python scripts/extraction/run_chembl_sqlite_extraction.py \\
        --db-path data/small_molecules/chembl_db/chembl/36/chembl_36.db \\
        --specific-targets CHEMBL203 CHEMBL217 \\
        --output-dir data/pairs

    # Sample extraction - 1000 molecules TOTAL (random across all targets)
    python scripts/extraction/run_chembl_sqlite_extraction.py \\
        --db-path data/small_molecules/chembl_db/chembl/36/chembl_36.db \\
        --top-n-targets 3 \\
        --sample-size 1000 \\
        --output-dir data/pairs

    # Sample extraction - 1000 molecules PER target (balanced, 3000 total)
    python scripts/extraction/run_chembl_sqlite_extraction.py \\
        --db-path data/small_molecules/chembl_db/chembl/36/chembl_36.db \\
        --top-n-targets 3 \\
        --sample-size 1000 \\
        --sample-per-target \\
        --output-dir data/pairs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import sqlite3
import time
import pandas as pd
from typing import Optional, List, Tuple
from src.data.small_molecule.mmp_long_format import LongFormatMMPExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SQLiteChEMBLExtractor:
    """Extract ChEMBL data from SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def get_top_targets(self, top_n: int = 10, min_molecules: int = 100) -> pd.DataFrame:
        """Get top N targets by molecule count."""
        query = """
        SELECT
            td.chembl_id as target_chembl_id,
            td.pref_name as target_name,
            COUNT(DISTINCT cs.molregno) as molecule_count
        FROM target_dictionary td
        JOIN assays ass ON td.tid = ass.tid
        JOIN activities act ON ass.assay_id = act.assay_id
        JOIN compound_structures cs ON act.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
            AND act.pchembl_value IS NOT NULL
            AND td.target_type = 'SINGLE PROTEIN'
        GROUP BY td.chembl_id, td.pref_name
        HAVING COUNT(DISTINCT cs.molregno) >= ?
        ORDER BY molecule_count DESC
        LIMIT ?
        """

        df = pd.read_sql_query(query, self.conn, params=(min_molecules, top_n))
        logger.info(f"Found {len(df)} targets with ≥{min_molecules} molecules")

        return df

    def extract_molecules_for_targets(
        self,
        target_ids: List[str],
        sample_size: Optional[int] = None,
        sample_per_target: bool = False
    ) -> pd.DataFrame:
        """
        Extract molecules for specific targets.

        Args:
            target_ids: List of target ChEMBL IDs
            sample_size: Number of molecules to sample
            sample_per_target: If True, sample N molecules per target (balanced).
                              If False, sample N total across all targets (may be unbalanced).

        Examples:
            # 1000 molecules PER target (5 targets = 5000 total)
            extract_molecules_for_targets(['CHEMBL203', ...], sample_size=1000, sample_per_target=True)

            # 1000 molecules TOTAL across all targets (random sampling)
            extract_molecules_for_targets(['CHEMBL203', ...], sample_size=1000, sample_per_target=False)
        """

        if sample_per_target and sample_size:
            # Balanced sampling: N molecules per target
            all_molecules = []

            for target_id in target_ids:
                query = f"""
                SELECT DISTINCT
                    md.chembl_id,
                    cs.canonical_smiles as smiles
                FROM compound_structures cs
                JOIN molecule_dictionary md ON cs.molregno = md.molregno
                JOIN activities act ON cs.molregno = act.molregno
                JOIN assays ass ON act.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                WHERE cs.canonical_smiles IS NOT NULL
                    AND act.pchembl_value IS NOT NULL
                    AND td.chembl_id = '{target_id}'
                ORDER BY RANDOM()
                LIMIT {sample_size}
                """

                target_df = pd.read_sql_query(query, self.conn)
                all_molecules.append(target_df)
                logger.info(f"  {target_id}: {len(target_df):,} molecules")

            df = pd.concat(all_molecules, ignore_index=True).drop_duplicates(subset=['chembl_id'])
            logger.info(f"Extracted {len(df):,} unique molecules ({sample_size} per target)")

        else:
            # Total sampling: N molecules across all targets (random)
            target_list = ','.join([f"'{tid}'" for tid in target_ids])

            query = f"""
            SELECT DISTINCT
                md.chembl_id,
                cs.canonical_smiles as smiles
            FROM compound_structures cs
            JOIN molecule_dictionary md ON cs.molregno = md.molregno
            JOIN activities act ON cs.molregno = act.molregno
            JOIN assays ass ON act.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            WHERE cs.canonical_smiles IS NOT NULL
                AND act.pchembl_value IS NOT NULL
                AND td.chembl_id IN ({target_list})
            """

            if sample_size:
                query += f" ORDER BY RANDOM() LIMIT {sample_size}"

            df = pd.read_sql_query(query, self.conn)
            logger.info(f"Extracted {len(df):,} unique molecules (random sample across {len(target_ids)} targets)")

        return df

    def extract_bioactivity_for_molecules(
        self,
        molecule_ids: List[str]
    ) -> pd.DataFrame:
        """Extract bioactivity for specific molecules."""

        # Split into chunks to avoid SQL parameter limits
        chunk_size = 500
        all_bioactivity = []

        for i in range(0, len(molecule_ids), chunk_size):
            chunk = molecule_ids[i:i + chunk_size]
            mol_list = ','.join([f"'{mid}'" for mid in chunk])

            query = f"""
            SELECT
                md.chembl_id,
                'pchembl_value' as property_name,
                act.pchembl_value as value,
                td.pref_name as target_name,
                td.chembl_id as target_chembl_id,
                act.doc_id,
                act.assay_id
            FROM activities act
            JOIN molecule_dictionary md ON act.molregno = md.molregno
            JOIN assays ass ON act.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN compound_structures cs ON act.molregno = cs.molregno
            WHERE cs.canonical_smiles IS NOT NULL
                AND act.pchembl_value IS NOT NULL
                AND md.chembl_id IN ({mol_list})
                AND td.target_type = 'SINGLE PROTEIN'
            """

            chunk_df = pd.read_sql_query(query, self.conn)
            all_bioactivity.append(chunk_df)

            if (i + chunk_size) % 5000 == 0:
                logger.info(f"  Processed {i + chunk_size:,}/{len(molecule_ids):,} molecules...")

        df = pd.concat(all_bioactivity, ignore_index=True)
        logger.info(f"Extracted {len(df):,} bioactivity measurements")

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Production ChEMBL MMP extraction from SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--db-path',
                            help='Path to ChEMBL SQLite database')
    input_group.add_argument('--molecules-file',
                            help='Pre-extracted molecules CSV file')

    # Additional inputs when using CSV files
    parser.add_argument('--bioactivity-file',
                       help='Pre-extracted bioactivity CSV file (required with --molecules-file)')

    # Target selection (for database extraction)
    parser.add_argument('--top-n-targets', type=int, default=10,
                       help='Extract top N targets by molecule count (default: 10)')
    parser.add_argument('--min-molecules-per-target', type=int, default=100,
                       help='Minimum molecules per target (default: 100)')
    parser.add_argument('--specific-targets', nargs='+',
                       help='Extract specific target IDs (e.g., CHEMBL203 CHEMBL217)')
    parser.add_argument('--sample-size', type=int,
                       help='Limit molecules extracted (behavior depends on --sample-per-target)')
    parser.add_argument('--sample-per-target', action='store_true',
                       help='If set, sample N molecules PER target. Otherwise, sample N total across all targets.')

    # MMP extraction parameters
    parser.add_argument('--max-cuts', type=int, default=1,
                       help='Maximum bond cuts (default: 1)')
    parser.add_argument('--max-mw-delta', type=float, default=250,
                       help='Maximum MW delta (default: 250)')
    parser.add_argument('--min-similarity', type=float, default=0.35,
                       help='Minimum Tanimoto similarity (default: 0.35)')
    parser.add_argument('--min-molecules-per-core', type=int, default=10,
                       help='Minimum molecules per core to keep (default: 10). Lower values = more pairs but higher memory usage.')

    # Output options
    parser.add_argument('--output-dir', default='data/pairs',
                       help='Output directory (default: data/pairs)')
    parser.add_argument('--output-name',
                       help='Output filename (default: chembl_pairs_{timestamp}.csv)')
    parser.add_argument('--checkpoint-dir',
                       help='Checkpoint directory for resuming (optional)')

    # Processing options
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip database extraction, use cached CSV files')
    parser.add_argument('--csv-only', action='store_true',
                       help='Extract only CSV files, skip MMP pair extraction')

    args = parser.parse_args()

    # Validate arguments
    if args.molecules_file and not args.bioactivity_file:
        parser.error("--bioactivity-file required when using --molecules-file")

    if args.csv_only and args.molecules_file:
        parser.error("--csv-only cannot be used with --molecules-file (CSV files already exist)")

    if args.csv_only and not args.db_path:
        parser.error("--csv-only requires --db-path")

    # Create output directories
    # Resolve relative paths relative to project root, not current working directory
    project_root = Path(__file__).parent.parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = project_root / checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        args.checkpoint_dir = str(checkpoint_dir)  # Update with absolute path

    # Generate output filename
    if args.output_name:
        output_file = output_dir / args.output_name
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"chembl_pairs_{timestamp}.csv"

    # Start pipeline
    logger.info("=" * 80)
    logger.info(" CHEMBL SQLite MMP EXTRACTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f" Output: {output_file}")
    logger.info(f" Max cuts: {args.max_cuts}")
    logger.info(f" Max MW delta: {args.max_mw_delta}")
    logger.info(f" Min similarity: {args.min_similarity}")
    logger.info("=" * 80)
    print()

    pipeline_start = time.time()

    try:
        # ===== STAGE 1: Data Extraction =====
        logger.info("=" * 80)
        logger.info(" STAGE 1: DATA EXTRACTION")
        logger.info("=" * 80)
        print()

        if args.molecules_file:
            # Use existing CSV files
            logger.info("Using existing CSV files:")
            logger.info(f"  Molecules: {args.molecules_file}")
            logger.info(f"  Bioactivity: {args.bioactivity_file}")

            molecules_df = pd.read_csv(args.molecules_file)
            bioactivity_df = pd.read_csv(args.bioactivity_file)

            logger.info(f"  Loaded {len(molecules_df):,} molecules")
            logger.info(f"  Loaded {len(bioactivity_df):,} bioactivity measurements")

        else:
            # Extract from SQLite database
            logger.info(f"Extracting from SQLite database: {args.db_path}")
            print()

            with SQLiteChEMBLExtractor(args.db_path) as extractor:
                # Get target IDs
                if args.specific_targets:
                    target_ids = args.specific_targets
                    logger.info(f"Using specific targets: {target_ids}")
                else:
                    # Get top N targets
                    targets_df = extractor.get_top_targets(
                        top_n=args.top_n_targets,
                        min_molecules=args.min_molecules_per_target
                    )

                    logger.info(f"\\nTop {len(targets_df)} targets:")
                    for idx, row in targets_df.iterrows():
                        logger.info(f"  {idx+1}. {row['target_name']} ({row['target_chembl_id']}): {row['molecule_count']:,} molecules")

                    target_ids = targets_df['target_chembl_id'].tolist()

                print()

                # Extract molecules
                logger.info("Extracting molecules...")
                if args.sample_per_target and args.sample_size:
                    logger.info(f"  Sampling {args.sample_size:,} molecules PER target")
                elif args.sample_size:
                    logger.info(f"  Sampling {args.sample_size:,} molecules TOTAL (random across all targets)")

                molecules_df = extractor.extract_molecules_for_targets(
                    target_ids=target_ids,
                    sample_size=args.sample_size,
                    sample_per_target=args.sample_per_target
                )

                # Extract bioactivity
                logger.info("\\nExtracting bioactivity...")
                bioactivity_df = extractor.extract_bioactivity_for_molecules(
                    molecule_ids=molecules_df['chembl_id'].tolist()
                )

            # Save to cache with descriptive filenames
            # Use top-level data directory (project root)
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / 'data' / 'small_molecules' / 'chembl'
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Build filename with hyperparameters
            filename_parts = []

            # Target selection
            if args.specific_targets:
                filename_parts.append(f"targets_{len(args.specific_targets)}")
            else:
                filename_parts.append(f"top_targets_{len(target_ids)}")

            # Sampling strategy
            if args.sample_size:
                if args.sample_per_target:
                    filename_parts.append(f"sample_{args.sample_size}_per_target")
                else:
                    filename_parts.append(f"sample_{args.sample_size}_total")
            else:
                filename_parts.append("all_molecules")

            filename_suffix = "_".join(filename_parts)

            molecules_file = cache_dir / f'molecules_{filename_suffix}.csv'
            bioactivity_file = cache_dir / f'bioactivity_{filename_suffix}.csv'

            molecules_df.to_csv(molecules_file, index=False)
            bioactivity_df.to_csv(bioactivity_file, index=False)

            logger.info(f"\\n✓ Cached to:")
            logger.info(f"  {molecules_file}")
            logger.info(f"  {bioactivity_file}")

        print()
        logger.info(f"✓ Stage 1 complete")
        print()

        # If CSV-only mode, skip MMP extraction
        if args.csv_only:
            logger.info("=" * 80)
            logger.info(" CSV-ONLY MODE: Skipping MMP pair extraction")
            logger.info("=" * 80)
            print()
            logger.info("CSV files saved to:")
            logger.info(f"  {molecules_file}")
            logger.info(f"  {bioactivity_file}")
            print()
            logger.info("To run MMP pair extraction later:")
            logger.info(f"  python scripts/extraction/build_pairs_long_format.py \\")
            logger.info(f"      --molecules-file {molecules_file} \\")
            logger.info(f"      --bioactivity-file {bioactivity_file} \\")
            logger.info(f"      --output data/pairs/chembl_pairs.csv \\")
            logger.info(f"      --max-cuts {args.max_cuts}")
            print()
            return 0

        # ===== STAGE 2: MMP Pair Extraction =====
        logger.info("=" * 80)
        logger.info(" STAGE 2: MMP PAIR EXTRACTION (with fast atom mapping)")
        logger.info("=" * 80)
        print()

        logger.info("Creating MMP extractor...")
        extractor = LongFormatMMPExtractor(max_cuts=args.max_cuts)

        logger.info("Extracting pairs...")
        logger.info(f"  This includes atom-level mapping (485x faster than MCS!)")
        print()

        extraction_start = time.time()

        pairs_df = extractor.extract_pairs_long_format(
            molecules_df=molecules_df,
            bioactivity_df=bioactivity_df,
            max_mw_delta=args.max_mw_delta,
            min_similarity=args.min_similarity,
            checkpoint_dir=args.checkpoint_dir,
            resume_from_checkpoint=True,
            min_molecules_per_core=args.min_molecules_per_core
        )

        extraction_time = time.time() - extraction_start

        if len(pairs_df) == 0:
            logger.error("\\n✗ No pairs extracted!")
            logger.error("  Try:")
            logger.error(f"    - Increasing --max-mw-delta (current: {args.max_mw_delta})")
            logger.error(f"    - Decreasing --min-similarity (current: {args.min_similarity})")
            logger.error(f"    - Using more molecules (current: {len(molecules_df):,})")
            return 1

        # Save output
        logger.info(f"\\nSaving pairs to: {output_file}")
        pairs_df.to_csv(output_file, index=False)

        logger.info(f"✓ Saved {len(pairs_df):,} pairs")
        print()

        # ===== STAGE 3: Validation =====
        logger.info("=" * 80)
        logger.info(" STAGE 3: OUTPUT VALIDATION")
        logger.info("=" * 80)
        print()

        # Check columns
        required_columns = [
            'mol_a', 'mol_b', 'edit_smiles', 'num_cuts',
            'property_name', 'value_a', 'value_b', 'delta',
            'target_name', 'target_chembl_id',
            'doc_id_a', 'doc_id_b', 'assay_id_a', 'assay_id_b',
            'removed_atoms_A', 'added_atoms_B', 'attach_atoms_A', 'mapped_pairs'
        ]

        missing_cols = [col for col in required_columns if col not in pairs_df.columns]

        if missing_cols:
            logger.error(f"✗ Missing columns: {missing_cols}")
            return 1

        logger.info(f"✓ All {len(required_columns)} required columns present")

        # Statistics
        n_unique_pairs = pairs_df[['mol_a', 'mol_b']].drop_duplicates().shape[0]
        n_unique_edits = pairs_df['edit_smiles'].nunique()
        n_unique_properties = pairs_df['property_name'].nunique()
        n_unique_targets = pairs_df['target_chembl_id'].nunique()

        logger.info("\\nDataset statistics:")
        logger.info(f"  Total rows: {len(pairs_df):,}")
        logger.info(f"  Unique molecule pairs: {n_unique_pairs:,}")
        logger.info(f"  Unique edits: {n_unique_edits:,}")
        logger.info(f"  Unique properties: {n_unique_properties}")
        logger.info(f"  Unique targets: {n_unique_targets}")

        # Atom mapping coverage
        not_empty = lambda col: (pairs_df[col].notna() & (pairs_df[col] != ""))
        coverage = {
            'removed_atoms_A': not_empty('removed_atoms_A').sum(),
            'added_atoms_B': not_empty('added_atoms_B').sum(),
            'attach_atoms_A': not_empty('attach_atoms_A').sum(),
            'mapped_pairs': not_empty('mapped_pairs').sum()
        }

        logger.info("\\nAtom mapping coverage:")
        for col, count in coverage.items():
            pct = 100 * count / len(pairs_df)
            logger.info(f"  {col}: {count:,}/{len(pairs_df):,} ({pct:.1f}%)")

        # Performance
        total_time = time.time() - pipeline_start
        time_per_pair = (extraction_time * 1000) / len(pairs_df) if len(pairs_df) > 0 else 0

        logger.info("\\nPerformance:")
        logger.info(f"  Total pipeline time: {total_time/60:.1f} min")
        logger.info(f"  Extraction time: {extraction_time/60:.1f} min")
        logger.info(f"  Time per pair: {time_per_pair:.1f}ms")
        logger.info(f"  Throughput: {len(pairs_df)/(extraction_time/60):.0f} pairs/min")

        print()
        logger.info("=" * 80)
        logger.info(f"✓ PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info(f" Output file: {output_file}")
        logger.info(f" Total pairs: {len(pairs_df):,}")
        logger.info(f" Total time: {total_time/60:.1f} min")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
ChEMBL downloader with PER-TARGET bioactivity columns.

This creates a wide-format dataset where each target+assay gets its own column.
Instead of aggregating bioactivity across targets, we preserve target-specific values.

Example output columns:
- Computed: alogp, mw, psa, hbd, hba, ...
- Bioactivity: egfr_ic50, ache_ki, thrombin_ic50, cyp3a4_ic50, ...

This enables target-specific structure-activity relationship learning!
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os

logger = logging.getLogger(__name__)


class ChEMBLPerTargetBioactivity:
    """
    Download ChEMBL molecules with per-target bioactivity columns.

    Instead of:
        mean_pchembl (averaged across all targets)

    You get:
        egfr_ic50, ache_ki, thrombin_ic50, etc. (one column per target)
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

    def get_top_targets(self,
                       top_n: int = 50,
                       activity_types: List[str] = ['IC50', 'Ki', 'EC50', 'Kd'],
                       min_molecules: int = 1000) -> pd.DataFrame:
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
            td.organism,
            act.standard_type as activity_type,
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

        GROUP BY td.chembl_id, td.pref_name, td.target_type, td.organism, act.standard_type
        HAVING COUNT(DISTINCT md.molregno) >= {min_molecules}

        ORDER BY n_molecules DESC
        LIMIT {top_n * 4};
        """

        df = self.chembl.query(query)
        logger.info(f"Found {len(df)} target+assay combinations")

        return df

    def download_with_per_target_bioactivity(
        self,
        n_molecules: int = 100000,
        max_mw: float = 800,
        min_mw: float = 100,
        top_targets: int = 50,
        activity_types: List[str] = ['IC50', 'Ki', 'EC50', 'Kd'],
        min_targets_per_molecule: int = 1
    ) -> pd.DataFrame:
        """
        Download molecules with per-target bioactivity columns.

        Args:
            n_molecules: Number of molecules to download
            max_mw: Maximum molecular weight
            min_mw: Minimum molecular weight
            top_targets: Number of top targets to include
            activity_types: Activity types to include
            min_targets_per_molecule: Minimum number of targets molecule must be tested against

        Returns:
            Wide-format DataFrame with per-target columns
        """
        logger.info("=" * 70)
        logger.info(" CHEMBL PER-TARGET BIOACTIVITY DOWNLOAD")
        logger.info("=" * 70)
        logger.info(f" Target molecules: {n_molecules:,}")
        logger.info(f" MW range: {min_mw}-{max_mw}")
        logger.info(f" Top targets: {top_targets}")
        logger.info(f" Activity types: {', '.join(activity_types)}")
        logger.info(f" Min targets/molecule: {min_targets_per_molecule}")
        logger.info("=" * 70)
        logger.info("")

        # Step 1: Get top targets
        logger.info("Step 1: Identifying top targets...")
        targets_df = self.get_top_targets(
            top_n=top_targets,
            activity_types=activity_types,
            min_molecules=1000
        )

        if len(targets_df) == 0:
            raise ValueError("No targets found!")

        # Create target+assay combinations (e.g., "CHEMBL203_IC50")
        target_assay_combos = []
        for _, row in targets_df.iterrows():
            combo_id = f"{row['target_chembl_id']}_{row['activity_type']}"
            target_assay_combos.append({
                'combo_id': combo_id,
                'target_chembl_id': row['target_chembl_id'],
                'target_name': row['target_name'],
                'activity_type': row['activity_type'],
                'n_molecules': row['n_molecules']
            })

        # Take top N unique combinations
        combo_df = pd.DataFrame(target_assay_combos).drop_duplicates('combo_id').head(top_targets)

        logger.info(f"Selected {len(combo_df)} target+assay combinations:")
        for _, row in combo_df.head(10).iterrows():
            logger.info(f"  • {row['combo_id']}: {row['target_name']} ({row['n_molecules']:,} molecules)")
        if len(combo_df) > 10:
            logger.info(f"  ... and {len(combo_df) - 10} more")
        logger.info("")

        # Step 2: Download base molecules with computed properties
        logger.info("Step 2: Downloading molecules with computed properties...")

        # Build list of target IDs for WHERE clause
        target_ids = combo_df['target_chembl_id'].unique()
        target_filter = "', '".join(target_ids)
        activity_filter = "', '".join(activity_types)

        # First, get molecules that have bioactivity for our targets
        query_base = f"""
        SELECT DISTINCT
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
            md.topical

        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN compound_properties cp ON md.molregno = cp.molregno

        WHERE
            cp.full_mwt >= {min_mw}
            AND cp.full_mwt <= {max_mw}
            AND cs.canonical_smiles IS NOT NULL
            AND cp.alogp IS NOT NULL

            -- Has bioactivity for our targets
            AND EXISTS (
                SELECT 1
                FROM activities act
                JOIN assays ass ON act.assay_id = ass.assay_id
                JOIN target_dictionary td ON ass.tid = td.tid
                WHERE act.molregno = md.molregno
                AND td.chembl_id IN ('{target_filter}')
                AND act.standard_type IN ('{activity_filter}')
                AND act.pchembl_value IS NOT NULL
                AND act.standard_relation = '='
            )

        LIMIT {n_molecules * 3};
        """

        df_base = self.chembl.query(query_base)
        logger.info(f"  Retrieved {len(df_base):,} candidate molecules")

        if len(df_base) == 0:
            raise ValueError("No molecules found with bioactivity for selected targets!")

        # Step 3: Get bioactivity data for all molecules
        logger.info("Step 3: Fetching per-target bioactivity...")

        molregnos = df_base['molregno'].tolist()
        molregno_chunks = [molregnos[i:i+5000] for i in range(0, len(molregnos), 5000)]

        all_bioactivity = []

        for i, chunk in enumerate(molregno_chunks):
            logger.info(f"  Processing chunk {i+1}/{len(molregno_chunks)} ({len(chunk)} molecules)...")

            molregno_filter = ','.join(map(str, chunk))

            query_bioactivity = f"""
            SELECT
                md.molregno,
                td.chembl_id as target_chembl_id,
                act.standard_type as activity_type,
                act.pchembl_value,
                act.standard_value,
                act.standard_units

            FROM activities act
            JOIN assays ass ON act.assay_id = ass.assay_id
            JOIN target_dictionary td ON ass.tid = td.tid
            JOIN molecule_dictionary md ON act.molregno = md.molregno

            WHERE
                md.molregno IN ({molregno_filter})
                AND td.chembl_id IN ('{target_filter}')
                AND act.standard_type IN ('{activity_filter}')
                AND act.pchembl_value IS NOT NULL
                AND act.standard_relation = '='
            """

            df_bio = self.chembl.query(query_bioactivity)
            all_bioactivity.append(df_bio)

        df_bioactivity = pd.concat(all_bioactivity, ignore_index=True)
        logger.info(f"  Retrieved {len(df_bioactivity):,} bioactivity measurements")

        # Step 4: Pivot bioactivity into wide format
        logger.info("Step 4: Pivoting bioactivity into per-target columns...")

        # Create combo_id in bioactivity data
        df_bioactivity['combo_id'] = df_bioactivity['target_chembl_id'] + '_' + df_bioactivity['activity_type']

        # Filter to only our selected combos
        df_bioactivity = df_bioactivity[df_bioactivity['combo_id'].isin(combo_df['combo_id'])]

        # For molecules with multiple measurements for same target+assay, take median
        df_bioactivity_agg = df_bioactivity.groupby(['molregno', 'combo_id']).agg({
            'pchembl_value': 'median',
            'standard_value': 'median',
            'standard_units': 'first'
        }).reset_index()

        # Pivot: one column per target+assay
        pivot_pchembl = df_bioactivity_agg.pivot(
            index='molregno',
            columns='combo_id',
            values='pchembl_value'
        ).reset_index()

        # Rename columns to be more readable (e.g., "CHEMBL203_IC50" -> "egfr_ic50")
        column_mapping = {'molregno': 'molregno'}
        for combo_id in combo_df['combo_id']:
            row = combo_df[combo_df['combo_id'] == combo_id].iloc[0]
            # Create readable name: first word of target + activity type
            target_short = row['target_name'].lower().split()[0].replace('-', '_')[:15]
            activity_lower = row['activity_type'].lower()
            readable_name = f"{target_short}_{activity_lower}"
            column_mapping[combo_id] = readable_name

        pivot_pchembl.rename(columns=column_mapping, inplace=True)

        # Step 5: Merge with base data
        logger.info("Step 5: Merging with base properties...")

        df_final = df_base.merge(pivot_pchembl, on='molregno', how='inner')

        # Filter: require minimum number of targets
        bioactivity_cols = [col for col in df_final.columns if col in column_mapping.values() and col != 'molregno']
        df_final['n_targets_tested'] = df_final[bioactivity_cols].notna().sum(axis=1)
        df_final = df_final[df_final['n_targets_tested'] >= min_targets_per_molecule]

        # Sort by number of targets tested (most tested first) and take top n_molecules
        df_final = df_final.sort_values('n_targets_tested', ascending=False).head(n_molecules)

        # Drop temporary columns
        df_final = df_final.drop(columns=['molregno', 'n_targets_tested'])

        logger.info(f"  Final dataset: {len(df_final):,} molecules")
        logger.info(f"  Total columns: {len(df_final.columns)}")
        logger.info(f"    - Base properties: {len(df_base.columns) - 1}")
        logger.info(f"    - Bioactivity columns: {len(bioactivity_cols)}")

        # Step 6: Save
        output_file = self.output_dir / f"chembl_per_target_{len(df_final)}.csv"
        df_final.to_csv(output_file, index=False)

        logger.info("")
        logger.info("=" * 70)
        logger.info(" SUCCESS!")
        logger.info("=" * 70)
        logger.info(f" Molecules: {len(df_final):,}")
        logger.info(f" Total properties: {len(df_final.columns)}")
        logger.info(f" Output: {output_file}")
        logger.info("=" * 70)
        logger.info("")

        # Show bioactivity coverage
        logger.info("Bioactivity coverage:")
        # Convert to list if needed
        bio_cols_list = list(bioactivity_cols) if not isinstance(bioactivity_cols, list) else bioactivity_cols
        for col in bio_cols_list[:10]:
            n_values = df_final[col].notna().sum()
            pct = 100 * n_values / len(df_final)
            logger.info(f"  • {col}: {int(n_values):,} molecules ({pct:.1f}%)")
        if len(bio_cols_list) > 10:
            logger.info(f"  ... and {len(bio_cols_list) - 10} more targets")

        logger.info("")

        # Save target mapping for reference
        target_mapping_file = self.output_dir / f"target_mapping_{len(df_final)}.csv"
        combo_df['readable_name'] = combo_df['combo_id'].map(column_mapping)
        combo_df.to_csv(target_mapping_file, index=False)
        logger.info(f" Target mapping saved to: {target_mapping_file}")
        logger.info("")

        return df_final


def main():
    """Main script for per-target bioactivity download."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download ChEMBL with per-target bioactivity columns"
    )

    parser.add_argument('--target', type=int, default=100000,
                       help='Number of molecules to download (default: 100000)')
    parser.add_argument('--max-mw', type=float, default=800,
                       help='Maximum molecular weight (default: 800)')
    parser.add_argument('--min-mw', type=float, default=100,
                       help='Minimum molecular weight (default: 100)')
    parser.add_argument('--top-targets', type=int, default=50,
                       help='Number of top targets to include (default: 50)')
    parser.add_argument('--activity-types', nargs='+', default=['IC50', 'Ki', 'EC50', 'Kd'],
                       help='Activity types to include (default: IC50 Ki EC50 Kd)')
    parser.add_argument('--min-targets', type=int, default=1,
                       help='Minimum targets molecule must be tested against (default: 1)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    # Create downloader
    downloader = ChEMBLPerTargetBioactivity()

    # Download data
    df = downloader.download_with_per_target_bioactivity(
        n_molecules=args.target,
        max_mw=args.max_mw,
        min_mw=args.min_mw,
        top_targets=args.top_targets,
        activity_types=args.activity_types,
        min_targets_per_molecule=args.min_targets
    )

    print()
    print("=" * 70)
    print(" NEXT STEPS")
    print("=" * 70)
    print()
    print("Use this data to build pairs dataset:")
    print(f"  python build_pairs_dataset.py \\")
    print(f"      --chembl-file data/chembl_bulk/chembl_per_target_{len(df)}.csv")
    print()
    print("This will extract pairs with deltas for ALL properties,")
    print("including per-target bioactivity!")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()

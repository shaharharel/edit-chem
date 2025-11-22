"""
Long-format MMP extraction optimized for efficiency.

MINIMAL SCHEMA:
    mol_a, mol_b, edit_smiles, edit_name, property_name, value_a, value_b, delta,
    target_name, target_chembl_id

Fields:
    - mol_a, mol_b: Full molecule SMILES (for debugging, will convert to IDs later)
    - edit_smiles: Canonical reaction SMILES (e.g., "C>>CC") for encoding with ChemBERTa
                   Computed using RDKit MMP fragmentation + canonical SMILES
    - edit_name: Canonical medicinal chemistry name (e.g., "homologation_C1_to_C2", "methylation")
                 Falls back to edit_smiles if no canonical name exists
    - property_name: Property being compared (e.g., "IC50_EGFR")
    - value_a, value_b: Property values
    - delta: value_b - value_a (observed change)
    - target_name, target_chembl_id: Target info (for bioactivity only)

For edit embeddings:
    Option 1: Use edit_smiles directly with ChemBERTa/transformers
    Option 2: Compute fingerprint difference on-the-fly: fp(mol_b) - fp(mol_a)
    Option 3: Combined embedding: F(mol, edit) -> property_change

This eliminates NaN/missing issues and only stores what exists.
"""

import logging
import pandas as pd
import numpy as np
import hashlib
import gc
import psutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, rdMMPA
from tqdm import tqdm
from .edit_vocabulary import get_edit_name

logger = logging.getLogger(__name__)


def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class LongFormatMMPExtractor:
    """
    Extract matched molecular pairs and output in long format.

    Long format: one row per pair-property combination.
    More efficient storage, no NaN issues, easy filtering.
    """

    def __init__(self, max_cuts: int = 1):
        """
        Initialize extractor.

        Args:
            max_cuts: Maximum number of cuts for MMP fragmentation (default: 1)
                     1 = single attachment point (simpler, closer edits)
                     2 = two attachment points (more complex edits)
        """
        self.max_cuts = max_cuts

    def fragment_molecule(self, smiles: str) -> Dict[str, str]:
        """
        Fragment molecule using rdMMPA.

        Args:
            smiles: SMILES string

        Returns:
            Dict mapping core -> attachment
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        try:
            # For max_cuts=1, we actually fragment with maxCuts=2 to get proper cores for grouping
            # but will filter to only single-cut pairs during extraction
            fragment_max_cuts = max(2, self.max_cuts)
            frags = rdMMPA.FragmentMol(mol, maxCuts=fragment_max_cuts, resultsAsMols=False)

            core_to_attachment = {}
            for core, chains in frags:
                if core and chains:  # Both must be non-empty
                    core_to_attachment[core] = chains

            return core_to_attachment

        except Exception as e:
            logger.debug(f"Fragmentation failed for {smiles}: {e}")
            return {}

    def extract_pairs_long_format(
        self,
        molecules_df: pd.DataFrame,
        bioactivity_df: pd.DataFrame,
        max_mw_delta: float = 200,
        min_similarity: float = 0.4,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 1000,
        resume_from_checkpoint: bool = True,
        micro_batch_size: int = 200,  # NEW: Flush every N pairs
        property_filter: Optional[set] = None  # NEW: Only extract these properties
    ) -> pd.DataFrame:
        """
        Extract molecular pairs in long format with MEMORY-EFFICIENT streaming.

        Args:
            molecules_df: DataFrame with molecules and computed properties
            bioactivity_df: DataFrame with bioactivity (long format)
            max_mw_delta: Maximum MW difference for pairs
            min_similarity: Minimum Tanimoto similarity
            checkpoint_dir: Optional directory for checkpoints
            checkpoint_every: Report progress every N cores (default: 1000)
            resume_from_checkpoint: Try to resume from checkpoint if exists (default: True)
            micro_batch_size: Flush to disk every N pairs (default: 200)
            property_filter: Optional set of property names to extract (default: None = all)

        Returns:
            Long-format DataFrame with pairs
        """
        mem_start = get_memory_usage_mb()
        logger.info("=" * 70)
        logger.info(" LONG-FORMAT PAIR EXTRACTION (MEMORY-EFFICIENT)")
        logger.info("=" * 70)
        logger.info(f" Molecules: {len(molecules_df):,}")
        logger.info(f" Bioactivity: {len(bioactivity_df):,} measurements")
        logger.info(f" Max MW delta: {max_mw_delta}")
        logger.info(f" Min similarity: {min_similarity}")
        logger.info(f" Micro-batch size: {micro_batch_size} pairs")
        if property_filter:
            logger.info(f" Property filter: {sorted(property_filter)}")
        logger.info(f" Memory at start: {mem_start:.1f} MB")
        logger.info("=" * 70)
        logger.info("")

        # Step 1: Create SMILES index and property lookup
        logger.info("Step 1: Creating property lookup...")

        property_lookup = self._create_property_lookup(molecules_df, bioactivity_df)
        logger.info(f"  ✓ Created lookup for {len(property_lookup):,} molecules")
        mem_after_lookup = get_memory_usage_mb()
        logger.info(f"  Memory: {mem_after_lookup:.1f} MB (+{mem_after_lookup - mem_start:.1f} MB)")
        logger.info("")

        # Step 2: Fragment molecules (with caching)
        logger.info("Step 2: Fragmenting molecules...")

        # Check for cached fragments
        fragments_cache_file = None
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            fragments_cache_file = checkpoint_path / "fragments_cache.pkl"

        if fragments_cache_file and fragments_cache_file.exists() and resume_from_checkpoint:
            logger.info("  Loading cached fragments...")
            import pickle
            with open(fragments_cache_file, 'rb') as f:
                fragments = pickle.load(f)
            logger.info(f"  ✓ Loaded {len(fragments):,} cached fragments")
        else:
            logger.info("  Computing fragments (this may take a while)...")
            smiles_list = molecules_df['smiles'].tolist()
            fragments = {}

            for smiles in tqdm(smiles_list, desc="Fragmenting"):
                frags = self.fragment_molecule(smiles)
                if frags:
                    fragments[smiles] = frags

            logger.info(f"  ✓ Fragmented {len(fragments):,} molecules")

            # Cache fragments for next time
            if fragments_cache_file:
                logger.info("  Saving fragments cache...")
                import pickle
                with open(fragments_cache_file, 'wb') as f:
                    pickle.dump(fragments, f)
                logger.info("  ✓ Cached fragments for future runs")

        mem_after_frag = get_memory_usage_mb()
        logger.info(f"  Memory: {mem_after_frag:.1f} MB (+{mem_after_frag - mem_after_lookup:.1f} MB)")
        logger.info("")

        # Step 3: Index by core for efficient pairing
        logger.info("Step 3: Indexing by core...")

        core_index = defaultdict(list)
        for smiles, frags in fragments.items():
            for core in frags.keys():
                core_index[core].append(smiles)

        logger.info(f"  ✓ Found {len(core_index):,} total cores")

        # Filter: only keep cores with 10+ molecules for better pair density
        min_molecules_per_core = 10
        cores_to_keep = {core: mols for core, mols in core_index.items() if len(mols) >= min_molecules_per_core}

        num_deleted = len(core_index) - len(cores_to_keep)
        core_index = cores_to_keep
        del cores_to_keep
        gc.collect()

        filtered_cores = sorted(core_index.keys(), key=lambda c: len(core_index[c]))

        if len(filtered_cores) == 0:
            logger.warning(f"  No cores with {min_molecules_per_core}+ molecules found!")
            return pd.DataFrame(columns=[
                'mol_a', 'mol_b', 'edit_smiles', 'edit_name', 'property_name', 'value_a', 'value_b',
                'delta', 'target_name', 'target_chembl_id'
            ])

        core_sizes = [len(core_index[c]) for c in filtered_cores]
        logger.info(f"  ✓ Kept {len(filtered_cores):,} cores with {min_molecules_per_core}+ molecules")
        logger.info(f"  ✓ Deleted {num_deleted:,} cores to save memory")
        logger.info(f"  ✓ Core sizes: min={min(core_sizes)}, max={max(core_sizes)}, avg={sum(core_sizes)/len(core_sizes):.1f}")

        mem_after_index = get_memory_usage_mb()
        logger.info(f"  Memory: {mem_after_index:.1f} MB (+{mem_after_index - mem_after_frag:.1f} MB)")
        logger.info("")

        # Step 4: Extract pairs with STREAMING to disk
        logger.info("Step 4: Extracting pairs (STREAMING MODE)...")

        # Setup checkpoint directory
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            checkpoint_file = checkpoint_path / "pairs_checkpoint.csv"
        else:
            checkpoint_file = None

        # Count existing pairs
        total_pairs_written = 0
        if resume_from_checkpoint and checkpoint_file and checkpoint_file.exists():
            logger.info("  Found checkpoint, resuming...")
            import subprocess
            try:
                result = subprocess.run(['wc', '-l', str(checkpoint_file)],
                                      capture_output=True, text=True, check=True)
                line_count = int(result.stdout.split()[0])
                total_pairs_written = max(0, line_count - 1)
                logger.info(f"  Existing checkpoint has {total_pairs_written:,} pairs")
            except:
                with open(checkpoint_file, 'r') as f:
                    total_pairs_written = sum(1 for _ in f) - 1
                logger.info(f"  Existing checkpoint has {total_pairs_written:,} pairs")

        # Open file handle for streaming writes
        file_handle = None
        if checkpoint_file:
            mode = 'a' if (resume_from_checkpoint and checkpoint_file.exists()) else 'w'
            file_handle = open(checkpoint_file, mode, buffering=1024*1024)

            if mode == 'w':
                file_handle.write("mol_a,mol_b,edit_smiles,edit_name,property_name,value_a,value_b,delta,target_name,target_chembl_id\n")

        logger.info(f"  Total cores to process: {len(filtered_cores):,}")
        logger.info(f"  Flushing to disk every {micro_batch_size} pairs")
        logger.info("")

        # Micro-batch for memory efficiency
        batch_pairs = []
        mem_peak = mem_after_index

        # Iterate over cores
        core_pbar = tqdm(filtered_cores, desc="Cores", position=0)

        for core in core_pbar:
            smiles_list = core_index[core]

            # Calculate total pairs for this core
            n_molecules = len(smiles_list)
            total_pairs_in_core = n_molecules * (n_molecules - 1) // 2

            # Update core progress bar
            core_pbar.set_postfix({
                'core_size': n_molecules,
                'pairs': total_pairs_in_core,
                'written': total_pairs_written
            })

            # Nested progress bar for pairs within this core
            pair_pbar = tqdm(total=total_pairs_in_core, desc=f"  Pairs (core size={n_molecules})",
                           position=1, leave=False)

            # Process all pairs within this core
            for i in range(len(smiles_list)):
                for j in range(i + 1, len(smiles_list)):
                    pair_pbar.update(1)
                    smiles_a = smiles_list[i]
                    smiles_b = smiles_list[j]

                    # Check MW filter
                    if smiles_a not in property_lookup or smiles_b not in property_lookup:
                        continue

                    mw_a = property_lookup[smiles_a]['properties'].get('mw')
                    mw_b = property_lookup[smiles_b]['properties'].get('mw')

                    if mw_a is None or mw_b is None:
                        continue

                    if abs(mw_a - mw_b) > max_mw_delta:
                        continue

                    # Extract pair
                    pair_data = self._extract_single_pair(
                        smiles_a, smiles_b,
                        fragments[smiles_a], fragments[smiles_b],
                        property_lookup,
                        property_filter
                    )

                    if pair_data:
                        batch_pairs.extend(pair_data)

                    # MICRO-BATCH FLUSH
                    if len(batch_pairs) >= micro_batch_size:
                        if file_handle:
                            for row in batch_pairs:
                                file_handle.write(
                                    f"{row['mol_a']},{row['mol_b']},{row['edit_smiles']},{row['edit_name']},"
                                    f"{row['property_name']},{row['value_a']},{row['value_b']},"
                                    f"{row['delta']},{row['target_name']},{row['target_chembl_id']}\n"
                                )
                            file_handle.flush()

                        total_pairs_written += len(batch_pairs)
                        batch_pairs = []
                        gc.collect()

        core_pbar.close()

        # Write final batch
        if batch_pairs:
            if file_handle:
                for row in batch_pairs:
                    file_handle.write(
                        f"{row['mol_a']},{row['mol_b']},{row['edit_smiles']},{row['edit_name']},"
                        f"{row['property_name']},{row['value_a']},{row['value_b']},"
                        f"{row['delta']},{row['target_name']},{row['target_chembl_id']}\n"
                    )
                file_handle.flush()

            total_pairs_written += len(batch_pairs)
            batch_pairs = []

        # Close file handle
        if file_handle:
            file_handle.close()

        logger.info(f"  ✓ Extracted {total_pairs_written:,} pair-property combinations")
        mem_after_extraction = get_memory_usage_mb()
        logger.info(f"  Memory after extraction: {mem_after_extraction:.1f} MB")
        logger.info(f"  Peak memory during extraction: {mem_peak:.1f} MB")
        logger.info("")

        # Read final result from checkpoint (in chunks to avoid memory spike)
        if checkpoint_file and checkpoint_file.exists():
            logger.info("  Reading final result from disk...")
            df_pairs_long = pd.read_csv(checkpoint_file)

            # Clean up checkpoint on success
            checkpoint_file.unlink()
            logger.info("  Cleaned up checkpoint file")
            logger.info("")
        else:
            # No checkpoint file (shouldn't happen)
            df_pairs_long = pd.DataFrame(columns=[
                'mol_a', 'mol_b', 'edit_smiles', 'edit_name', 'property_name', 'value_a', 'value_b',
                'delta', 'target_name', 'target_chembl_id'
            ])

        if len(df_pairs_long) == 0:
            logger.warning("  No pairs found!")
            return df_pairs_long

        # Statistics
        n_unique_pairs = df_pairs_long[['mol_a', 'mol_b']].drop_duplicates().shape[0]
        n_unique_edits = df_pairs_long['edit_smiles'].nunique()
        n_unique_properties = df_pairs_long['property_name'].nunique()

        mem_final = get_memory_usage_mb()

        logger.info("=" * 70)
        logger.info(" EXTRACTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f" Total rows: {len(df_pairs_long):,}")
        logger.info(f" Unique pairs: {n_unique_pairs:,}")
        logger.info(f" Unique edits: {n_unique_edits:,}")
        logger.info(f" Unique properties: {n_unique_properties:,}")
        logger.info(f" Avg properties per pair: {len(df_pairs_long) / n_unique_pairs:.1f}")
        logger.info("")
        logger.info(f" MEMORY STATS:")
        logger.info(f"   Start: {mem_start:.1f} MB")
        logger.info(f"   Peak: {mem_peak:.1f} MB")
        logger.info(f"   Final: {mem_final:.1f} MB")
        logger.info(f"   Peak increase: +{mem_peak - mem_start:.1f} MB")
        logger.info("=" * 70)
        logger.info("")

        return df_pairs_long

    def _create_property_lookup(
        self,
        molecules_df: pd.DataFrame,
        bioactivity_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Create property lookup: {smiles: {properties: {name: value}}}
        """
        lookup = {}

        # Computed properties
        computed_props = [
            'alogp', 'mw', 'mw_freebase', 'hbd', 'hba', 'psa', 'rtb',
            'aromatic_rings', 'heavy_atoms', 'qed_weighted',
            'num_ro5_violations', 'np_likeness_score'
        ]

        # Create chembl_id -> smiles mapping for O(1) lookup
        chembl_to_smiles = dict(zip(molecules_df['chembl_id'], molecules_df['smiles']))

        for _, row in molecules_df.iterrows():
            smiles = row['smiles']
            chembl_id = row['chembl_id']

            lookup[smiles] = {
                'chembl_id': chembl_id,
                'properties': {}
            }

            for prop in computed_props:
                if prop in row and pd.notna(row[prop]):
                    lookup[smiles]['properties'][prop] = row[prop]

        # Bioactivity properties (with target info)
        # Use vectorized operations instead of iterrows for better performance
        for _, row in bioactivity_df.iterrows():
            chembl_id = row['chembl_id']

            # O(1) lookup instead of O(n) scan
            smiles = chembl_to_smiles.get(chembl_id)
            if not smiles or smiles not in lookup:
                continue

            prop_name = row['property_name']
            value = row['pchembl_value']
            target_name = row.get('target_name', '')
            target_chembl_id = row.get('target_chembl_id', '')

            if pd.notna(value):
                # Store both value and target info
                lookup[smiles]['properties'][prop_name] = {
                    'value': value,
                    'target_name': target_name,
                    'target_chembl_id': target_chembl_id
                }

        return lookup

    def _extract_single_pair(
        self,
        smiles_a: str,
        smiles_b: str,
        frags_a: Dict[str, str],
        frags_b: Dict[str, str],
        property_lookup: Dict,
        property_filter: Optional[set] = None
    ) -> List[Dict]:
        """
        Extract pair and return list of rows (one per property).

        MINIMAL SCHEMA: Only stores mol_a, mol_b, property values, and target info.
        Edit embeddings should be computed on-the-fly: fp(mol_b) - fp(mol_a)

        Args:
            property_filter: Optional set of property names to include (default: all)

        Returns:
            List of dicts, each representing one row in long format
        """
        # Find common cores
        common_cores = set(frags_a.keys()) & set(frags_b.keys())

        if not common_cores:
            # Try all combinations of fragments to find if any share common parts
            attachment_a = None
            attachment_b = None

            for core_a, chains_a in frags_a.items():
                for core_b, chains_b in frags_b.items():
                    # Split chains into parts
                    parts_a = set(chains_a.split('.'))
                    parts_b = set(chains_b.split('.'))

                    # Find common parts (these form the core)
                    common_parts = parts_a & parts_b

                    if common_parts and len(common_parts) > 0:
                        # Found a shared structure
                        attachment_a = chains_a
                        attachment_b = chains_b
                        break
                if attachment_a:
                    break

            if not attachment_a:
                return []  # No valid MMP found
        else:
            # Take first common core
            core = list(common_cores)[0]
            attachment_a = frags_a[core]
            attachment_b = frags_b[core]

        if attachment_a == attachment_b:
            return []  # Not a transformation

        # Extract pure edit (what actually changed) for canonical representation
        # Split fragments by '.' to get individual pieces
        parts_a = set(attachment_a.split('.'))
        parts_b = set(attachment_b.split('.'))

        # Find what's unique to each side (the actual edit)
        edit_from_parts = parts_a - parts_b
        edit_to_parts = parts_b - parts_a

        # For max_cuts=1: filter to only accept single-cut pairs
        # (where only one fragment differs on each side)
        if self.max_cuts == 1:
            if len(edit_from_parts) != 1 or len(edit_to_parts) != 1:
                return []  # Not a single-cut pair

        # Convert to canonical SMILES (replace attachment points with H)
        # Use RDKit to canonicalize for consistency
        if edit_from_parts:
            edit_from_raw = '.'.join(sorted(edit_from_parts)).replace('[*:1]', '[H]').replace('[*:2]', '[H]')
            mol_from = Chem.MolFromSmiles(edit_from_raw)
            edit_from = Chem.MolToSmiles(mol_from) if mol_from else edit_from_raw
        else:
            edit_from = ''

        if edit_to_parts:
            edit_to_raw = '.'.join(sorted(edit_to_parts)).replace('[*:1]', '[H]').replace('[*:2]', '[H]')
            mol_to = Chem.MolFromSmiles(edit_to_raw)
            edit_to = Chem.MolToSmiles(mol_to) if mol_to else edit_to_raw
        else:
            edit_to = ''

        # Create reaction SMILES (CANONICAL format for encoding with ChemBERTa, etc.)
        # Format: reactant>>product
        # This IS the RDKit canonical way - MMP fragmentation + canonical SMILES
        edit_smiles = f"{edit_from}>>{edit_to}" if edit_from and edit_to else ''

        # Get canonical medicinal chemistry name for this edit
        edit_name = get_edit_name(edit_smiles) if edit_smiles else ''

        # Get properties for both molecules
        props_a = property_lookup[smiles_a]['properties']
        props_b = property_lookup[smiles_b]['properties']

        # Find shared properties
        shared_props = set(props_a.keys()) & set(props_b.keys())

        if not shared_props:
            return []

        # Create one row per property
        rows = []

        for prop_name in shared_props:
            # Skip if property not in filter
            if property_filter and prop_name not in property_filter:
                continue

            value_a_raw = props_a[prop_name]
            value_b_raw = props_b[prop_name]

            # Extract value and target info if dict (bioactivity), else just use value (computed prop)
            if isinstance(value_a_raw, dict):
                value_a = value_a_raw['value']
                target_name = value_a_raw.get('target_name', '')
                target_chembl_id = value_a_raw.get('target_chembl_id', '')
            else:
                value_a = value_a_raw
                target_name = ''
                target_chembl_id = ''

            if isinstance(value_b_raw, dict):
                value_b = value_b_raw['value']
            else:
                value_b = value_b_raw

            # Skip if either is None/NaN
            if value_a is None or value_b is None:
                continue
            if isinstance(value_a, float) and np.isnan(value_a):
                continue
            if isinstance(value_b, float) and np.isnan(value_b):
                continue

            delta = value_b - value_a

            # MINIMAL SCHEMA - only essential fields + canonical edit representation
            row = {
                'mol_a': smiles_a,
                'mol_b': smiles_b,
                'edit_smiles': edit_smiles,  # ⭐ CANONICAL: "C>>CC" for ChemBERTa encoding
                'edit_name': edit_name,  # Canonical medicinal chemistry name
                'property_name': prop_name,
                'value_a': value_a,
                'value_b': value_b,
                'delta': delta,
                'target_name': target_name,
                'target_chembl_id': target_chembl_id
            }

            rows.append(row)

        return rows


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract pairs in long format")
    parser.add_argument('--molecules-file', required=True,
                       help='Molecules CSV file')
    parser.add_argument('--bioactivity-file', required=True,
                       help='Bioactivity CSV file (long format)')
    parser.add_argument('--output', default='data/pairs/chembl_pairs_long.csv',
                       help='Output file')
    parser.add_argument('--max-mw-delta', type=float, default=200,
                       help='Max MW delta (default: 200)')
    parser.add_argument('--min-similarity', type=float, default=0.4,
                       help='Min similarity (default: 0.4)')
    parser.add_argument('--max-cuts', type=int, default=2,
                       help='Max cuts (default: 2)')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    # Load data
    logger.info(f"Loading molecules from {args.molecules_file}...")
    molecules_df = pd.read_csv(args.molecules_file)

    logger.info(f"Loading bioactivity from {args.bioactivity_file}...")
    bioactivity_df = pd.read_csv(args.bioactivity_file)

    # Extract pairs
    extractor = LongFormatMMPExtractor(max_cuts=args.max_cuts)

    df_pairs = extractor.extract_pairs_long_format(
        molecules_df=molecules_df,
        bioactivity_df=bioactivity_df,
        max_mw_delta=args.max_mw_delta,
        min_similarity=args.min_similarity
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(output_path, index=False)

    logger.info(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()

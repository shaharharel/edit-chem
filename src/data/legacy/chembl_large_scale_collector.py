"""
Large-scale ChEMBL data collection with checkpointing and rate limiting.

Designed to safely collect 50,000+ molecules overnight while respecting API limits.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Set
from tqdm import tqdm
import time
import json
from pathlib import Path
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)


class LargeScaleChEMBLCollector:
    """
    Robust collector for large-scale ChEMBL data downloads.

    Features:
    - Checkpointing (resume from interruptions)
    - Rate limiting (respects ChEMBL API limits)
    - Multiple property collection
    - Progress tracking
    - Error handling with retries
    """

    # ChEMBL API rate limits (conservative estimates)
    # https://www.ebi.ac.uk/chembl/faq#faq40
    REQUESTS_PER_SECOND = 1  # Conservative: 1 request per second
    BATCH_SIZE = 1000  # Save checkpoint every 1000 molecules
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    def __init__(self, output_dir: str = "data/chembl_large_scale"):
        """
        Initialize large-scale collector.

        Args:
            output_dir: Directory for data and checkpoints
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "checkpoint.pkl"
        self.data_file = self.output_dir / "compounds.csv"
        self.log_file = self.output_dir / "collection_log.json"

        # Initialize ChEMBL client
        try:
            from chembl_webresource_client.new_client import new_client
            self.molecule = new_client.molecule
            self.activity = new_client.activity
            logger.info("ChEMBL client initialized")
        except ImportError:
            logger.error("chembl_webresource_client not installed")
            raise

        # Collection state
        self.collected_ids: Set[str] = set()
        self.compounds: List[Dict] = []
        self.start_time = None
        self.total_requests = 0
        self.failed_requests = 0

        logger.info(f"Initialized large-scale collector. Output: {output_dir}")

    def collect_compounds(self,
                         target_count: int = 50000,
                         max_mw: float = 600,
                         min_mw: float = 100,
                         has_alogp: bool = True,
                         resume: bool = True) -> pd.DataFrame:
        """
        Collect large dataset of compounds with multiple properties.

        Args:
            target_count: Target number of compounds (default 50,000)
            max_mw: Maximum molecular weight
            min_mw: Minimum molecular weight
            has_alogp: Require ALogP values
            resume: Resume from checkpoint if available

        Returns:
            DataFrame with collected compounds
        """
        self.start_time = datetime.now()

        # Try to resume from checkpoint
        if resume and self.checkpoint_file.exists():
            logger.info("Found checkpoint. Resuming collection...")
            self._load_checkpoint()

        logger.info(f"Starting collection: target={target_count}, current={len(self.compounds)}")
        logger.info(f"Filters: MW {min_mw}-{max_mw}, has_alogp={has_alogp}")
        logger.info(f"Estimated time: {self._estimate_time(target_count - len(self.compounds))}")

        try:
            # Strategy: Query molecules in pages
            offset = len(self.collected_ids)
            page_size = 100  # Fetch IDs in batches

            pbar = tqdm(total=target_count, initial=len(self.compounds),
                       desc="Collecting molecules")

            while len(self.compounds) < target_count:
                # Fetch a batch of molecule IDs
                try:
                    molecule_ids = self._fetch_molecule_ids_batch(
                        offset=offset,
                        limit=page_size,
                        max_mw=max_mw,
                        min_mw=min_mw
                    )

                    if not molecule_ids:
                        logger.warning("No more molecules found. Breaking.")
                        break

                    # Fetch details for each molecule
                    for mol_id in molecule_ids:
                        if mol_id in self.collected_ids:
                            continue

                        # Rate limiting
                        self._rate_limit()

                        # Fetch molecule data with retries
                        mol_data = self._fetch_with_retry(mol_id)

                        if mol_data:
                            self.compounds.append(mol_data)
                            self.collected_ids.add(mol_id)
                            pbar.update(1)

                            # Checkpoint every BATCH_SIZE molecules
                            if len(self.compounds) % self.BATCH_SIZE == 0:
                                self._save_checkpoint()
                                self._log_progress()

                        # Check if we've reached target
                        if len(self.compounds) >= target_count:
                            break

                    offset += page_size

                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    self._save_checkpoint()
                    time.sleep(10)  # Wait before retrying
                    continue

            pbar.close()

            # Final save
            df = pd.DataFrame(self.compounds)
            df.to_csv(self.data_file, index=False)
            self._save_checkpoint()
            self._log_progress(final=True)

            logger.info(f"Collection complete! Collected {len(df)} compounds")
            logger.info(f"Data saved to: {self.data_file}")

            return df

        except KeyboardInterrupt:
            logger.warning("Collection interrupted by user")
            self._save_checkpoint()
            logger.info("Checkpoint saved. You can resume later.")
            raise

    def _fetch_molecule_ids_batch(self,
                                  offset: int,
                                  limit: int,
                                  max_mw: float,
                                  min_mw: float) -> List[str]:
        """
        Fetch batch of molecule IDs with filters.

        Args:
            offset: Starting offset
            limit: Number of IDs to fetch
            max_mw: Maximum molecular weight
            min_mw: Minimum molecular weight

        Returns:
            List of ChEMBL IDs
        """
        try:
            # Query molecules with property filters
            molecules = self.molecule.filter(
                molecule_properties__full_mwt__gte=min_mw,
                molecule_properties__full_mwt__lte=max_mw,
                molecule_structures__isnull=False
            ).only(['molecule_chembl_id'])[offset:offset+limit]

            self.total_requests += 1

            ids = [m['molecule_chembl_id'] for m in molecules if m.get('molecule_chembl_id')]
            logger.debug(f"Fetched {len(ids)} molecule IDs (offset={offset})")

            return ids

        except Exception as e:
            logger.error(f"Error fetching molecule IDs: {e}")
            self.failed_requests += 1
            return []

    def _fetch_with_retry(self, chembl_id: str, retries: int = MAX_RETRIES) -> Optional[Dict]:
        """
        Fetch molecule data with retry logic.

        Args:
            chembl_id: ChEMBL molecule ID
            retries: Number of retries

        Returns:
            Dictionary with molecule data or None
        """
        for attempt in range(retries):
            try:
                mol_data = self.molecule.get(chembl_id)
                self.total_requests += 1

                if not mol_data:
                    return None

                # Extract all relevant data
                return self._extract_molecule_data(mol_data)

            except Exception as e:
                logger.debug(f"Error fetching {chembl_id} (attempt {attempt+1}): {e}")
                self.failed_requests += 1

                if attempt < retries - 1:
                    time.sleep(self.RETRY_DELAY)
                else:
                    logger.warning(f"Failed to fetch {chembl_id} after {retries} attempts")
                    return None

    def _extract_molecule_data(self, mol_data: Dict) -> Optional[Dict]:
        """
        Extract all properties from molecule data.

        Args:
            mol_data: Raw ChEMBL molecule data

        Returns:
            Dictionary with standardized properties
        """
        # Get structure
        if 'molecule_structures' not in mol_data or not mol_data['molecule_structures']:
            return None

        structures = mol_data['molecule_structures']
        smiles = structures.get('canonical_smiles')

        if not smiles:
            return None

        # Get properties
        if 'molecule_properties' not in mol_data:
            return None

        props = mol_data['molecule_properties']

        # Extract comprehensive property set
        data = {
            'chembl_id': mol_data.get('molecule_chembl_id'),
            'smiles': smiles,
            'inchi_key': structures.get('standard_inchi_key'),

            # Chemical properties (calculated)
            'alogp': props.get('alogp'),  # Lipophilicity
            'mw': props.get('full_mwt'),  # Molecular weight
            'hbd': props.get('hbd'),  # H-bond donors
            'hba': props.get('hba'),  # H-bond acceptors
            'psa': props.get('psa'),  # Polar surface area
            'rtb': props.get('rtb'),  # Rotatable bonds

            # Structural features
            'aromatic_rings': props.get('aromatic_rings'),
            'heavy_atoms': props.get('heavy_atoms'),
            'num_ro5_violations': props.get('num_ro5_violations'),  # Lipinski violations
            'num_alerts': props.get('num_alerts'),  # Structural alerts

            # Complexity metrics
            'molecular_species': props.get('molecular_species'),
            'cx_logp': props.get('cx_logp'),  # Alternative LogP
            'cx_logd': props.get('cx_logd'),  # LogD at pH 7.4

            # Drug-likeness
            'qed_weighted': props.get('qed_weighted'),  # Quantitative Estimate of Drug-likeness
            'ro3_pass': props.get('ro3_pass'),  # Rule of 3 (fragments)

            # Solubility indicators
            'aromatic_rings': props.get('aromatic_rings'),
            'mw_freebase': props.get('mw_freebase'),
            'mw_monoisotopic': props.get('mw_monoisotopic'),

            # Metadata
            'molecule_type': mol_data.get('molecule_type'),
            'max_phase': mol_data.get('max_phase'),  # Clinical development phase
            'oral': mol_data.get('oral'),  # Oral drug
            'parenteral': mol_data.get('parenteral'),
            'topical': mol_data.get('topical'),
        }

        return data

    def _rate_limit(self):
        """
        Implement rate limiting to respect API limits.
        """
        time.sleep(1.0 / self.REQUESTS_PER_SECOND)

    def _save_checkpoint(self):
        """
        Save collection progress to checkpoint file.
        """
        checkpoint = {
            'collected_ids': list(self.collected_ids),
            'compounds': self.compounds,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'timestamp': datetime.now().isoformat()
        }

        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

        logger.debug(f"Checkpoint saved: {len(self.compounds)} compounds")

    def _load_checkpoint(self):
        """
        Load collection progress from checkpoint file.
        """
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)

            self.collected_ids = set(checkpoint['collected_ids'])
            self.compounds = checkpoint['compounds']
            self.total_requests = checkpoint.get('total_requests', 0)
            self.failed_requests = checkpoint.get('failed_requests', 0)

            logger.info(f"Checkpoint loaded: {len(self.compounds)} compounds, "
                       f"{self.total_requests} requests, {self.failed_requests} failures")
            logger.info(f"Last checkpoint: {checkpoint.get('timestamp')}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh collection")

    def _log_progress(self, final: bool = False):
        """
        Log collection progress to JSON file.

        Args:
            final: Whether this is the final log entry
        """
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'n_compounds': len(self.compounds),
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': 1 - (self.failed_requests / max(self.total_requests, 1)),
            'elapsed_seconds': elapsed,
            'compounds_per_second': len(self.compounds) / max(elapsed, 1),
            'final': final
        }

        # Append to log file
        logs = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                logs = json.load(f)

        logs.append(log_entry)

        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)

        logger.info(f"Progress: {len(self.compounds)} compounds, "
                   f"{log_entry['compounds_per_second']:.2f} compounds/sec")

    def _estimate_time(self, remaining: int) -> str:
        """
        Estimate remaining collection time.

        Args:
            remaining: Number of remaining compounds to collect

        Returns:
            Human-readable time estimate
        """
        # Conservative estimate: 1 request per second, 2 requests per compound
        seconds = remaining * 2 / self.REQUESTS_PER_SECOND

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)

        return f"{hours}h {minutes}m (conservative estimate)"

    def get_statistics(self) -> pd.DataFrame:
        """
        Get collection statistics summary.

        Returns:
            DataFrame with property statistics
        """
        if not self.compounds:
            logger.warning("No compounds collected yet")
            return pd.DataFrame()

        df = pd.DataFrame(self.compounds)

        # Compute statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        stats = df[numeric_cols].describe()

        logger.info("Collection Statistics:")
        print(stats)

        return stats


def main():
    """
    Main collection script for overnight run.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Collect large-scale ChEMBL dataset")
    parser.add_argument('--target', type=int, default=50000,
                       help='Target number of compounds (default: 50000)')
    parser.add_argument('--output', type=str, default='data/chembl_large_scale',
                       help='Output directory (default: data/chembl_large_scale)')
    parser.add_argument('--max-mw', type=float, default=600,
                       help='Maximum molecular weight (default: 600)')
    parser.add_argument('--min-mw', type=float, default=100,
                       help='Minimum molecular weight (default: 100)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{args.output}/collection.log'),
            logging.StreamHandler()
        ]
    )

    # Create collector
    collector = LargeScaleChEMBLCollector(output_dir=args.output)

    # Run collection
    logger.info("=" * 60)
    logger.info("LARGE-SCALE CHEMBL COLLECTION")
    logger.info("=" * 60)
    logger.info(f"Target: {args.target} compounds")
    logger.info(f"MW range: {args.min_mw} - {args.max_mw}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Resume: {not args.no_resume}")
    logger.info("=" * 60)

    try:
        df = collector.collect_compounds(
            target_count=args.target,
            max_mw=args.max_mw,
            min_mw=args.min_mw,
            resume=not args.no_resume
        )

        logger.info("=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total compounds: {len(df)}")
        logger.info(f"Total requests: {collector.total_requests}")
        logger.info(f"Failed requests: {collector.failed_requests}")
        logger.info(f"Success rate: {(1 - collector.failed_requests/max(collector.total_requests,1))*100:.1f}%")
        logger.info(f"Output file: {collector.data_file}")
        logger.info("=" * 60)

        # Print property statistics
        collector.get_statistics()

    except KeyboardInterrupt:
        logger.warning("\nCollection interrupted by user")
        logger.info("Progress has been saved. Resume with the same command.")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == '__main__':
    main()

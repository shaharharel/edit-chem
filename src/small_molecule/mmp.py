"""Matched Molecular Pair (MMP) extraction."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMMPA, AllChem

from src.core.edit import Edit
from src.utils.chemistry import smiles_to_mol, standardize_smiles

logger = logging.getLogger(__name__)


@dataclass
class MatchedPair:
    """
    Represents a matched molecular pair.

    Two molecules that differ by a single transformation (edit).
    """
    smiles1: str
    smiles2: str
    edit: Edit
    core: str  # Common core structure
    delta_property: Optional[float] = None  # Change in property

    def __repr__(self):
        return f"MMP({self.smiles1[:20]}... -> {self.smiles2[:20]}..., {self.edit})"


class MMPExtractor:
    """
    Extract Matched Molecular Pairs from compound series.

    Uses RDKit's rdMMPA module to find molecules that differ
    by single transformations, which are ideal for causal effect estimation.
    """

    def __init__(self, max_cuts: int = 2, min_core_size: int = 5):
        """
        Initialize MMP extractor.

        Args:
            max_cuts: Maximum number of bonds to cut (1-3)
            min_core_size: Minimum number of heavy atoms in core
        """
        self.max_cuts = max_cuts
        self.min_core_size = min_core_size

    def extract_pairs(self,
                     smiles_list: List[str],
                     properties_dict: Optional[Dict[str, float]] = None,
                     property_name: str = None) -> List[MatchedPair]:
        """
        Extract all matched pairs from a list of molecules.

        Args:
            smiles_list: List of SMILES strings
            properties_dict: Optional mapping of SMILES -> property value
            property_name: Name of the property for the edit

        Returns:
            List of MatchedPair objects
        """
        logger.info(f"Extracting MMPs from {len(smiles_list)} molecules...")

        # Standardize SMILES
        standardized = {}
        for smi in smiles_list:
            std_smi = standardize_smiles(smi)
            if std_smi:
                standardized[std_smi] = smi

        logger.info(f"Standardized to {len(standardized)} unique molecules")

        # Fragment all molecules
        fragments_by_core = defaultdict(list)

        for std_smi in standardized.keys():
            mol = smiles_to_mol(std_smi)
            if mol is None:
                continue

            # Get fragments using RDKit's MMP algorithm
            try:
                frags = rdMMPA.FragmentMol(mol, maxCuts=self.max_cuts, resultsAsMols=False)

                for core, chains in frags:
                    # Filter by minimum core size
                    core_mol = Chem.MolFromSmiles(core)
                    if core_mol and core_mol.GetNumHeavyAtoms() >= self.min_core_size:
                        fragments_by_core[core].append((std_smi, chains))

            except Exception as e:
                logger.warning(f"Failed to fragment {std_smi}: {e}")

        logger.info(f"Found {len(fragments_by_core)} unique cores")

        # Find pairs with same core but different substituents
        pairs = []
        edit_counter = 0

        for core, molecules in fragments_by_core.items():
            if len(molecules) < 2:
                continue

            # Compare all pairs with this core
            for i in range(len(molecules)):
                for j in range(i + 1, len(molecules)):
                    smi1, chains1 = molecules[i]
                    smi2, chains2 = molecules[j]

                    # Check if they differ by exactly one substituent
                    if self._is_single_point_change(chains1, chains2):
                        # Create edit
                        from_smarts, to_smarts = self._extract_transformation(chains1, chains2)

                        if from_smarts and to_smarts:
                            edit = Edit(
                                edit_id=f"edit_{edit_counter}",
                                from_smarts=from_smarts,
                                to_smarts=to_smarts,
                                context_smarts=core,
                                edit_type="substitution"
                            )

                            # Calculate property delta if provided
                            delta = None
                            if properties_dict:
                                try:
                                    # Use original SMILES for property lookup
                                    orig1 = standardized.get(smi1, smi1)
                                    orig2 = standardized.get(smi2, smi2)

                                    if orig1 in properties_dict and orig2 in properties_dict:
                                        delta = properties_dict[orig2] - properties_dict[orig1]
                                except Exception as e:
                                    logger.warning(f"Failed to compute delta: {e}")

                            pair = MatchedPair(
                                smiles1=smi1,
                                smiles2=smi2,
                                edit=edit,
                                core=core,
                                delta_property=delta
                            )

                            pairs.append(pair)
                            edit_counter += 1

        logger.info(f"Extracted {len(pairs)} matched pairs")
        return pairs

    def _is_single_point_change(self, chains1: str, chains2: str) -> bool:
        """
        Check if two chain strings differ by exactly one substituent.

        Args:
            chains1: Fragment chains from molecule 1
            chains2: Fragment chains from molecule 2

        Returns:
            True if single point change, False otherwise
        """
        # Parse chain strings (format: "[R1]...[R2]...")
        # For now, simple implementation - assume single change if different
        return chains1 != chains2

    def _extract_transformation(self, chains1: str, chains2: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract from/to SMARTS patterns from fragment chains.

        Args:
            chains1: Chains from molecule 1 (source)
            chains2: Chains from molecule 2 (target)

        Returns:
            (from_smarts, to_smarts) tuple
        """
        try:
            # Parse chains (simplified - RDKit returns in specific format)
            # Format is typically: "[*]CCO.[*]C" for multiple attachment points
            # We'll use the chains directly as SMARTS for now
            return (chains1, chains2)
        except Exception as e:
            logger.warning(f"Failed to extract transformation: {e}")
            return (None, None)

    def aggregate_pairs_by_edit(self, pairs: List[MatchedPair]) -> Dict[str, List[MatchedPair]]:
        """
        Group matched pairs by their edit.

        This allows us to see how many times each transformation was observed.

        Args:
            pairs: List of MatchedPair objects

        Returns:
            Dictionary mapping edit_id -> list of pairs
        """
        grouped = defaultdict(list)

        for pair in pairs:
            # Use hash of transformation as key
            key = f"{pair.edit.from_smarts}>>>{pair.edit.to_smarts}"
            grouped[key].append(pair)

        return dict(grouped)

    def compute_edit_statistics(self, pairs: List[MatchedPair]) -> Dict[str, dict]:
        """
        Compute statistics for each unique edit.

        Args:
            pairs: List of MatchedPair objects with delta_property values

        Returns:
            Dictionary mapping edit_key -> statistics
        """
        import numpy as np

        grouped = self.aggregate_pairs_by_edit(pairs)
        statistics = {}

        for edit_key, edit_pairs in grouped.items():
            deltas = [p.delta_property for p in edit_pairs if p.delta_property is not None]

            if deltas:
                statistics[edit_key] = {
                    'n_observations': len(deltas),
                    'mean_delta': np.mean(deltas),
                    'std_delta': np.std(deltas),
                    'min_delta': np.min(deltas),
                    'max_delta': np.max(deltas)
                }

        return statistics

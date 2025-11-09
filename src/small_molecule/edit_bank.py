"""File-based edit bank for storing edits and their effects."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np

from src.core.edit import Edit
from src.core.effect import CausalEffect

logger = logging.getLogger(__name__)


class EditBank:
    """
    Repository of molecular edits and their observed effects.

    Stores edits and effects as JSON files for persistence.
    Provides methods for querying, aggregating, and analyzing edit effects.
    """

    def __init__(self, data_dir: str = "data/edit_bank"):
        """
        Initialize edit bank.

        Args:
            data_dir: Directory to store edit bank files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.edits_file = self.data_dir / "edits_registry.json"
        self.effects_file = self.data_dir / "edit_effects.json"

        # Load existing data
        self.edits = self._load_edits()
        self.effects = self._load_effects()

        logger.info(f"Loaded edit bank: {len(self.edits)} edits, {len(self.effects)} effect records")

    def add_edit(self, edit: Edit) -> None:
        """
        Register a new edit in the bank.

        Args:
            edit: Edit object to register
        """
        if edit.edit_id not in self.edits:
            self.edits[edit.edit_id] = edit.to_dict()
            self._save_edits()
            logger.debug(f"Added edit: {edit.edit_id}")

    def add_effect(self, effect: CausalEffect) -> None:
        """
        Record an observed effect of an edit.

        Args:
            effect: CausalEffect object to record
        """
        # Ensure edit is registered
        self.add_edit(effect.edit)

        # Add effect
        if effect.edit.edit_id not in self.effects:
            self.effects[effect.edit.edit_id] = []

        self.effects[effect.edit.edit_id].append(effect.to_dict())
        self._save_effects()

        logger.debug(f"Added effect for edit {effect.edit.edit_id}: Δ{effect.property_name}={effect.delta_mean}")

    def add_effects_batch(self, effects: List[CausalEffect]) -> None:
        """
        Add multiple effects efficiently.

        Args:
            effects: List of CausalEffect objects
        """
        for effect in effects:
            self.add_edit(effect.edit)

            if effect.edit.edit_id not in self.effects:
                self.effects[effect.edit.edit_id] = []

            self.effects[effect.edit.edit_id].append(effect.to_dict())

        self._save_edits()
        self._save_effects()

        logger.info(f"Added {len(effects)} effects to edit bank")

    def get_edit(self, edit_id: str) -> Optional[Edit]:
        """
        Retrieve an edit by ID.

        Args:
            edit_id: Edit identifier

        Returns:
            Edit object or None if not found
        """
        if edit_id not in self.edits:
            return None

        return Edit.from_dict(self.edits[edit_id])

    def get_edit_effects(self, edit_id: str, property_name: Optional[str] = None) -> List[dict]:
        """
        Retrieve all observed effects for an edit.

        Args:
            edit_id: Edit identifier
            property_name: Optional filter by property name

        Returns:
            List of effect dictionaries
        """
        effects = self.effects.get(edit_id, [])

        if property_name:
            effects = [e for e in effects if e['property_name'] == property_name]

        return effects

    def aggregate_effects(self, edit_id: str, property_name: str) -> Optional[dict]:
        """
        Compute aggregate statistics for an edit's effect on a property.

        Args:
            edit_id: Edit identifier
            property_name: Property name

        Returns:
            Dictionary with mean, std, n, etc., or None if no data
        """
        effects = self.get_edit_effects(edit_id, property_name)

        if not effects:
            return None

        deltas = [e['delta_mean'] for e in effects]

        return {
            'edit_id': edit_id,
            'property_name': property_name,
            'mean': np.mean(deltas),
            'std': np.std(deltas),
            'min': np.min(deltas),
            'max': np.max(deltas),
            'n_observations': len(deltas),
            'datasets': list(set(e['source_dataset'] for e in effects))
        }

    def get_all_edits(self) -> List[Edit]:
        """
        Get all registered edits.

        Returns:
            List of Edit objects
        """
        return [Edit.from_dict(data) for data in self.edits.values()]

    def get_edits_by_property(self, property_name: str) -> List[str]:
        """
        Get all edit IDs that have effects on a specific property.

        Args:
            property_name: Property name

        Returns:
            List of edit IDs
        """
        edit_ids = set()

        for edit_id, effects in self.effects.items():
            for effect in effects:
                if effect['property_name'] == property_name:
                    edit_ids.add(edit_id)

        return list(edit_ids)

    def get_statistics_summary(self) -> dict:
        """
        Get overall statistics about the edit bank.

        Returns:
            Dictionary with summary statistics
        """
        # Count effects by property
        property_counts = defaultdict(int)
        for effects_list in self.effects.values():
            for effect in effects_list:
                property_counts[effect['property_name']] += 1

        # Count unique transformations
        unique_transforms = set()
        for edit_data in self.edits.values():
            key = f"{edit_data['from_smarts']}>>>{edit_data['to_smarts']}"
            unique_transforms.add(key)

        return {
            'n_edits': len(self.edits),
            'n_unique_transformations': len(unique_transforms),
            'n_total_effects': sum(len(e) for e in self.effects.values()),
            'effects_by_property': dict(property_counts)
        }

    def find_similar_edits(self, edit: Edit, threshold: float = 0.8) -> List[Edit]:
        """
        Find edits similar to the given edit.

        Uses simple string similarity for now - could be enhanced with
        chemical similarity measures.

        Args:
            edit: Reference edit
            threshold: Similarity threshold (0-1)

        Returns:
            List of similar Edit objects
        """
        # Simple implementation - exact match on from/to
        similar = []

        for edit_data in self.edits.values():
            if (edit_data['from_smarts'] == edit.from_smarts or
                edit_data['to_smarts'] == edit.to_smarts):
                similar.append(Edit.from_dict(edit_data))

        return similar

    def export_to_csv(self, output_path: str, property_name: Optional[str] = None) -> None:
        """
        Export edit bank to CSV for analysis.

        Args:
            output_path: Path to output CSV file
            property_name: Optional filter by property
        """
        import pandas as pd

        rows = []

        for edit_id, effects in self.effects.items():
            edit = self.get_edit(edit_id)

            for effect in effects:
                if property_name and effect['property_name'] != property_name:
                    continue

                rows.append({
                    'edit_id': edit_id,
                    'from_smarts': edit.from_smarts,
                    'to_smarts': edit.to_smarts,
                    'context_smarts': edit.context_smarts,
                    'property_name': effect['property_name'],
                    'delta_mean': effect['delta_mean'],
                    'delta_std': effect['delta_std'],
                    'n_observations': effect['n_observations'],
                    'source_dataset': effect['source_dataset']
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported {len(rows)} effects to {output_path}")

    def _load_edits(self) -> Dict:
        """Load edits from JSON file."""
        if self.edits_file.exists():
            with open(self.edits_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_edits(self) -> None:
        """Save edits to JSON file."""
        with open(self.edits_file, 'w') as f:
            json.dump(self.edits, f, indent=2)

    def _load_effects(self) -> Dict:
        """Load effects from JSON file."""
        if self.effects_file.exists():
            with open(self.effects_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_effects(self) -> None:
        """Save effects to JSON file."""
        with open(self.effects_file, 'w') as f:
            json.dump(self.effects, f, indent=2)

    def clear(self) -> None:
        """Clear all data from the edit bank."""
        self.edits = {}
        self.effects = {}
        self._save_edits()
        self._save_effects()
        logger.warning("Edit bank cleared")

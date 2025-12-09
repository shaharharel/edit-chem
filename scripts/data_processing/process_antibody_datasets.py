#!/usr/bin/env python3
"""
Process all antibody datasets into unified format.

This script converts AbAgym, SKEMPI2, AbBiBench, and Trastuzumab DMS datasets
into the unified AbEditPairs format, including structural data where available.

Usage:
    python scripts/data_processing/process_antibody_datasets.py --output data/antibody/unified

Output:
    - unified_antibody_data.json: Complete dataset in JSON format
    - unified_antibody_data.csv: Simplified CSV format
    - structures/: PDB files for entries with structural data
    - processing_report.json: Processing statistics and warnings
"""

import argparse
import json
import os
import re
import shutil
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AssayType(Enum):
    """Types of binding/activity assays."""
    BINDING_AFFINITY = "binding_affinity"
    DDG = "ddg"
    ENRICHMENT = "enrichment"
    EXPRESSION = "expression"
    STABILITY = "stability"
    BINARY_BINDING = "binary_binding"
    OTHER = "other"


@dataclass
class UnifiedMutation:
    """Unified mutation representation."""
    chain: str  # 'H' for heavy, 'L' for light
    position: int  # 0-indexed position in sequence
    from_aa: str
    to_aa: str
    pdb_position: Optional[str] = None  # PDB residue number (may have insertion codes)
    region: Optional[str] = None  # CDR1, CDR2, CDR3, FR1, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            'chain': self.chain,
            'position': self.position,
            'from_aa': self.from_aa,
            'to_aa': self.to_aa,
            'pdb_position': self.pdb_position,
            'region': self.region,
        }

    def __str__(self) -> str:
        return f"{self.chain}{self.from_aa}{self.position + 1}{self.to_aa}"


@dataclass
class UnifiedAntibodyEntry:
    """
    Unified format for antibody mutation data.

    This is the standardized format that all datasets are converted to.
    """
    # Identifiers
    entry_id: str
    antibody_id: str
    antigen_id: Optional[str]
    source_dataset: str

    # Sequences
    heavy_wt: str
    light_wt: str  # Empty string for nanobodies/VHH

    # Mutations
    mutations: List[Dict[str, Any]]  # List of mutation dicts

    # Measurements
    delta_value: float
    assay_type: str

    # Optional measurements
    raw_wt_value: Optional[float] = None
    raw_mut_value: Optional[float] = None

    # Structure information
    has_structure: bool = False
    structure_id: Optional[str] = None  # PDB ID
    structure_path: Optional[str] = None  # Path to local PDB file
    interface_distance: Optional[float] = None  # Distance to antigen interface

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def heavy_mut(self) -> str:
        """Get mutant heavy chain sequence."""
        seq = list(self.heavy_wt)
        for mut in self.mutations:
            if mut['chain'].upper() == 'H' and mut['position'] < len(seq):
                seq[mut['position']] = mut['to_aa']
        return ''.join(seq)

    @property
    def light_mut(self) -> str:
        """Get mutant light chain sequence."""
        if not self.light_wt:
            return ''
        seq = list(self.light_wt)
        for mut in self.mutations:
            if mut['chain'].upper() in ['L', 'K'] and mut['position'] < len(seq):
                seq[mut['position']] = mut['to_aa']
        return ''.join(seq)

    @property
    def num_mutations(self) -> int:
        return len(self.mutations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_id': self.entry_id,
            'antibody_id': self.antibody_id,
            'antigen_id': self.antigen_id,
            'source_dataset': self.source_dataset,
            'heavy_wt': self.heavy_wt,
            'light_wt': self.light_wt,
            'heavy_mut': self.heavy_mut,
            'light_mut': self.light_mut,
            'mutations': self.mutations,
            'num_mutations': self.num_mutations,
            'delta_value': self.delta_value,
            'assay_type': self.assay_type,
            'raw_wt_value': self.raw_wt_value,
            'raw_mut_value': self.raw_mut_value,
            'has_structure': self.has_structure,
            'structure_id': self.structure_id,
            'structure_path': self.structure_path,
            'interface_distance': self.interface_distance,
            'metadata': self.metadata,
        }


class DatasetProcessor:
    """Base class for dataset processors."""

    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.warnings = []
        self.stats = defaultdict(int)

    def process(self) -> List[UnifiedAntibodyEntry]:
        raise NotImplementedError

    def log_warning(self, msg: str):
        self.warnings.append(msg)

    def get_report(self) -> Dict[str, Any]:
        return {
            'stats': dict(self.stats),
            'warnings': self.warnings[:100],  # Limit warnings
            'num_warnings': len(self.warnings),
        }


class AbAgymProcessor(DatasetProcessor):
    """
    Process AbAgym dataset.

    AbAgym contains 324k single-site mutations from 68 DMS experiments
    with PDB structures and interface information.
    """

    def process(self) -> List[UnifiedAntibodyEntry]:
        print("Processing AbAgym dataset...")

        # Load data
        data_file = self.data_dir / "abagym" / "AbAgym_data_non-redundant.csv"
        meta_file = self.data_dir / "abagym" / "AbAgym_metadata.csv"
        seq_file = self.data_dir / "abagym" / "abagym_sequences.json"

        if not data_file.exists():
            raise FileNotFoundError(f"AbAgym data not found at {data_file}")

        df = pd.read_csv(data_file)
        meta_df = pd.read_csv(meta_file) if meta_file.exists() else None

        # Load sequences from JSON file
        sequences = {}
        if seq_file.exists():
            with open(seq_file, 'r') as f:
                sequences = json.load(f)
            print(f"  Loaded sequences for {len(sequences)} antibodies")

        # Create antigen lookup from metadata
        antigen_lookup = {}
        if meta_df is not None:
            for _, row in meta_df.iterrows():
                antigen_lookup[row['DMS_name']] = row.get('antigen_name', 'unknown')

        # Process entries
        entries = []

        # Group by DMS experiment to get parent sequences
        for dms_name, group in df.groupby('DMS_name'):
            self.stats['dms_experiments'] += 1

            # Get PDB file info
            pdb_file = group.iloc[0]['PDB_file']
            pdb_id = self._extract_pdb_id(pdb_file)

            # Get sequences for this antibody
            ab_seqs = sequences.get(dms_name, {})
            heavy_wt = ab_seqs.get('heavy', '')
            light_wt = ab_seqs.get('light', '')

            if not heavy_wt:
                self.log_warning(f"No sequence found for {dms_name}")
                self.stats['missing_sequences'] += 1

            for _, row in group.iterrows():
                try:
                    entry = self._process_abagym_row(row, dms_name, antigen_lookup, pdb_id, heavy_wt, light_wt)
                    if entry:
                        entries.append(entry)
                        self.stats['entries_processed'] += 1
                except Exception as e:
                    self.log_warning(f"Error processing AbAgym row: {e}")
                    self.stats['entries_failed'] += 1

        print(f"  Processed {len(entries)} entries from {self.stats['dms_experiments']} experiments")
        return entries

    def _extract_pdb_id(self, pdb_file: str) -> str:
        """Extract PDB ID from filename."""
        # Format like: "G6_27_30A_corrected_4zfg"
        match = re.search(r'_([0-9][a-z0-9]{3})$', pdb_file.lower())
        if match:
            return match.group(1).upper()
        return pdb_file

    def _process_abagym_row(
        self,
        row: pd.Series,
        dms_name: str,
        antigen_lookup: Dict[str, str],
        pdb_id: str,
        heavy_wt: str,
        light_wt: str,
    ) -> Optional[UnifiedAntibodyEntry]:
        """Process a single AbAgym row."""

        # Parse mutation
        mut_name = row['mut_names']  # e.g., "PH100A"
        chain = row['chains']  # Chain identifier
        site = int(row['site'])  # Position
        wt_aa = row['wildtype']
        mut_aa = row['mutation']

        # Determine if heavy or light chain
        # AbAgym uses various chain identifiers
        ab_chain = 'H'  # Default to heavy
        if chain in ['L', 'K', 'l', 'k']:
            ab_chain = 'L'
        elif any(c in chain.upper() for c in ['L', 'K']):
            # Multi-chain format, try to infer
            if mut_name.startswith('L') or mut_name.startswith('K'):
                ab_chain = 'L'

        mutation = UnifiedMutation(
            chain=ab_chain,
            position=site - 1,  # Convert to 0-indexed
            from_aa=wt_aa,
            to_aa=mut_aa,
            pdb_position=str(site),
        )

        # Get DMS score
        dms_score = row['DMS_score']
        interface_dist = row.get('closest_interface_atom_distance', None)

        # Create entry with sequences from the JSON file
        entry = UnifiedAntibodyEntry(
            entry_id=f"abagym_{dms_name}_{mut_name}",
            antibody_id=dms_name,
            antigen_id=antigen_lookup.get(dms_name, 'unknown'),
            source_dataset='abagym',
            heavy_wt=heavy_wt,
            light_wt=light_wt,
            mutations=[mutation.to_dict()],
            delta_value=float(dms_score),
            assay_type=AssayType.ENRICHMENT.value,
            has_structure=True,
            structure_id=pdb_id,
            interface_distance=float(interface_dist) if pd.notna(interface_dist) else None,
            metadata={
                'dms_name': dms_name,
                'pdb_file': row['PDB_file'],
                'normalized_score': row.get('MinMax_normalized_DMS_score', None),
                'original_chain': chain,
            },
        )

        return entry


class SKEMPI2Processor(DatasetProcessor):
    """
    Process SKEMPI2 dataset (antibody subset).

    SKEMPI2 contains explicit WT/Mutant pairs with ddG values
    and PDB structures.
    """

    def process(self) -> List[UnifiedAntibodyEntry]:
        print("Processing SKEMPI2 dataset...")

        data_file = self.data_dir / "skempi2" / "skempi_v2.csv"
        seq_file = self.data_dir / "skempi2" / "skempi2_chain_info.json"

        if not data_file.exists():
            raise FileNotFoundError(f"SKEMPI2 data not found at {data_file}")

        df = pd.read_csv(data_file, sep=';')

        # Load sequences from JSON file
        sequences = {}
        if seq_file.exists():
            with open(seq_file, 'r') as f:
                sequences = json.load(f)
            print(f"  Loaded sequences for {len(sequences)} complexes")

        # Filter for antibody entries
        antibody_keywords = ['antibody', 'fab', 'igg', 'fv', 'scfv', 'nanobody', 'vhh', 'immunoglobulin']
        is_antibody = df.apply(lambda r: any(
            kw in str(r.get('Protein 1', '')).lower() or kw in str(r.get('Protein 2', '')).lower()
            for kw in antibody_keywords
        ), axis=1)

        ab_df = df[is_antibody].copy()
        self.stats['total_antibody_entries'] = len(ab_df)

        entries = []

        for _, row in ab_df.iterrows():
            try:
                entry = self._process_skempi2_row(row, sequences)
                if entry:
                    entries.append(entry)
                    self.stats['entries_processed'] += 1
            except Exception as e:
                self.log_warning(f"Error processing SKEMPI2 row: {e}")
                self.stats['entries_failed'] += 1

        print(f"  Processed {len(entries)} entries")
        return entries

    def _process_skempi2_row(self, row: pd.Series, sequences: Dict[str, Any]) -> Optional[UnifiedAntibodyEntry]:
        """Process a single SKEMPI2 row."""

        pdb_entry = row.get('#Pdb', '')
        pdb_id = pdb_entry.split('_')[0]
        mutation_str = row.get('Mutation(s)_cleaned', '')

        # Parse mutations
        mutations = self._parse_skempi_mutations(mutation_str)
        if not mutations:
            return None

        # Calculate ddG from Kd values
        ddg = self._calculate_ddg(row)
        if ddg is None:
            return None

        # Determine antibody vs antigen
        protein1 = row.get('Protein 1', '')
        protein2 = row.get('Protein 2', '')

        if any(kw in protein1.lower() for kw in ['antibody', 'fab', 'igg']):
            antigen_id = protein2
        else:
            antigen_id = protein1

        # Get sequences from the JSON file
        # Try different key formats: "1AHW_AB_C" or just "1AHW"
        heavy_wt = ''
        light_wt = ''
        for key in [pdb_entry, pdb_id]:
            if key in sequences:
                seq_info = sequences[key]
                heavy_wt = seq_info.get('heavy_sequence', '')
                light_wt = seq_info.get('light_sequence', '')
                break

        entry = UnifiedAntibodyEntry(
            entry_id=f"skempi2_{pdb_id}_{mutation_str.replace(',', '_')}",
            antibody_id=pdb_id,
            antigen_id=antigen_id,
            source_dataset='skempi2',
            heavy_wt=heavy_wt,
            light_wt=light_wt,
            mutations=[m.to_dict() for m in mutations],
            delta_value=ddg,
            assay_type=AssayType.DDG.value,
            has_structure=True,
            structure_id=pdb_id,
            metadata={
                'temperature': row.get('Temperature', None),
                'kd_wt': row.get('Affinity_wt_parsed', None),
                'kd_mut': row.get('Affinity_mut_parsed', None),
                'protein1': protein1,
                'protein2': protein2,
            },
        )

        return entry

    def _parse_skempi_mutations(self, mutation_str: str) -> List[UnifiedMutation]:
        """Parse SKEMPI2 mutation string."""
        if not mutation_str or pd.isna(mutation_str):
            return []

        mutations = []
        parts = str(mutation_str).split(',')

        for part in parts:
            part = part.strip()
            if not part or len(part) < 3:
                continue

            # Format: XC123Y where X=from_aa, C=chain, 123=position, Y=to_aa
            # Or: CX123Y where C=chain
            try:
                # Try to parse
                if part[1].isalpha() and not part[1].isdigit():
                    # Format like TC110A - first char is from_aa
                    from_aa = part[0]
                    chain = part[1]
                    pos_str = ''
                    for i, c in enumerate(part[2:], 2):
                        if c.isdigit():
                            pos_str += c
                        else:
                            to_aa = c
                            break
                    position = int(pos_str)
                else:
                    # Try simpler format
                    from_aa = part[0]
                    to_aa = part[-1]
                    pos_str = part[1:-1]
                    # Extract chain if present
                    chain = 'H'
                    if pos_str[0].isalpha():
                        chain = pos_str[0]
                        pos_str = pos_str[1:]
                    position = int(pos_str)

                # Map chain identifiers
                if chain.upper() in ['A', 'B', 'H']:
                    chain = 'H'
                elif chain.upper() in ['C', 'D', 'L', 'K']:
                    chain = 'L'

                mutations.append(UnifiedMutation(
                    chain=chain,
                    position=position - 1,  # 0-indexed
                    from_aa=from_aa,
                    to_aa=to_aa,
                    pdb_position=str(position),
                ))
            except (ValueError, IndexError) as e:
                continue

        return mutations

    def _calculate_ddg(self, row: pd.Series) -> Optional[float]:
        """Calculate ddG from Kd values."""
        kd_mut = row.get('Affinity_mut_parsed', None)
        kd_wt = row.get('Affinity_wt_parsed', None)

        if pd.isna(kd_mut) or pd.isna(kd_wt):
            return None

        try:
            kd_mut = float(kd_mut)
            kd_wt = float(kd_wt)

            if kd_wt <= 0 or kd_mut <= 0:
                return None

            # Parse temperature
            temp_str = str(row.get('Temperature', '298'))
            match = re.search(r'(\d+(?:\.\d+)?)', temp_str)
            temp = float(match.group(1)) if match else 298.0

            # ddG = RT * ln(Kd_mut / Kd_wt)
            R = 0.001987  # kcal/(mol*K)
            ddg = R * temp * math.log(kd_mut / kd_wt)

            return ddg
        except (ValueError, TypeError):
            return None


class AbBiBenchProcessor(DatasetProcessor):
    """
    Process AbBiBench dataset.

    AbBiBench contains full sequences with binding scores
    but no explicit mutation information - we need to construct pairs.
    """

    def process(self) -> List[UnifiedAntibodyEntry]:
        print("Processing AbBiBench dataset...")

        data_file = self.data_dir / "abbibench" / "train.csv"

        if not data_file.exists():
            raise FileNotFoundError(f"AbBiBench data not found at {data_file}")

        df = pd.read_csv(data_file)
        self.stats['total_samples'] = len(df)

        # AbBiBench doesn't have explicit mutations
        # We can either:
        # 1. Use it as-is for binding prediction (not edit-based)
        # 2. Construct pairs by finding related sequences

        # For now, we'll group by shared heavy chains and
        # find pairs that differ by a few mutations

        entries = []

        # Group by heavy chain
        heavy_groups = df.groupby('heavy_chain_seq')

        for heavy_seq, group in heavy_groups:
            if len(group) < 2:
                continue

            self.stats['heavy_chain_groups'] += 1

            # Sort by binding score to find best reference
            group_sorted = group.sort_values('binding_score', ascending=False)

            # Use highest scoring as reference
            ref_row = group_sorted.iloc[0]
            ref_light = ref_row['light_chain_seq']
            ref_score = ref_row['binding_score']

            # Create pairs with other entries
            for i, (_, row) in enumerate(group_sorted.iloc[1:].iterrows()):
                # Find mutations in light chain
                light_seq = row['light_chain_seq']
                score = row['binding_score']

                if len(light_seq) != len(ref_light):
                    continue

                # Find differences
                mutations = []
                for pos, (a, b) in enumerate(zip(ref_light, light_seq)):
                    if a != b:
                        mutations.append(UnifiedMutation(
                            chain='L',
                            position=pos,
                            from_aa=a,
                            to_aa=b,
                        ))

                # Only include if 1-5 mutations
                if 0 < len(mutations) <= 5:
                    entry = UnifiedAntibodyEntry(
                        entry_id=f"abbibench_{hash(heavy_seq) % 100000}_{i}",
                        antibody_id=f"abbibench_{hash(heavy_seq) % 100000}",
                        antigen_id=None,
                        source_dataset='abbibench',
                        heavy_wt=heavy_seq,
                        light_wt=ref_light,
                        mutations=[m.to_dict() for m in mutations],
                        delta_value=score - ref_score,  # Delta binding score
                        assay_type=AssayType.BINDING_AFFINITY.value,
                        raw_wt_value=ref_score,
                        raw_mut_value=score,
                        has_structure=False,
                        metadata={
                            'ref_binding_score': ref_score,
                            'mut_binding_score': score,
                        },
                    )
                    entries.append(entry)
                    self.stats['pairs_created'] += 1

        print(f"  Created {len(entries)} mutation pairs from {self.stats['heavy_chain_groups']} sequence groups")
        return entries


class TrastuzumabProcessor(DatasetProcessor):
    """
    Process Trastuzumab CDRH3 DMS dataset.

    Contains binary binding classification for CDRH3 variants.
    """

    # Trastuzumab wild-type CDRH3 (10 aa)
    WT_CDRH3 = "WGGDGFYAMD"  # This is the typical trastuzumab CDRH3

    def process(self) -> List[UnifiedAntibodyEntry]:
        print("Processing Trastuzumab DMS dataset...")

        pos_file = self.data_dir / "trastuzumab_dms" / "data" / "mHER_H3_AgPos.csv"
        neg_file = self.data_dir / "trastuzumab_dms" / "data" / "mHER_H3_AgNeg.csv"

        if not pos_file.exists() or not neg_file.exists():
            raise FileNotFoundError(f"Trastuzumab data not found")

        pos_df = pd.read_csv(pos_file)
        neg_df = pd.read_csv(neg_file)

        pos_df['binding'] = 1
        neg_df['binding'] = 0

        df = pd.concat([pos_df, neg_df], ignore_index=True)
        self.stats['total_variants'] = len(df)

        entries = []

        for idx, row in df.iterrows():
            try:
                entry = self._process_trastuzumab_row(row, idx)
                if entry:
                    entries.append(entry)
                    self.stats['entries_processed'] += 1
            except Exception as e:
                self.log_warning(f"Error processing Trastuzumab row {idx}: {e}")
                self.stats['entries_failed'] += 1

        print(f"  Processed {len(entries)} CDRH3 variants")
        return entries

    def _process_trastuzumab_row(self, row: pd.Series, idx: int) -> Optional[UnifiedAntibodyEntry]:
        """Process a single Trastuzumab row."""

        cdrh3_seq = row['AASeq']
        binding = row['binding']

        if len(cdrh3_seq) != 10:
            return None

        # Find mutations relative to WT
        # Note: The dataset may have a different WT, so we'll use the first
        # sequence as reference or mark all positions as the variant
        mutations = []

        for pos, (wt_aa, mut_aa) in enumerate(zip(self.WT_CDRH3, cdrh3_seq)):
            if wt_aa != mut_aa:
                mutations.append(UnifiedMutation(
                    chain='H',
                    position=pos,  # Position within CDRH3
                    from_aa=wt_aa,
                    to_aa=mut_aa,
                    region='CDRH3',
                ))

        entry = UnifiedAntibodyEntry(
            entry_id=f"trastuzumab_{idx}",
            antibody_id="trastuzumab",
            antigen_id="HER2",
            source_dataset='trastuzumab_dms',
            heavy_wt=self.WT_CDRH3,  # Just the CDRH3 region
            light_wt='',
            mutations=[m.to_dict() for m in mutations],
            delta_value=float(binding),
            assay_type=AssayType.BINARY_BINDING.value,
            has_structure=False,  # Could add trastuzumab structure
            metadata={
                'cdrh3_sequence': cdrh3_seq,
                'count': row.get('Count', None),
                'fraction': row.get('Fraction', None),
            },
        )

        return entry


def process_all_datasets(
    data_dir: Path,
    output_dir: Path,
    include_abbibench: bool = True,
    include_skempi2: bool = True,
    include_abagym: bool = True,
    include_trastuzumab: bool = True,
) -> Dict[str, Any]:
    """
    Process all antibody datasets and combine into unified format.

    Returns:
        Processing report with statistics
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    reports = {}

    # Process each dataset
    if include_abagym:
        try:
            processor = AbAgymProcessor(data_dir, output_dir)
            entries = processor.process()
            all_entries.extend(entries)
            reports['abagym'] = processor.get_report()
        except Exception as e:
            print(f"  Error processing AbAgym: {e}")
            reports['abagym'] = {'error': str(e)}

    if include_skempi2:
        try:
            processor = SKEMPI2Processor(data_dir, output_dir)
            entries = processor.process()
            all_entries.extend(entries)
            reports['skempi2'] = processor.get_report()
        except Exception as e:
            print(f"  Error processing SKEMPI2: {e}")
            reports['skempi2'] = {'error': str(e)}

    if include_abbibench:
        try:
            processor = AbBiBenchProcessor(data_dir, output_dir)
            entries = processor.process()
            all_entries.extend(entries)
            reports['abbibench'] = processor.get_report()
        except Exception as e:
            print(f"  Error processing AbBiBench: {e}")
            reports['abbibench'] = {'error': str(e)}

    if include_trastuzumab:
        try:
            processor = TrastuzumabProcessor(data_dir, output_dir)
            entries = processor.process()
            all_entries.extend(entries)
            reports['trastuzumab'] = processor.get_report()
        except Exception as e:
            print(f"  Error processing Trastuzumab: {e}")
            reports['trastuzumab'] = {'error': str(e)}

    print(f"\nTotal entries: {len(all_entries)}")

    # Save unified dataset
    print("Saving unified dataset...")

    # JSON format (complete)
    json_path = output_dir / "unified_antibody_data.json"
    with open(json_path, 'w') as f:
        json.dump([e.to_dict() for e in all_entries], f, indent=2)
    print(f"  Saved JSON: {json_path}")

    # CSV format (simplified)
    csv_data = []
    for entry in all_entries:
        d = entry.to_dict()
        # Flatten mutations
        d['mutations_str'] = ','.join(
            f"{m['chain']}{m['from_aa']}{m['position']+1}{m['to_aa']}"
            for m in d['mutations']
        )
        del d['mutations']
        del d['metadata']
        csv_data.append(d)

    csv_path = output_dir / "unified_antibody_data.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # Summary statistics
    summary = {
        'total_entries': len(all_entries),
        'by_source': {},
        'by_assay_type': {},
        'with_structure': sum(1 for e in all_entries if e.has_structure),
        'single_mutations': sum(1 for e in all_entries if e.num_mutations == 1),
        'multi_mutations': sum(1 for e in all_entries if e.num_mutations > 1),
    }

    for entry in all_entries:
        source = entry.source_dataset
        assay = entry.assay_type
        summary['by_source'][source] = summary['by_source'].get(source, 0) + 1
        summary['by_assay_type'][assay] = summary['by_assay_type'].get(assay, 0) + 1

    # Save report
    report = {
        'summary': summary,
        'processors': reports,
    }

    report_path = output_dir / "processing_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report: {report_path}")

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total entries: {summary['total_entries']:,}")
    print(f"With structure: {summary['with_structure']:,}")
    print(f"Single mutations: {summary['single_mutations']:,}")
    print(f"Multi mutations: {summary['multi_mutations']:,}")
    print("\nBy source:")
    for source, count in summary['by_source'].items():
        print(f"  {source}: {count:,}")
    print("\nBy assay type:")
    for assay, count in summary['by_assay_type'].items():
        print(f"  {assay}: {count:,}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description='Process antibody datasets into unified format'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/antibody',
        help='Directory containing raw datasets'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/antibody/unified',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--skip_abbibench',
        action='store_true',
        help='Skip AbBiBench processing'
    )
    parser.add_argument(
        '--skip_skempi2',
        action='store_true',
        help='Skip SKEMPI2 processing'
    )
    parser.add_argument(
        '--skip_abagym',
        action='store_true',
        help='Skip AbAgym processing'
    )
    parser.add_argument(
        '--skip_trastuzumab',
        action='store_true',
        help='Skip Trastuzumab processing'
    )

    args = parser.parse_args()

    process_all_datasets(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        include_abbibench=not args.skip_abbibench,
        include_skempi2=not args.skip_skempi2,
        include_abagym=not args.skip_abagym,
        include_trastuzumab=not args.skip_trastuzumab,
    )


if __name__ == '__main__':
    main()

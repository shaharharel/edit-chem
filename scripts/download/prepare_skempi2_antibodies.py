"""
Prepare SKEMPI2 antibody data by fetching sequences from PDB.

This script:
1. Identifies antibody entries in SKEMPI2
2. Fetches sequences from PDB
3. Creates a processed CSV with full sequences

Usage:
    python scripts/download/prepare_skempi2_antibodies.py --input data/antibody/skempi2/skempi_v2.csv
"""

import argparse
import requests
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict


def fetch_pdb_sequences(pdb_id: str) -> Dict[str, str]:
    """
    Fetch sequences for all chains from PDB using GraphQL API.

    Args:
        pdb_id: PDB ID (e.g., "1AHW")

    Returns:
        Dict mapping chain ID to sequence
    """
    url = 'https://data.rcsb.org/graphql'
    query = '''
    {
      entry(entry_id: "%s") {
        polymer_entities {
          rcsb_polymer_entity_container_identifiers {
            auth_asym_ids
          }
          entity_poly {
            pdbx_seq_one_letter_code_can
          }
        }
      }
    }
    ''' % pdb_id.lower()

    try:
        response = requests.post(url, json={'query': query}, timeout=30)
        if response.status_code != 200:
            return {}

        data = response.json()
        entry = data.get('data', {}).get('entry', {})
        if not entry:
            return {}

        chains = {}
        for entity in entry.get('polymer_entities', []):
            chain_ids = entity.get('rcsb_polymer_entity_container_identifiers', {}).get('auth_asym_ids', [])
            sequence = entity.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can', '')

            for chain_id in chain_ids:
                chains[chain_id] = sequence

        return chains

    except Exception as e:
        print(f"  Error fetching {pdb_id}: {e}")
        return {}


def identify_heavy_light_chains(chains: Dict[str, str], pdb_chain_str: str) -> Tuple[str, str, str, str]:
    """
    Identify heavy and light chain sequences.

    Args:
        chains: Dict mapping chain ID to sequence
        pdb_chain_str: PDB ID with chain info (e.g., "1AHW_AB_C" where AB is antibody, C is antigen)

    Returns:
        (heavy_chain_id, heavy_seq, light_chain_id, light_seq)
    """
    parts = pdb_chain_str.split('_')
    if len(parts) < 2:
        return '', '', '', ''

    pdb_id = parts[0]
    ab_chains = parts[1] if len(parts) > 1 else ''

    # Antibody chains are typically listed first
    heavy_id = ''
    heavy_seq = ''
    light_id = ''
    light_seq = ''

    # Try to identify H/L chains from chain letters
    for i, chain_letter in enumerate(ab_chains):
        if chain_letter not in chains:
            continue

        seq = chains[chain_letter]

        # Heuristic: Heavy chains are usually longer
        if not heavy_seq or len(seq) > len(heavy_seq):
            if heavy_seq:
                # Previous heavy becomes light
                light_id = heavy_id
                light_seq = heavy_seq
            heavy_id = chain_letter
            heavy_seq = seq
        elif not light_seq or len(seq) > len(light_seq):
            light_id = chain_letter
            light_seq = seq

    return heavy_id, heavy_seq, light_id, light_seq


def prepare_skempi2_antibodies(
    input_file: str,
    output_dir: str,
    max_pdbs: Optional[int] = None,
) -> str:
    """
    Prepare SKEMPI2 antibody data with full sequences.

    Args:
        input_file: Path to SKEMPI2 CSV
        output_dir: Output directory
        max_pdbs: Maximum number of PDBs to process (for testing)

    Returns:
        Path to output file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SKEMPI2
    df = pd.read_csv(input_file, sep=';')
    print(f"Loaded {len(df)} SKEMPI2 entries")

    # Filter for antibody entries
    antibody_keywords = ['antibody', 'fab', 'igg', 'fv', 'scfv', 'nanobody', 'vhh', 'immunoglobulin']

    is_antibody = df.apply(lambda r: any(
        kw in str(r.get('Protein 1', '')).lower() or kw in str(r.get('Protein 2', '')).lower()
        for kw in antibody_keywords
    ), axis=1)

    antibody_df = df[is_antibody].copy()
    print(f"Found {len(antibody_df)} antibody entries")

    # Get unique PDB IDs
    unique_pdbs = antibody_df['#Pdb'].unique()
    print(f"Unique PDB structures: {len(unique_pdbs)}")

    if max_pdbs:
        unique_pdbs = unique_pdbs[:max_pdbs]
        print(f"Processing first {max_pdbs} PDBs")

    # Fetch sequences for each PDB
    pdb_sequences = {}
    pdb_chain_info = {}

    for i, pdb_str in enumerate(unique_pdbs):
        pdb_id = pdb_str.split('_')[0]

        print(f"[{i+1}/{len(unique_pdbs)}] Fetching {pdb_id}...")

        if pdb_id not in pdb_sequences:
            chains = fetch_pdb_sequences(pdb_id)
            pdb_sequences[pdb_id] = chains
            time.sleep(0.5)  # Be nice to PDB API

        chains = pdb_sequences[pdb_id]
        if chains:
            heavy_id, heavy_seq, light_id, light_seq = identify_heavy_light_chains(chains, pdb_str)
            pdb_chain_info[pdb_str] = {
                'heavy_chain': heavy_id,
                'heavy_sequence': heavy_seq,
                'light_chain': light_id,
                'light_sequence': light_seq,
            }
            print(f"  Found H={heavy_id} ({len(heavy_seq) if heavy_seq else 0}aa), L={light_id} ({len(light_seq) if light_seq else 0}aa)")
        else:
            print(f"  No chains found")

    # Add sequences to dataframe
    antibody_df['heavy_chain'] = antibody_df['#Pdb'].map(lambda x: pdb_chain_info.get(x, {}).get('heavy_chain', ''))
    antibody_df['heavy_sequence'] = antibody_df['#Pdb'].map(lambda x: pdb_chain_info.get(x, {}).get('heavy_sequence', ''))
    antibody_df['light_chain'] = antibody_df['#Pdb'].map(lambda x: pdb_chain_info.get(x, {}).get('light_chain', ''))
    antibody_df['light_sequence'] = antibody_df['#Pdb'].map(lambda x: pdb_chain_info.get(x, {}).get('light_sequence', ''))

    # Filter out entries without sequences
    has_sequence = antibody_df['heavy_sequence'].str.len() > 0
    antibody_df_filtered = antibody_df[has_sequence]

    print(f"\nEntries with sequences: {len(antibody_df_filtered)}")

    # Save
    output_file = output_dir / 'skempi2_antibodies.csv'
    antibody_df_filtered.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")

    # Also save chain info as JSON for reference
    chain_info_file = output_dir / 'skempi2_chain_info.json'
    with open(chain_info_file, 'w') as f:
        json.dump(pdb_chain_info, f, indent=2)

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='Prepare SKEMPI2 antibody data')
    parser.add_argument('--input', type=str, default='data/antibody/skempi2/skempi_v2.csv',
                        help='Input SKEMPI2 CSV file')
    parser.add_argument('--output_dir', type=str, default='data/antibody/skempi2',
                        help='Output directory')
    parser.add_argument('--max_pdbs', type=int, default=None,
                        help='Max PDBs to process (for testing)')

    args = parser.parse_args()

    prepare_skempi2_antibodies(args.input, args.output_dir, args.max_pdbs)


if __name__ == '__main__':
    main()

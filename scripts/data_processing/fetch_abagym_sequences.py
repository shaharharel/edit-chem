#!/usr/bin/env python3
"""
Fetch antibody sequences from PDB for AbAgym dataset.

This script extracts VH/VL sequences from PDB structures referenced in the
AbAgym metadata file and creates a sequences JSON file for the loader.
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd


def fetch_pdb_sequence(pdb_id: str, chain_id: str) -> Optional[str]:
    """
    Fetch sequence for a specific chain from PDB.

    Uses the RCSB PDB REST API to get chain sequences.
    """
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None

        # This returns polymer entities, need to find by chain
        # Try a different approach - get chain sequences directly
        return None
    except Exception as e:
        print(f"Error fetching {pdb_id}: {e}")
        return None


def fetch_chain_sequences_from_pdb(pdb_id: str) -> Dict[str, str]:
    """
    Fetch all chain sequences from a PDB entry.

    Uses the RCSB PDB GraphQL API for reliable sequence extraction.
    """
    url = "https://data.rcsb.org/graphql"

    query = """
    query ($pdb_id: String!) {
        entry(entry_id: $pdb_id) {
            polymer_entities {
                entity_poly {
                    pdbx_strand_id
                    pdbx_seq_one_letter_code_can
                }
                rcsb_polymer_entity {
                    pdbx_description
                }
            }
        }
    }
    """

    try:
        response = requests.post(
            url,
            json={"query": query, "variables": {"pdb_id": pdb_id.upper()}},
            timeout=30
        )

        if response.status_code != 200:
            print(f"  Failed to fetch {pdb_id}: HTTP {response.status_code}")
            return {}

        data = response.json()

        if not data.get("data", {}).get("entry"):
            print(f"  No entry found for {pdb_id}")
            return {}

        chains = {}
        for entity in data["data"]["entry"]["polymer_entities"]:
            entity_poly = entity.get("entity_poly", {})
            if not entity_poly:
                continue

            strand_ids = entity_poly.get("pdbx_strand_id", "")
            sequence = entity_poly.get("pdbx_seq_one_letter_code_can", "")
            description = entity.get("rcsb_polymer_entity", {}).get("pdbx_description", "")

            if not sequence:
                continue

            # Clean sequence (remove newlines, etc.)
            sequence = sequence.replace("\n", "").replace(" ", "")

            # Map to chain IDs
            for chain_id in strand_ids.split(","):
                chain_id = chain_id.strip()
                if chain_id:
                    chains[chain_id] = {
                        "sequence": sequence,
                        "description": description.lower()
                    }

        return chains

    except Exception as e:
        print(f"  Error fetching {pdb_id}: {e}")
        return {}


def identify_antibody_chains(chains: Dict[str, dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify heavy and light chain IDs from chain data.

    Uses descriptions and chain naming conventions.
    """
    heavy_chain = None
    light_chain = None

    for chain_id, info in chains.items():
        desc = info.get("description", "")
        seq = info.get("sequence", "")

        # Skip short sequences (likely small molecules or peptides)
        if len(seq) < 100:
            continue

        # Check description for chain type
        if any(kw in desc for kw in ["heavy", "vh", "fab heavy", "h chain"]):
            heavy_chain = chain_id
        elif any(kw in desc for kw in ["light", "vl", "kappa", "lambda", "fab light", "l chain"]):
            light_chain = chain_id
        # Use naming convention (H/L are common)
        elif chain_id.upper() == "H" and not heavy_chain:
            heavy_chain = chain_id
        elif chain_id.upper() == "L" and not light_chain:
            light_chain = chain_id

    # Fallback: if no clear identification, try A/B convention
    if not heavy_chain and not light_chain:
        for chain_id, info in chains.items():
            seq = info.get("sequence", "")
            if len(seq) < 100:
                continue

            # A often heavy, B often light in Fab structures
            if chain_id.upper() == "A" and not heavy_chain:
                heavy_chain = chain_id
            elif chain_id.upper() == "B" and not light_chain:
                light_chain = chain_id

    return heavy_chain, light_chain


def process_abagym_metadata(metadata_path: Path, output_path: Path):
    """
    Process AbAgym metadata and fetch sequences for all antibodies.
    """
    df = pd.read_csv(metadata_path)

    sequences = {}
    failed = []

    print(f"Processing {len(df)} antibodies...")

    for idx, row in df.iterrows():
        dms_name = row["DMS_name"]
        pdb_id = row["template_PDB_ID"]

        print(f"[{idx+1}/{len(df)}] {dms_name} (PDB: {pdb_id})")

        # Fetch chain sequences
        chains = fetch_chain_sequences_from_pdb(pdb_id)

        if not chains:
            print(f"  Failed to fetch sequences")
            failed.append(dms_name)
            continue

        # Identify H/L chains
        heavy_id, light_id = identify_antibody_chains(chains)

        if not heavy_id:
            print(f"  Could not identify heavy chain from: {list(chains.keys())}")
            # Try to use first two protein chains
            protein_chains = [k for k, v in chains.items() if len(v["sequence"]) >= 100]
            if len(protein_chains) >= 1:
                heavy_id = protein_chains[0]
                if len(protein_chains) >= 2:
                    light_id = protein_chains[1]

        heavy_seq = chains.get(heavy_id, {}).get("sequence", "") if heavy_id else ""
        light_seq = chains.get(light_id, {}).get("sequence", "") if light_id else ""

        if not heavy_seq:
            print(f"  No heavy chain sequence found")
            failed.append(dms_name)
            continue

        sequences[dms_name] = {
            "heavy": heavy_seq,
            "light": light_seq,
            "pdb_id": pdb_id,
            "heavy_chain_id": heavy_id,
            "light_chain_id": light_id,
            "antigen": row.get("antigen_name", ""),
        }

        print(f"  Heavy ({heavy_id}): {len(heavy_seq)} aa, Light ({light_id}): {len(light_seq)} aa")

        # Rate limiting
        time.sleep(0.5)

    # Save sequences
    with open(output_path, "w") as f:
        json.dump(sequences, f, indent=2)

    print(f"\nSaved {len(sequences)} antibody sequences to {output_path}")

    if failed:
        print(f"\nFailed to process {len(failed)} antibodies:")
        for name in failed:
            print(f"  - {name}")

    return sequences


def main():
    base_dir = Path(__file__).parent.parent.parent

    metadata_path = base_dir / "data" / "antibody" / "abagym" / "AbAgym_metadata.csv"
    output_path = base_dir / "data" / "antibody" / "abagym" / "abagym_sequences.json"

    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return

    sequences = process_abagym_metadata(metadata_path, output_path)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total antibodies: {len(sequences)}")

    # Count unique PDB IDs
    pdb_ids = set(s["pdb_id"] for s in sequences.values())
    print(f"Unique PDB structures: {len(pdb_ids)}")

    # Sequence length stats
    heavy_lens = [len(s["heavy"]) for s in sequences.values() if s["heavy"]]
    light_lens = [len(s["light"]) for s in sequences.values() if s["light"]]

    if heavy_lens:
        print(f"Heavy chain length: {min(heavy_lens)}-{max(heavy_lens)} aa (mean: {sum(heavy_lens)/len(heavy_lens):.0f})")
    if light_lens:
        print(f"Light chain length: {min(light_lens)}-{max(light_lens)} aa (mean: {sum(light_lens)/len(light_lens):.0f})")


if __name__ == "__main__":
    main()

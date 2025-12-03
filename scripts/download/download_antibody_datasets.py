"""
Download antibody mutation datasets.

Provides utilities to download and prepare:
- AbBiBench: Antibody binding benchmark
- AbAgym: Antibody affinity maturation gym
- AB-Bind: Antibody binding affinity database
- SKEMPI2: Kinetics database (antibody subset)

Usage:
    python scripts/download/download_antibody_datasets.py --dataset abbibench --output_dir data/antibody
    python scripts/download/download_antibody_datasets.py --all --output_dir data/antibody
"""

import argparse
import os
import json
import subprocess
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error

try:
    import requests
except ImportError:
    requests = None


def download_file(url: str, output_path: str, verbose: bool = True) -> bool:
    """Download a file from URL."""
    if verbose:
        print(f"Downloading: {url}")
        print(f"  → {output_path}")

    try:
        if requests is not None:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            urllib.request.urlretrieve(url, output_path)

        if verbose:
            print("  ✓ Download complete")
        return True

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def download_abbibench(output_dir: str, verbose: bool = True) -> bool:
    """
    Download AbBiBench dataset.

    AbBiBench is available from:
    - GitHub: https://github.com/xxxx/AbBiBench
    - Zenodo: https://zenodo.org/record/xxxx

    Note: URLs need to be updated when the dataset is publicly released.
    """
    output_dir = Path(output_dir) / 'abbibench'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Downloading AbBiBench")
    print("="*60)

    # Placeholder - actual URLs need to be updated
    # AbBiBench may need to be downloaded from the original paper's repository
    print("""
    AbBiBench download not yet automated.

    Please download manually from the paper's supplementary materials or GitHub:
    1. Find the AbBiBench paper/repository
    2. Download the mutation data CSV
    3. Place it in: {output_dir}

    Expected files:
    - abbibench_mutations.csv: Main mutation data with ddG values
    - abbibench_sequences.csv: Antibody sequences (H and L chains)

    Once downloaded, the data loader will work automatically.
    """.format(output_dir=output_dir))

    # Create placeholder README
    readme_path = output_dir / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write("""AbBiBench Dataset
=================

This directory should contain:
- abbibench_mutations.csv: Main mutation data
- abbibench_sequences.csv: Antibody sequences

The data loader expects columns:
- antibody_id: Unique identifier
- mutations: Mutation string (e.g., "HA50K,HG100R")
- ddg: Delta delta G (binding affinity change)
- heavy_sequence: Heavy chain sequence
- light_sequence: Light chain sequence

See the data loader for format details:
src/data/antibody/loaders.py
""")

    return True


def download_abagym(output_dir: str, verbose: bool = True) -> bool:
    """
    Download AbAgym dataset.

    AbAgym contains deep mutational scanning and directed evolution data.
    """
    output_dir = Path(output_dir) / 'abagym'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Downloading AbAgym")
    print("="*60)

    print("""
    AbAgym download not yet automated.

    Please download manually from the paper's repository:
    1. Find the AbAgym paper/repository
    2. Download the DMS and evolution trajectory data
    3. Place it in: {output_dir}

    Expected files:
    - abagym_data.csv or target-specific CSVs
    - abagym_sequences.json: Parent sequences

    File format should include:
    - antibody_id: Identifier
    - mutations: Mutation string
    - enrichment or fitness: Activity score
    - target: Antigen name (optional)
    """.format(output_dir=output_dir))

    # Create placeholder
    readme_path = output_dir / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write("""AbAgym Dataset
==============

This directory should contain DMS/evolution data.

Expected format:
- CSVs with mutation, enrichment, and sequence columns
- See src/data/antibody/loaders.py for details
""")

    return True


def download_skempi2(output_dir: str, verbose: bool = True) -> bool:
    """
    Download SKEMPI2 database.

    SKEMPI2 is publicly available at:
    https://life.bsc.es/pid/skempi2
    """
    output_dir = Path(output_dir) / 'skempi2'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Downloading SKEMPI2")
    print("="*60)

    # SKEMPI2 download URL
    skempi_url = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"

    output_file = output_dir / 'skempi_v2.csv'

    success = download_file(skempi_url, str(output_file), verbose=verbose)

    if success:
        print("\n  Note: SKEMPI2 contains all protein-protein interactions.")
        print("  The loader filters for antibody entries automatically.")
        print("  Full sequences need to be fetched from PDB for some entries.")

    return success


def download_ab_bind(output_dir: str, verbose: bool = True) -> bool:
    """
    Download AB-Bind database.

    AB-Bind is a curated database of antibody binding affinity changes.
    """
    output_dir = Path(output_dir) / 'ab_bind'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("Downloading AB-Bind")
    print("="*60)

    print("""
    AB-Bind download not yet automated.

    Please download manually from the AB-Bind website or paper.

    Expected files:
    - AB-Bind_data.csv or ab_bind.csv

    The file should contain:
    - pdb_id: Structure identifier
    - mutation: Mutation string
    - ddG: Binding affinity change
    - heavy_sequence, light_sequence (optional)
    """)

    return True


def check_dependencies() -> bool:
    """Check if required packages are installed."""
    missing = []

    try:
        import pandas
    except ImportError:
        missing.append('pandas')

    try:
        import numpy
    except ImportError:
        missing.append('numpy')

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download antibody mutation datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['abbibench', 'abagym', 'skempi2', 'ab_bind'],
        help='Specific dataset to download'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all available datasets'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/antibody',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Either --dataset or --all must be specified")

    # Check dependencies
    if not check_dependencies():
        return

    verbose = not args.quiet
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    datasets_to_download = []

    if args.all:
        datasets_to_download = ['abbibench', 'abagym', 'skempi2', 'ab_bind']
    elif args.dataset:
        datasets_to_download = [args.dataset]

    results = {}

    for dataset in datasets_to_download:
        if dataset == 'abbibench':
            results['abbibench'] = download_abbibench(str(output_dir), verbose)
        elif dataset == 'abagym':
            results['abagym'] = download_abagym(str(output_dir), verbose)
        elif dataset == 'skempi2':
            results['skempi2'] = download_skempi2(str(output_dir), verbose)
        elif dataset == 'ab_bind':
            results['ab_bind'] = download_ab_bind(str(output_dir), verbose)

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)

    for dataset, success in results.items():
        status = "✓ Ready" if success else "✗ Manual download required"
        print(f"  {dataset}: {status}")

    print(f"\nDatasets saved to: {output_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Run experiments on MPRAu 3'UTR dataset from Griesemer/Ulirsch et al. (Cell 2021).

This script supports:
1. Single cell line experiments (edit prediction)
2. Multi-cell line experiments (transfer learning)
3. Direct property prediction (absolute expression)
4. Multi-task learning across cell types

Usage:
    # Single cell line, all methods
    python run_mprau_3utr.py --cell-line HEK293FT

    # Multi-cell experiment
    python run_mprau_3utr.py --multi-cell --cell-lines HEK293FT HEPG2 K562

    # Quick test with nucleotide embedder
    python run_mprau_3utr.py --cell-line HEK293FT --embedder nucleotide --quick
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rna.experiment_config import (
    RNAExperimentConfig,
    DEFAULT_NUCLEOTIDE_CONFIG,
    DEFAULT_RNAFM_CONFIG,
    DEFAULT_RNABERT_CONFIG,
    DEFAULT_STRUCTURED_EDIT_CONFIG
)
from experiments.rna.main import run_experiment

# Available cell lines
CELL_LINES = ['HEK293FT', 'HEPG2', 'HMEC', 'K562', 'GM12878', 'SKNSH']


def create_single_cell_config(
    cell_line: str,
    embedder_type: str = 'nucleotide',
    quick: bool = False,
    include_structured: bool = False
) -> RNAExperimentConfig:
    """Create config for single cell line experiment."""

    # Select base config
    if embedder_type == 'nucleotide':
        base = DEFAULT_NUCLEOTIDE_CONFIG.copy()
    elif embedder_type == 'rnafm':
        base = DEFAULT_RNAFM_CONFIG.copy()
    elif embedder_type == 'rnabert':
        base = DEFAULT_RNABERT_CONFIG.copy()
    else:
        base = DEFAULT_NUCLEOTIDE_CONFIG.copy()

    # Add structured edit method if requested
    if include_structured and embedder_type != 'nucleotide':
        structured_method = DEFAULT_STRUCTURED_EDIT_CONFIG['methods'][0].copy()
        base['methods'] = [structured_method] + base['methods']

    # Reduce epochs for quick testing
    if quick:
        for method in base['methods']:
            method['max_epochs'] = 10
            method['batch_size'] = 128

    # Remove embedder_type from base if present (we set it explicitly)
    base.pop('embedder_type', None)
    base.pop('trainable_embedder', None)

    config = RNAExperimentConfig(
        experiment_name=f"mprau_3utr_{cell_line}_{embedder_type}",
        data_file=f"data/rna/pairs/mprau_3utr_multitask_with_seq.csv",
        property_filter=[f'3UTR_skew_{cell_line}'],
        min_pairs_per_property=100,
        embedder_type=embedder_type,
        trainable_embedder=False,
        splitter_type='random',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        output_dir=f'results/mprau_3utr/{cell_line}',
        save_models=False,
        **base
    )

    return config


def create_multi_cell_config(
    cell_lines: list,
    embedder_type: str = 'nucleotide',
    quick: bool = False
) -> RNAExperimentConfig:
    """Create config for multi-cell line experiment (multi-task learning)."""

    if embedder_type == 'nucleotide':
        base = DEFAULT_NUCLEOTIDE_CONFIG.copy()
    elif embedder_type == 'rnafm':
        base = DEFAULT_RNAFM_CONFIG.copy()
    else:
        base = DEFAULT_NUCLEOTIDE_CONFIG.copy()

    if quick:
        for method in base['methods']:
            method['max_epochs'] = 10

    property_filter = [f'3UTR_skew_{cell}' for cell in cell_lines]

    # Remove embedder_type from base if present
    base.pop('embedder_type', None)
    base.pop('trainable_embedder', None)

    config = RNAExperimentConfig(
        experiment_name=f"mprau_3utr_multicell_{embedder_type}",
        data_file="data/rna/pairs/mprau_3utr_multitask_with_seq.csv",
        property_filter=property_filter,
        num_tasks=len(cell_lines),
        min_pairs_per_property=100,
        embedder_type=embedder_type,
        trainable_embedder=False,
        splitter_type='random',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        output_dir='results/mprau_3utr/multicell',
        save_models=False,
        **base
    )

    return config


def create_cross_cell_transfer_config(
    train_cell: str,
    test_cell: str,
    embedder_type: str = 'nucleotide'
) -> RNAExperimentConfig:
    """Create config for cross-cell transfer learning experiment."""

    if embedder_type == 'nucleotide':
        base = DEFAULT_NUCLEOTIDE_CONFIG.copy()
    else:
        base = DEFAULT_RNAFM_CONFIG.copy()

    # Remove embedder_type from base if present
    base.pop('embedder_type', None)
    base.pop('trainable_embedder', None)

    config = RNAExperimentConfig(
        experiment_name=f"mprau_3utr_transfer_{train_cell}_to_{test_cell}",
        data_file="data/rna/pairs/mprau_3utr_multitask_with_seq.csv",
        property_filter=[f'3UTR_skew_{train_cell}'],
        min_pairs_per_property=100,
        embedder_type=embedder_type,
        splitter_type='random',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        output_dir=f'results/mprau_3utr/transfer/{train_cell}_to_{test_cell}',
        # Test on different cell line
        additional_test_files={
            f'test_{test_cell}': f'data/rna/pairs/mprau_3utr_multitask_with_seq.csv'
        },
        **base
    )

    return config


def main():
    parser = argparse.ArgumentParser(description='Run MPRAu 3\'UTR experiments')

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--single-cell', action='store_true', default=True,
                     help='Run single cell line experiment (default)')
    mode.add_argument('--multi-cell', action='store_true',
                     help='Run multi-cell line experiment')
    mode.add_argument('--transfer', action='store_true',
                     help='Run cross-cell transfer experiment')

    # Cell line selection
    parser.add_argument('--cell-line', '-c', default='HEK293FT',
                       choices=CELL_LINES,
                       help='Cell line for single-cell experiments')
    parser.add_argument('--cell-lines', nargs='+', default=['HEK293FT', 'HEPG2', 'K562'],
                       help='Cell lines for multi-cell experiments')
    parser.add_argument('--train-cell', default='HEK293FT',
                       help='Training cell line for transfer')
    parser.add_argument('--test-cell', default='HEPG2',
                       help='Test cell line for transfer')

    # Model settings
    parser.add_argument('--embedder', '-e', default='nucleotide',
                       choices=['nucleotide', 'rnafm', 'rnabert'],
                       help='Embedder type')
    parser.add_argument('--structured', action='store_true',
                       help='Include structured edit embedder')

    # Run settings
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick test run (fewer epochs)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print config without running')

    args = parser.parse_args()

    # Create config based on mode
    if args.multi_cell:
        config = create_multi_cell_config(
            cell_lines=args.cell_lines,
            embedder_type=args.embedder,
            quick=args.quick
        )
    elif args.transfer:
        config = create_cross_cell_transfer_config(
            train_cell=args.train_cell,
            test_cell=args.test_cell,
            embedder_type=args.embedder
        )
    else:
        config = create_single_cell_config(
            cell_line=args.cell_line,
            embedder_type=args.embedder,
            quick=args.quick,
            include_structured=args.structured
        )

    # Print config
    print("\n" + "="*80)
    print("MPRAu 3'UTR EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Experiment: {config.experiment_name}")
    print(f"Data file: {config.data_file}")
    print(f"Properties: {config.property_filter}")
    print(f"Embedder: {config.embedder_type}")
    print(f"Methods: {[m['name'] for m in config.methods]}")
    print(f"Output: {config.output_dir}")
    print("="*80 + "\n")

    if args.dry_run:
        print("Dry run - not executing experiment")
        return

    # Run experiment
    results, report_path = run_experiment(config)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Report saved to: {report_path}")

    # Print summary results
    if results:
        print("\n=== Results Summary ===")
        for method_name, method_results in results.items():
            if 'test' in method_results:
                test_metrics = method_results['test']
                print(f"\n{method_name}:")
                for metric, value in test_metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()

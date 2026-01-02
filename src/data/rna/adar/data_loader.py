"""
ADAR RNA editing data loader.

Processes RNA-seq data to create a dataset of edited and non-edited adenosine
positions for learning RNA editing site preference.

Data requirements:
- Reference genome (FASTA)
- Gene annotations (GTF)
- Edited sites table (TSV with coordinates, base counts, editing rates)
- Pileup files (per-position coverage from RNA-seq)
"""

import lzma
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_genome(fasta_path: Path) -> Dict[str, str]:
    """
    Load reference genome from FASTA file.

    Uses pyfaidx for efficient random access if available,
    otherwise falls back to simple parsing.

    Args:
        fasta_path: Path to genome FASTA file

    Returns:
        Dict mapping chromosome names to sequences
    """
    try:
        from pyfaidx import Fasta
        genome = Fasta(str(fasta_path))
        return {name: str(genome[name][:]).upper() for name in genome.keys()}
    except ImportError:
        logger.warning("pyfaidx not available, using simple parser (slower)")
        return _parse_fasta_simple(fasta_path)


def _parse_fasta_simple(fasta_path: Path) -> Dict[str, str]:
    """Simple FASTA parser fallback."""
    genome = {}
    current_chrom = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_chrom is not None:
                    genome[current_chrom] = ''.join(current_seq).upper()
                current_chrom = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_chrom is not None:
            genome[current_chrom] = ''.join(current_seq).upper()

    return genome


def load_transcript_annotations(gtf_path: Path) -> pd.DataFrame:
    """
    Load gene/transcript annotations from GTF file.

    Extracts transcript boundaries and strand information.

    Args:
        gtf_path: Path to GTF annotation file

    Returns:
        DataFrame with transcript annotations
    """
    records = []

    with open(gtf_path) as f:
        for line in f:
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            feature_type = fields[2]
            if feature_type not in ('transcript', 'exon'):
                continue

            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]

            # Parse attributes
            attrs = {}
            for attr in fields[8].split(';'):
                attr = attr.strip()
                if ' ' in attr:
                    key, value = attr.split(' ', 1)
                    attrs[key] = value.strip('"')

            records.append({
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': strand,
                'feature_type': feature_type,
                'gene_id': attrs.get('gene_id', ''),
                'gene_name': attrs.get('gene_name', ''),
                'transcript_id': attrs.get('transcript_id', ''),
            })

    return pd.DataFrame(records)


def load_edited_sites(
    tsv_path: Path,
    min_coverage: int = 20,
    min_editing_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Load edited sites from filtered results table.

    Args:
        tsv_path: Path to edited sites TSV file
        min_coverage: Minimum coverage threshold
        min_editing_rate: Minimum editing rate threshold

    Returns:
        DataFrame with edited sites
    """
    df = pd.read_csv(tsv_path, sep='\t')

    # Clean up chromosome names (remove quotes if present)
    df['Chromosome'] = df['Chromosome'].str.strip('"')
    df['Gbase'] = df['Gbase'].str.strip('"')

    # Rename columns for consistency
    df = df.rename(columns={
        'Chromosome': 'chrom',
        'Coordinate': 'position',
        'Gbase': 'ref_base',
        'Coverage': 'coverage',
        'EditingRate': 'editing_rate',
        'P_value_FDR': 'pvalue',
    })

    # Determine strand from reference base
    # A = plus strand (A->G editing)
    # T = minus strand (T->C on genome = A->G on transcript)
    df['strand'] = df['ref_base'].map({'A': '+', 'T': '-'})

    # Apply filters
    df = df[df['coverage'] >= min_coverage]
    df = df[df['editing_rate'] >= min_editing_rate]

    # Add label
    df['is_edited'] = 1

    logger.info(f"Loaded {len(df)} edited sites after filtering")

    return df[['chrom', 'position', 'strand', 'ref_base', 'coverage',
               'editing_rate', 'pvalue', 'is_edited', 'A', 'C', 'G', 'T']]


def load_pileup_data(
    pileup_path: Path,
    min_coverage: int = 10,
) -> pd.DataFrame:
    """
    Load pileup data from compressed cmpileup file.

    Format: chrom, region_start, region_end, position, ref_base,
            coverage, A_count, C_count, G_count, T_count, N, del

    Args:
        pileup_path: Path to .cmpileup.xz file
        min_coverage: Minimum coverage to include

    Returns:
        DataFrame with per-position pileup data
    """
    # Determine if file is compressed
    if str(pileup_path).endswith('.xz'):
        opener = lzma.open
    else:
        opener = open

    records = []
    with opener(pileup_path, 'rt') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 11:
                continue

            coverage = int(fields[5])
            if coverage < min_coverage:
                continue

            ref_base = fields[4]
            # Only keep A and T positions (potential editing sites)
            if ref_base not in ('A', 'T'):
                continue

            records.append({
                'chrom': fields[0],
                'position': int(fields[3]),
                'ref_base': ref_base,
                'coverage': coverage,
                'A': int(fields[6]),
                'C': int(fields[7]),
                'G': int(fields[8]),
                'T': int(fields[9]),
            })

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} A/T positions from pileup with coverage >= {min_coverage}")

    return df


def aggregate_pileups(
    pileup_paths: List[Path],
    min_coverage_per_sample: int = 5,
    min_total_coverage: int = 20,
) -> pd.DataFrame:
    """
    Aggregate pileup data from multiple samples.

    Sums base counts across samples for the same positions.

    Args:
        pileup_paths: List of paths to pileup files
        min_coverage_per_sample: Minimum coverage in at least one sample
        min_total_coverage: Minimum total coverage after aggregation

    Returns:
        Aggregated DataFrame
    """
    all_dfs = []
    for path in pileup_paths:
        logger.info(f"Loading pileup: {path}")
        df = load_pileup_data(path, min_coverage=min_coverage_per_sample)
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    # Concatenate and aggregate
    combined = pd.concat(all_dfs, ignore_index=True)

    # Group by position and sum counts
    agg_df = combined.groupby(['chrom', 'position', 'ref_base']).agg({
        'coverage': 'sum',
        'A': 'sum',
        'C': 'sum',
        'G': 'sum',
        'T': 'sum',
    }).reset_index()

    # Apply coverage filter
    agg_df = agg_df[agg_df['coverage'] >= min_total_coverage]

    logger.info(f"Aggregated to {len(agg_df)} positions with total coverage >= {min_total_coverage}")

    return agg_df


# =============================================================================
# Strand and Sequence Handling
# =============================================================================

def get_transcript_strand(
    chrom: str,
    position: int,
    annotations: pd.DataFrame,
    chrom_mapping: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Get strand of transcript(s) overlapping a position.

    Args:
        chrom: Chromosome name
        position: Genomic position (1-based)
        annotations: Transcript annotations DataFrame
        chrom_mapping: Optional mapping between chromosome naming conventions

    Returns:
        '+' or '-' if unambiguous, None if no transcript or ambiguous
    """
    # Apply chromosome name mapping if needed
    query_chrom = chrom
    if chrom_mapping and chrom in chrom_mapping:
        query_chrom = chrom_mapping[chrom]

    # Find overlapping transcripts
    overlapping = annotations[
        (annotations['chrom'] == query_chrom) &
        (annotations['feature_type'] == 'transcript') &
        (annotations['start'] <= position) &
        (annotations['end'] >= position)
    ]

    if len(overlapping) == 0:
        return None

    strands = overlapping['strand'].unique()
    if len(strands) == 1:
        return strands[0]

    # Ambiguous (overlapping genes on both strands)
    return None


def reverse_complement(seq: str) -> str:
    """Compute reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(n, 'N') for n in reversed(seq.upper()))


def dna_to_rna(seq: str) -> str:
    """Convert DNA sequence to RNA (T -> U)."""
    return seq.upper().replace('T', 'U')


def extract_sequence_window(
    genome: Dict[str, str],
    chrom: str,
    position: int,
    strand: str,
    window_size: int = 100,
    as_rna: bool = True,
) -> Optional[str]:
    """
    Extract sequence window centered on a position.

    For minus strand, returns reverse complement so that:
    - The center is always the adenosine that may be edited
    - The sequence represents the actual RNA molecule

    Args:
        genome: Dict of chromosome sequences
        chrom: Chromosome name
        position: 1-based genomic position
        strand: '+' or '-'
        window_size: Nucleotides on each side of center
        as_rna: Convert to RNA (T->U)

    Returns:
        Sequence window or None if position is invalid
    """
    if chrom not in genome:
        return None

    seq = genome[chrom]

    # Convert to 0-based
    pos_0 = position - 1

    # Calculate window boundaries
    start = pos_0 - window_size
    end = pos_0 + window_size + 1

    # Handle edge cases
    if start < 0 or end > len(seq):
        # Pad with N's if needed
        left_pad = max(0, -start)
        right_pad = max(0, end - len(seq))
        actual_start = max(0, start)
        actual_end = min(len(seq), end)

        window = 'N' * left_pad + seq[actual_start:actual_end] + 'N' * right_pad
    else:
        window = seq[start:end]

    # Handle strand
    if strand == '-':
        window = reverse_complement(window)

    if as_rna:
        window = dna_to_rna(window)

    return window


# =============================================================================
# Main Data Loader Class
# =============================================================================

@dataclass
class ADARDataConfig:
    """Configuration for ADAR data loading."""
    genome_path: Path
    gtf_path: Path
    edited_sites_path: Path
    pileup_paths: List[Path]

    # Filtering thresholds
    min_coverage: int = 20
    min_editing_rate_positive: float = 0.05
    max_editing_rate_negative: float = 0.01

    # Sequence extraction
    window_size: int = 100
    as_rna: bool = True

    # Negative sampling
    negative_ratio: Optional[float] = None  # None = keep all negatives
    balance_by_coverage: bool = True

    # Chromosome name mapping (pileup/edited -> GTF)
    chrom_prefix: str = 'chr'  # GTF uses 'chrI', pileup uses 'I'


class ADARDataLoader:
    """
    Main data loader for ADAR editing dataset.

    Creates a dataset of edited and non-edited adenosine positions
    with RNA sequence context for training editing site preference models.
    """

    def __init__(self, config: ADARDataConfig):
        """
        Initialize data loader.

        Args:
            config: Data loading configuration
        """
        self.config = config
        self._genome = None
        self._annotations = None
        self._chrom_mapping = None

    @property
    def genome(self) -> Dict[str, str]:
        """Lazy-load genome."""
        if self._genome is None:
            logger.info(f"Loading genome from {self.config.genome_path}")
            self._genome = load_genome(self.config.genome_path)
            logger.info(f"Loaded {len(self._genome)} chromosomes")
        return self._genome

    @property
    def annotations(self) -> pd.DataFrame:
        """Lazy-load annotations."""
        if self._annotations is None:
            logger.info(f"Loading annotations from {self.config.gtf_path}")
            self._annotations = load_transcript_annotations(self.config.gtf_path)
            logger.info(f"Loaded {len(self._annotations)} annotation records")
        return self._annotations

    @property
    def chrom_mapping(self) -> Dict[str, str]:
        """Build chromosome name mapping (e.g., 'I' -> 'chrI')."""
        if self._chrom_mapping is None:
            self._chrom_mapping = {}
            for chrom in self.genome.keys():
                # Map 'I' -> 'chrI' for GTF lookup
                if not chrom.startswith(self.config.chrom_prefix):
                    self._chrom_mapping[chrom] = f"{self.config.chrom_prefix}{chrom}"
        return self._chrom_mapping

    def load_edited_sites(self) -> pd.DataFrame:
        """Load and filter edited sites (positives)."""
        return load_edited_sites(
            self.config.edited_sites_path,
            min_coverage=self.config.min_coverage,
            min_editing_rate=self.config.min_editing_rate_positive,
        )

    def load_negative_candidates(self) -> pd.DataFrame:
        """
        Load negative candidates from pileup data.

        Returns positions that:
        - Have A or T as reference base
        - Have sufficient coverage
        - Have low or zero editing signal
        """
        # Aggregate pileups
        pileup_df = aggregate_pileups(
            self.config.pileup_paths,
            min_coverage_per_sample=5,
            min_total_coverage=self.config.min_coverage,
        )

        if len(pileup_df) == 0:
            return pd.DataFrame()

        # Calculate editing rate
        # For A positions: editing = G / (A + G)
        # For T positions: editing = C / (C + T)
        def calc_editing_rate(row):
            if row['ref_base'] == 'A':
                denom = row['A'] + row['G']
                return row['G'] / denom if denom > 0 else 0
            else:  # T
                denom = row['C'] + row['T']
                return row['C'] / denom if denom > 0 else 0

        pileup_df['editing_rate'] = pileup_df.apply(calc_editing_rate, axis=1)

        # Filter to low editing rate (negatives)
        negatives = pileup_df[
            pileup_df['editing_rate'] <= self.config.max_editing_rate_negative
        ].copy()

        # Assign strand based on reference base
        negatives['strand'] = negatives['ref_base'].map({'A': '+', 'T': '-'})
        negatives['is_edited'] = 0
        negatives['pvalue'] = np.nan

        logger.info(f"Found {len(negatives)} negative candidates")

        return negatives

    def create_dataset(
        self,
        restrict_to_transcripts: bool = True,
    ) -> pd.DataFrame:
        """
        Create the full dataset with positives and negatives.

        Args:
            restrict_to_transcripts: Only include positions within annotated transcripts

        Returns:
            DataFrame with columns:
                - chrom, position, strand, ref_base
                - coverage, editing_rate
                - is_edited (label)
                - sequence (RNA sequence window)
        """
        # Load positives
        positives = self.load_edited_sites()
        logger.info(f"Loaded {len(positives)} positive sites")

        # Load negatives
        negatives = self.load_negative_candidates()
        logger.info(f"Loaded {len(negatives)} negative candidates")

        # Remove positions that are in positives from negatives
        positive_positions = set(
            zip(positives['chrom'], positives['position'])
        )
        negatives = negatives[
            ~negatives.apply(
                lambda r: (r['chrom'], r['position']) in positive_positions,
                axis=1
            )
        ]
        logger.info(f"After removing overlaps: {len(negatives)} negatives")

        # Combine
        common_cols = ['chrom', 'position', 'strand', 'ref_base', 'coverage',
                       'editing_rate', 'is_edited', 'A', 'C', 'G', 'T']

        # Ensure both have required columns
        for col in common_cols:
            if col not in positives.columns:
                positives[col] = np.nan
            if col not in negatives.columns:
                negatives[col] = np.nan

        dataset = pd.concat([
            positives[common_cols],
            negatives[common_cols]
        ], ignore_index=True)

        # Optionally restrict to annotated transcripts
        if restrict_to_transcripts:
            dataset = self._filter_to_transcripts(dataset)

        # Extract sequence windows
        logger.info("Extracting sequence windows...")
        dataset['sequence'] = dataset.apply(
            lambda r: extract_sequence_window(
                self.genome,
                r['chrom'],
                r['position'],
                r['strand'],
                window_size=self.config.window_size,
                as_rna=self.config.as_rna,
            ),
            axis=1
        )

        # Remove positions where sequence extraction failed
        n_before = len(dataset)
        dataset = dataset[dataset['sequence'].notna()]
        n_after = len(dataset)
        if n_before != n_after:
            logger.warning(f"Removed {n_before - n_after} positions with invalid sequences")

        # Sample negatives if ratio specified
        if self.config.negative_ratio is not None:
            dataset = self._sample_negatives(dataset)

        logger.info(f"Final dataset: {len(dataset)} sites "
                   f"({dataset['is_edited'].sum()} edited, "
                   f"{(~dataset['is_edited'].astype(bool)).sum()} non-edited)")

        return dataset

    def _filter_to_transcripts(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset to positions within annotated transcripts."""
        logger.info("Filtering to positions within annotated transcripts...")

        # Get transcript positions
        transcripts = self.annotations[
            self.annotations['feature_type'] == 'transcript'
        ].copy()

        # Build a set of (chrom, position) that are within transcripts
        # Use vectorized approach: for each chromosome, find positions in any transcript

        in_transcript_mask = np.zeros(len(dataset), dtype=bool)

        # Map dataset chromosomes to GTF naming
        dataset_chroms = dataset['chrom'].apply(
            lambda c: self.chrom_mapping.get(c, c)
        )

        # Process by chromosome for efficiency
        for gtf_chrom in transcripts['chrom'].unique():
            # Get all transcripts on this chromosome
            chrom_transcripts = transcripts[transcripts['chrom'] == gtf_chrom]

            # Get dataset positions on this chromosome
            chrom_mask = dataset_chroms == gtf_chrom
            if not chrom_mask.any():
                continue

            chrom_positions = dataset.loc[chrom_mask, 'position'].values
            chrom_indices = np.where(chrom_mask)[0]

            # For each position, check if it falls in any transcript
            for _, tx in chrom_transcripts.iterrows():
                matches = (chrom_positions >= tx['start']) & (chrom_positions <= tx['end'])
                in_transcript_mask[chrom_indices[matches]] = True

        dataset = dataset[in_transcript_mask].copy()
        logger.info(f"After transcript filtering: {len(dataset)} positions")

        return dataset

    def _sample_negatives(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Sample negatives to achieve desired ratio."""
        positives = dataset[dataset['is_edited'] == 1]
        negatives = dataset[dataset['is_edited'] == 0]

        n_positives = len(positives)
        n_negatives_target = int(n_positives * self.config.negative_ratio)

        if len(negatives) <= n_negatives_target:
            logger.warning(f"Not enough negatives for ratio {self.config.negative_ratio}")
            return dataset

        if self.config.balance_by_coverage:
            # Sample to match coverage distribution
            sampled_negatives = self._sample_matching_coverage(
                negatives, positives, n_negatives_target
            )
        else:
            sampled_negatives = negatives.sample(n=n_negatives_target, random_state=42)

        return pd.concat([positives, sampled_negatives], ignore_index=True)

    def _sample_matching_coverage(
        self,
        negatives: pd.DataFrame,
        positives: pd.DataFrame,
        n_samples: int,
    ) -> pd.DataFrame:
        """Sample negatives to match coverage distribution of positives."""
        # Create coverage bins
        bins = np.percentile(positives['coverage'], [0, 25, 50, 75, 100])
        bins[0] = 0  # Ensure we catch all
        bins[-1] = float('inf')

        positive_counts = pd.cut(positives['coverage'], bins=bins).value_counts(normalize=True)

        sampled = []
        for interval, prop in positive_counts.items():
            n_from_bin = int(n_samples * prop)
            bin_negatives = negatives[
                (negatives['coverage'] >= interval.left) &
                (negatives['coverage'] < interval.right)
            ]
            if len(bin_negatives) > 0:
                n_sample = min(n_from_bin, len(bin_negatives))
                sampled.append(bin_negatives.sample(n=n_sample, random_state=42))

        return pd.concat(sampled, ignore_index=True) if sampled else pd.DataFrame()

    def get_summary_stats(self, dataset: pd.DataFrame) -> Dict:
        """Get summary statistics for the dataset."""
        positives = dataset[dataset['is_edited'] == 1]
        negatives = dataset[dataset['is_edited'] == 0]

        return {
            'n_total': len(dataset),
            'n_positives': len(positives),
            'n_negatives': len(negatives),
            'ratio': len(negatives) / len(positives) if len(positives) > 0 else 0,
            'positive_coverage_mean': positives['coverage'].mean(),
            'positive_coverage_median': positives['coverage'].median(),
            'negative_coverage_mean': negatives['coverage'].mean(),
            'negative_coverage_median': negatives['coverage'].median(),
            'positive_editing_rate_mean': positives['editing_rate'].mean(),
            'n_plus_strand': len(dataset[dataset['strand'] == '+']),
            'n_minus_strand': len(dataset[dataset['strand'] == '-']),
            'sequence_length': len(dataset['sequence'].iloc[0]) if len(dataset) > 0 else 0,
        }

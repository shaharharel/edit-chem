"""
Configuration for ChEMBL data extraction and pair generation.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ChEMBLConfig:
    """
    Configuration for ChEMBL molecule download and pair extraction.

    This class holds all configurable parameters for the full pipeline:
    1. ChEMBL database download
    2. Molecule and bioactivity extraction
    3. Matched molecular pair (MMP) generation
    """

    # === ChEMBL Download Settings ===
    chembl_version: Optional[str] = None  # None = latest
    n_molecules: int = 50000  # Number of molecules to download
    activity_types: List[str] = field(default_factory=lambda: ['IC50', 'Ki', 'EC50', 'Kd'])
    min_assays_per_molecule: int = 1  # Minimum bioactivity measurements

    # === MMP Extraction Settings ===
    max_cuts: int = 1  # Maximum cuts for MMP (1 = single cut, simpler edits)
    max_mw_delta: float = 200.0  # Maximum molecular weight difference
    min_similarity: float = 0.4  # Minimum Tanimoto similarity

    # === Property Filtering ===
    property_filter: Optional[List[str]] = None  # Only extract these properties (None = all)
    exclude_computed: bool = False  # Exclude computed properties (mw, logp, etc.)

    # === Checkpointing ===
    checkpoint_every: int = 1000  # Save checkpoint every N cores
    resume_from_checkpoint: bool = True  # Resume from checkpoint if exists

    # === Output Paths ===
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @property
    def chembl_dir(self) -> Path:
        """Directory for ChEMBL downloads."""
        return self.data_dir / "chembl"

    @property
    def db_dir(self) -> Path:
        """Directory for ChEMBL SQLite database."""
        return self.chembl_dir / "db"

    @property
    def molecules_file(self) -> Path:
        """Path to molecules CSV."""
        return self.chembl_dir / f"molecules_{self.n_molecules}.csv"

    @property
    def bioactivity_file(self) -> Path:
        """Path to bioactivity CSV."""
        return self.chembl_dir / f"bioactivity_{self.n_molecules}.csv"

    @property
    def pairs_dir(self) -> Path:
        """Directory for generated pairs."""
        return self.data_dir / "pairs"

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for checkpoints."""
        return self.pairs_dir / "checkpoints"

    @property
    def pairs_output(self) -> Path:
        """Default output file for pairs."""
        return self.pairs_dir / f"pairs_n{self.n_molecules}_cuts{self.max_cuts}.csv"

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        # Create directories
        self.chembl_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.pairs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

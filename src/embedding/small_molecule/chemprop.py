"""
ChemProp (D-MPNN graph neural network) molecule embeddings.

Uses the official ChemProp v2 library to generate molecular representations.
Compatible with ChemProp v2.x API.
"""

import numpy as np
from typing import Union, List, Optional
from .base import MoleculeEmbedder


class ChemPropEmbedder(MoleculeEmbedder):
    """
    ChemProp D-MPNN molecule embedder (v2.x compatible).

    Uses ChemProp v2 library for molecular representations. Without a trained model,
    uses Morgan fingerprints from ChemProp's featurizer.

    Installation:
        pip install chemprop

    Args:
        model_path: Path to trained ChemProp v2 model checkpoint (optional)
                   If None, uses featurization based on `featurizer_type`
        batch_size: Batch size for encoding
        featurizer_type: Type of featurizer to use when no model provided:
                        - 'morgan': Morgan binary fingerprints (default, fast)
                        - 'rdkit2d': RDKit 2D descriptors (217 features, interpretable)
                        - 'graph': D-MPNN graph neural network (300-dim, learned structure)
        morgan_radius: Morgan fingerprint radius (ChemProp default: 2)
        morgan_length: Morgan fingerprint length (ChemProp default: 2048)
        include_chirality: Include chirality in Morgan fingerprints (default: True)

    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 50,
        featurizer_type: str = 'morgan',
        morgan_radius: int = 2,
        morgan_length: int = 2048,  # ChemProp default
        include_chirality: bool = True,
        device: str = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.featurizer_type = featurizer_type
        self.morgan_radius = morgan_radius
        self.morgan_length = morgan_length
        self.include_chirality = include_chirality
        self.device = device

        # Try to import ChemProp v2
        try:
            import chemprop
            self.chemprop = chemprop
            self._chemprop_available = True
        except ImportError:
            raise ImportError(
                "ChemProp is not installed. Install with: pip install chemprop"
            )

        # Initialize
        if model_path:
            self._init_with_model()
        else:
            self._init_featurization_only()

    def _init_with_model(self):
        """Initialize with a trained ChemProp v2 model."""
        from chemprop.models import load_model

        print(f"Loading ChemProp v2 model from {self.model_path}")
        self.model = load_model(self.model_path)
        self.model.eval()

        # Get embedding dimension from model
        # In v2, we need to inspect the model architecture
        self._embedding_dim = 300  # Default, will be updated after first encode
        self._use_model = True
        self.featurizer = None

    def _init_featurization_only(self):
        """Initialize featurization without trained model."""
        self.model = None
        self._use_model = False

        if self.featurizer_type == 'morgan':
            from chemprop.featurizers import MorganBinaryFeaturizer

            print(f"Using ChemProp v2 Morgan fingerprints (r={self.morgan_radius}, "
                  f"len={self.morgan_length}, chirality={self.include_chirality})")

            self.featurizer = MorganBinaryFeaturizer(
                radius=self.morgan_radius,
                length=self.morgan_length,
                include_chirality=self.include_chirality
            )
            self._embedding_dim = self.morgan_length

        elif self.featurizer_type == 'rdkit2d':
            from chemprop.featurizers import RDKit2DFeaturizer

            print("Using ChemProp v2 RDKit 2D descriptors (217 features)")

            self.featurizer = RDKit2DFeaturizer()
            self._embedding_dim = 217  # RDKit2D has 217 features

        elif self.featurizer_type == 'graph':
            from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
            from chemprop.nn import BondMessagePassing, MeanAggregation
            import torch

            print(f"Using ChemProp v2 D-MPNN graph embeddings (300-dim) on {self.device}")

            # Create graph featurizer
            self.featurizer = SimpleMoleculeMolGraphFeaturizer()
            d_v, d_e = self.featurizer.shape  # (72, 14)

            # Create message passing network (randomly initialized)
            self.message_passing = BondMessagePassing(
                d_v=d_v,
                d_e=d_e,
                d_h=300,  # hidden dimension
                depth=3,   # 3 message passing layers
                dropout=0.0
            )
            self.aggregation = MeanAggregation()

            # Move to GPU if available
            self.message_passing = self.message_passing.to(self.device)
            self.aggregation = self.aggregation.to(self.device)

            # Set to eval mode (frozen - no training)
            self.message_passing.eval()
            for param in self.message_passing.parameters():
                param.requires_grad = False

            self._embedding_dim = 300

        else:
            raise ValueError(
                f"Unknown featurizer_type: {self.featurizer_type}. "
                "Supported: 'morgan', 'rdkit2d', 'graph'"
            )

    def encode(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Encode molecule(s) to embedding vectors using ChemProp v2.

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Embedding vector(s) as numpy array
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
            return_single = True
        else:
            # Convert numpy array to list if needed
            if isinstance(smiles, np.ndarray):
                smiles_list = smiles.tolist()
            else:
                smiles_list = smiles
            return_single = False

        # Encode using appropriate method
        if self._use_model:
            embeddings = self._encode_with_model(smiles_list)
        else:
            embeddings = self._encode_with_featurization(smiles_list)

        if return_single:
            return embeddings[0]
        else:
            return embeddings

    def _encode_with_model(self, smiles_list: List[str]) -> np.ndarray:
        """Encode using trained ChemProp v2 model to extract learned representations."""
        import torch
        from chemprop.data import MoleculeDatapoint

        # Create datapoints
        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles_list]

        # Extract embeddings
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(datapoints), self.batch_size):
                batch_dps = datapoints[i:i + self.batch_size]

                # Get model predictions (this uses the encoder internally)
                # For embeddings, we'd need to access the encoder directly
                # This depends on the specific model architecture
                try:
                    batch_embs = self.model.encode([dp.mol for dp in batch_dps])
                    embeddings.append(batch_embs.cpu().numpy())
                except AttributeError:
                    # Fallback: use model forward and extract hidden layer
                    raise NotImplementedError(
                        "Model-based encoding requires a model with encode() method. "
                        "For now, use featurization mode (no model_path)."
                    )

        return np.vstack(embeddings)

    def _encode_with_featurization(self, smiles_list: List[str]) -> np.ndarray:
        """
        Encode using ChemProp v2 featurization (Morgan, RDKit2D, or graph-based).
        """
        from chemprop.data import MoleculeDatapoint

        # Create datapoints
        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles_list]

        # Graph-based encoding requires special handling
        if self.featurizer_type == 'graph':
            return self._encode_with_graph_mpnn(datapoints)

        # Standard featurization (Morgan or RDKit2D)
        embeddings = []
        for dp in datapoints:
            try:
                # Use the featurizer on the RDKit mol object
                feat = self.featurizer(dp.mol)
                embeddings.append(feat)
            except Exception as e:
                # If featurization fails, use zero vector
                print(f"Warning: Could not featurize SMILES {dp.smiles}, using zeros: {e}")
                embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))

        return np.array(embeddings, dtype=np.float32)

    def _encode_with_graph_mpnn(self, datapoints: List) -> np.ndarray:
        """
        Encode using D-MPNN graph neural network.

        Uses message passing on molecular graphs to get learned representations.
        """
        import torch
        from chemprop.data import BatchMolGraph

        # Build molecular graphs
        mol_graphs = []
        for dp in datapoints:
            try:
                mol_graph = self.featurizer(dp.mol)
                mol_graphs.append(mol_graph)
            except Exception as e:
                print(f"Warning: Could not build graph for {dp.smiles}: {e}")
                # Use None as placeholder
                mol_graphs.append(None)

        # Process in batches
        embeddings = []
        self.message_passing.eval()

        with torch.no_grad():
            for i in range(0, len(mol_graphs), self.batch_size):
                batch_graphs = mol_graphs[i:i + self.batch_size]

                # Filter out None graphs
                valid_graphs = [g for g in batch_graphs if g is not None]
                if not valid_graphs:
                    # All graphs failed, use zeros
                    embeddings.extend([
                        np.zeros(self._embedding_dim, dtype=np.float32)
                        for _ in batch_graphs
                    ])
                    continue

                # Batch graphs and move to device
                try:
                    batch_graph = BatchMolGraph(valid_graphs)
                    # Use official ChemProp v2 .to() method to move graph to device
                    batch_graph = batch_graph.to(self.device)
                except Exception as e:
                    print(f"Error creating BatchMolGraph: {e}")
                    # If batching fails, use zeros for all
                    embeddings.extend([
                        np.zeros(self._embedding_dim, dtype=np.float32)
                        for _ in batch_graphs
                    ])
                    continue

                # Forward through message passing
                h = self.message_passing(batch_graph)

                # Aggregate to molecule-level embeddings
                mol_embeddings = self.aggregation(h, batch_graph.batch)

                # Convert to numpy
                batch_embeddings = mol_embeddings.cpu().numpy()

                # Handle failed graphs
                valid_idx = 0
                for g in batch_graphs:
                    if g is None:
                        embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))
                    else:
                        embeddings.append(batch_embeddings[valid_idx])
                        valid_idx += 1

        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this embedding method."""
        if self._use_model:
            return f"chemprop_v2_model_{self._embedding_dim}"
        elif self.featurizer_type == 'morgan':
            chiral = "_chiral" if self.include_chirality else ""
            return f"chemprop_v2_morgan_r{self.morgan_radius}_l{self.morgan_length}{chiral}"
        elif self.featurizer_type == 'rdkit2d':
            return "chemprop_v2_rdkit2d_217"
        elif self.featurizer_type == 'graph':
            return "chemprop_v2_dmpnn_300"
        else:
            return f"chemprop_v2_{self.featurizer_type}_{self._embedding_dim}"

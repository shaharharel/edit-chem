"""
ChemProp (graph neural network) molecule embeddings.

NOTE: ChemProp v2.x has a different API. This is a simplified wrapper
that uses RDKit graph descriptors as a fallback.

For true ChemProp embeddings, use fingerprints or ChemBERTa instead.
Future versions will integrate with ChemProp v2 properly.
"""

import numpy as np
from typing import Union, List, Optional
from .base import MoleculeEmbedder


class ChemPropEmbedder(MoleculeEmbedder):
    """
    Graph-based molecule embedder using RDKit descriptors.

    NOTE: This is a simplified version that doesn't require ChemProp.
    For actual ChemProp D-MPNN embeddings, please use ChemBERTa or
    wait for full ChemProp v2 integration.

    Args:
        hidden_size: Output dimension (default: 300)
    """

    def __init__(self, hidden_size: int = 300):
        self.hidden_size = hidden_size
        print("WARNING: ChemPropEmbedder currently uses RDKit graph descriptors.")
        print("For better embeddings, use FingerprintEmbedder or ChemBERTaEmbedder.")

    def encode(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Encode molecule(s) to graph-based feature vectors.

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Feature vector(s) as numpy array
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
            return_single = True
        else:
            smiles_list = smiles
            return_single = False

        # Extract graph features using RDKit
        embeddings = self._extract_graph_features(smiles_list)

        if return_single:
            return embeddings[0]
        else:
            return embeddings

    def _extract_graph_features(self, smiles_list: List[str]) -> np.ndarray:
        """
        Extract graph-based features using RDKit descriptors.

        This is a fallback implementation. For true graph neural network
        embeddings, use ChemBERTa or wait for ChemProp v2 integration.
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors, GraphDescriptors, Lipinski

        features = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                features.append(np.zeros(self.hidden_size, dtype=np.float32))
                continue

            # Extract comprehensive graph descriptors
            try:
                feat = np.array([
                    # Basic graph properties
                    mol.GetNumAtoms(),
                    mol.GetNumBonds(),
                    mol.GetNumHeavyAtoms(),

                    # Molecular weight and related
                    Descriptors.MolWt(mol),
                    Descriptors.ExactMolWt(mol),

                    # H-bonding
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),

                    # Polarity
                    Descriptors.TPSA(mol),
                    Descriptors.LabuteASA(mol),

                    # Flexibility
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAliphaticRings(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumSaturatedRings(mol),

                    # Complexity
                    GraphDescriptors.BertzCT(mol),
                    GraphDescriptors.Chi0(mol),
                    GraphDescriptors.Chi1(mol),
                    GraphDescriptors.Chi0n(mol),
                    GraphDescriptors.Chi1n(mol),

                    # Ring info
                    Lipinski.RingCount(mol),
                    Lipinski.NumAliphaticRings(mol),
                    Lipinski.NumAromaticRings(mol),

                    # Additional
                    Descriptors.NumValenceElectrons(mol),
                    Descriptors.NumRadicalElectrons(mol),
                    Descriptors.FractionCSP3(mol),

                ], dtype=np.float32)
            except:
                # Fallback to zeros if descriptor calculation fails
                feat = np.zeros(27, dtype=np.float32)

            # Pad or truncate to hidden_size
            if len(feat) < self.hidden_size:
                feat = np.pad(feat, (0, self.hidden_size - len(feat)))
            else:
                feat = feat[:self.hidden_size]

            features.append(feat)

        return np.array(features)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self.hidden_size

    @property
    def name(self) -> str:
        """Return the name of this embedding method."""
        return f"graph_descriptors_{self.hidden_size}"


# Convenience constructor
def chemprop_embedder(hidden_size: int = 300) -> ChemPropEmbedder:
    """
    Create graph descriptor embedder.

    NOTE: This uses RDKit descriptors, not actual ChemProp D-MPNN.
    For better embeddings, use FingerprintEmbedder or ChemBERTaEmbedder.

    Args:
        hidden_size: Output dimension

    Returns:
        ChemPropEmbedder instance
    """
    return ChemPropEmbedder(hidden_size=hidden_size)

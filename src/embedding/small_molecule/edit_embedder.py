"""
Edit embedding via difference of molecule embeddings.

Computes edit embeddings as:
    edit_embedding = embedding(product) - embedding(reactant)

This works with any molecule embedder (fingerprints, ChemBERTa, ChemProp, etc.)
"""

import numpy as np
from typing import Union, List, Tuple
from .base import MoleculeEmbedder


class EditEmbedder:
    """
    Edit embedder using difference of molecule embeddings.

    Computes: edit = product - reactant

    This approach:
    - Works with ANY molecule embedder (fingerprints, transformers, GNNs)
    - Preserves the same dimensionality as the molecule embeddings
    - Captures the change in molecular properties
    - Can be used directly for ML without additional processing

    Args:
        molecule_embedder: Any MoleculeEmbedder instance
                          (FingerprintEmbedder, ChemBERTaEmbedder, etc.)

    Example:
        >>> from src.embedding import FingerprintEmbedder, EditEmbedder
        >>>
        >>> # Option 1: Morgan fingerprints
        >>> mol_emb = FingerprintEmbedder(fp_type='morgan', radius=2, n_bits=2048)
        >>> edit_emb = EditEmbedder(mol_emb)
        >>>
        >>> # Compute edit embedding from SMILES pair
        >>> edit_vec = edit_emb.encode_from_smiles('CCO', 'CC(=O)O')  # ethanol -> acetic acid
        >>> print(edit_vec.shape)  # (2048,)
        >>>
        >>> # Option 2: ChemBERTa
        >>> from src.embedding import ChemBERTaEmbedder
        >>> mol_emb = ChemBERTaEmbedder(model_name='chemberta')
        >>> edit_emb = EditEmbedder(mol_emb)
        >>> edit_vec = edit_emb.encode_from_smiles('CCO', 'CC(=O)O')
        >>> print(edit_vec.shape)  # (768,) - ChemBERTa hidden size
    """

    def __init__(self, molecule_embedder: MoleculeEmbedder):
        self.molecule_embedder = molecule_embedder

    def encode_from_smiles(
        self,
        reactant: Union[str, List[str]],
        product: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Encode edit(s) from reactant and product SMILES.

        Args:
            reactant: Reactant SMILES (single or list)
            product: Product SMILES (single or list)

        Returns:
            Edit embedding(s) as numpy array
            - Single pair: shape (embedding_dim,)
            - Multiple pairs: shape (n_edits, embedding_dim)
        """
        # Handle single vs batch
        if isinstance(reactant, str):
            assert isinstance(product, str), "Reactant and product must both be strings or lists"
            reactants = [reactant]
            products = [product]
            return_single = True
        else:
            assert len(reactant) == len(product), "Reactant and product lists must have same length"
            reactants = reactant
            products = product
            return_single = False

        # Encode molecules
        reactant_emb = self.molecule_embedder.encode(reactants)
        product_emb = self.molecule_embedder.encode(products)

        # Compute difference
        edit_emb = product_emb - reactant_emb

        if return_single:
            return edit_emb[0]
        else:
            return edit_emb

    def encode_from_edit_smiles(
        self,
        edit_smiles: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Encode edit(s) from reaction SMILES format.

        Args:
            edit_smiles: Reaction SMILES "reactant>>product" (single or list)

        Returns:
            Edit embedding(s) as numpy array

        Example:
            >>> edit_emb.encode_from_edit_smiles("CCO>>CC(=O)O")
            >>> edit_emb.encode_from_edit_smiles(["C>>CC", "F>>Cl"])
        """
        if isinstance(edit_smiles, str):
            edit_smiles = [edit_smiles]
            return_single = True
        else:
            return_single = False

        # Parse reaction SMILES
        reactants = []
        products = []
        for edit in edit_smiles:
            if '>>' not in edit:
                raise ValueError(f"Invalid edit SMILES (missing '>>'): {edit}")

            react, prod = edit.split('>>')
            reactants.append(react)
            products.append(prod)

        # Encode
        edit_emb = self.encode_from_smiles(reactants, products)

        if return_single:
            return edit_emb[0] if edit_emb.ndim > 1 else edit_emb
        else:
            return edit_emb

    def encode_from_pair_df(self, pairs_df) -> np.ndarray:
        """
        Encode edits from pairs DataFrame.

        Args:
            pairs_df: DataFrame with 'mol_a' and 'mol_b' columns

        Returns:
            Edit embeddings array of shape (n_pairs, embedding_dim)

        Example:
            >>> import pandas as pd
            >>> pairs = pd.DataFrame({
            ...     'mol_a': ['CCO', 'c1ccccc1'],
            ...     'mol_b': ['CC(=O)O', 'c1ccncc1']
            ... })
            >>> edit_embeddings = edit_emb.encode_from_pair_df(pairs)
        """
        return self.encode_from_smiles(
            pairs_df['mol_a'].tolist(),
            pairs_df['mol_b'].tolist()
        )

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of edit embeddings."""
        return self.molecule_embedder.embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this edit embedding method."""
        return f"edit_diff_{self.molecule_embedder.name}"


# Convenience constructors
def edit_embedder_morgan(radius: int = 2, n_bits: int = 2048) -> EditEmbedder:
    """
    Create edit embedder using Morgan fingerprint differences.

    This is the recommended baseline approach.

    Args:
        radius: Morgan fingerprint radius (default: 2 for ECFP4)
        n_bits: Number of bits (default: 2048)

    Returns:
        EditEmbedder instance

    Example:
        >>> edit_emb = edit_embedder_morgan(radius=2, n_bits=2048)
        >>> edit_vec = edit_emb.encode_from_edit_smiles("C>>CC")  # methylation
    """
    from .fingerprints import FingerprintEmbedder
    mol_emb = FingerprintEmbedder(fp_type='morgan', radius=radius, n_bits=n_bits)
    return EditEmbedder(mol_emb)


def edit_embedder_chemberta(model_name: str = 'chemberta', pooling: str = 'mean') -> EditEmbedder:
    """
    Create edit embedder using ChemBERTa embedding differences.

    Args:
        model_name: ChemBERTa model variant
        pooling: Pooling strategy

    Returns:
        EditEmbedder instance

    Example:
        >>> edit_emb = edit_embedder_chemberta()
        >>> edit_vec = edit_emb.encode_from_edit_smiles("C>>CC")
    """
    try:
        from .chemberta import ChemBERTaEmbedder
        mol_emb = ChemBERTaEmbedder(model_name=model_name, pooling=pooling)
        return EditEmbedder(mol_emb)
    except ImportError as e:
        raise ImportError(
            "ChemBERTa requires torch and transformers. "
            "Install with: pip install torch transformers"
        ) from e

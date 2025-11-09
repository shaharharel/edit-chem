"""
Edit embedding utilities for molecular transformations.

This module provides functions to embed chemical edits for use in
property prediction models: F(molecule, edit) -> property_change
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, Tuple


def embed_molecule(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Embed a molecule using Morgan fingerprints.

    Args:
        smiles: SMILES string
        radius: Morgan fingerprint radius (default: 2)
        n_bits: Fingerprint size (default: 2048)

    Returns:
        Fingerprint as numpy array, or None if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=float)


def embed_edit(edit_from: str, edit_to: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Embed a chemical edit/transformation as difference fingerprint.

    Args:
        edit_from: SMILES of what was removed (e.g., "C[H]" for -CH3)
        edit_to: SMILES of what was added (e.g., "CC[H]" for -CH2CH3)
        radius: Morgan fingerprint radius (default: 2)
        n_bits: Fingerprint size (default: 2048)

    Returns:
        Edit embedding (difference fingerprint), or None if parsing fails

    Example:
        >>> edit_emb = embed_edit("C[H]", "CC[H]")  # -CH3 to -CH2CH3
        >>> edit_emb.shape
        (2048,)
    """
    # Parse SMILES
    mol_from = Chem.MolFromSmiles(edit_from)
    mol_to = Chem.MolFromSmiles(edit_to)

    if mol_from is None or mol_to is None:
        return None

    # Generate fingerprints
    fp_from = AllChem.GetMorganFingerprintAsBitVect(mol_from, radius, nBits=n_bits)
    fp_to = AllChem.GetMorganFingerprintAsBitVect(mol_to, radius, nBits=n_bits)

    # Difference = edit embedding
    edit_emb = np.array(fp_to, dtype=float) - np.array(fp_from, dtype=float)

    return edit_emb


def embed_molecule_and_edit(
    mol_smiles: str,
    edit_from: str,
    edit_to: str,
    radius: int = 2,
    n_bits: int = 2048,
    combine: str = 'concat'
) -> Optional[np.ndarray]:
    """
    Create combined embedding for F(molecule, edit) -> property_change.

    This is the main function for your neural network input.

    Args:
        mol_smiles: SMILES of the molecule
        edit_from: SMILES of what was removed
        edit_to: SMILES of what was added
        radius: Morgan fingerprint radius (default: 2)
        n_bits: Fingerprint size (default: 2048)
        combine: How to combine ('concat', 'product', 'sum')

    Returns:
        Combined embedding, shape depends on `combine`:
            - 'concat': (2 * n_bits,) - [mol_features | edit_features]
            - 'product': (n_bits,) - element-wise product
            - 'sum': (n_bits,) - element-wise sum

    Example:
        >>> # Benzene + (CH3 -> CH2CH3) edit
        >>> emb = embed_molecule_and_edit("c1ccccc1", "C[H]", "CC[H]")
        >>> emb.shape
        (4096,)  # 2048 mol + 2048 edit
    """
    # Embed molecule
    mol_emb = embed_molecule(mol_smiles, radius, n_bits)
    if mol_emb is None:
        return None

    # Embed edit
    edit_emb = embed_edit(edit_from, edit_to, radius, n_bits)
    if edit_emb is None:
        return None

    # Combine
    if combine == 'concat':
        return np.concatenate([mol_emb, edit_emb])
    elif combine == 'product':
        return mol_emb * edit_emb
    elif combine == 'sum':
        return mol_emb + edit_emb
    else:
        raise ValueError(f"Unknown combine method: {combine}")


def batch_embed_pairs(pairs_df, radius: int = 2, n_bits: int = 2048, combine: str = 'concat'):
    """
    Batch embed all pairs from a DataFrame.

    Args:
        pairs_df: DataFrame with columns: mol_a, edit_from, edit_to, delta
        radius: Morgan fingerprint radius
        n_bits: Fingerprint size
        combine: How to combine molecule and edit embeddings

    Returns:
        X: Feature matrix (n_pairs, embedding_dim)
        y: Target values (n_pairs,) - property deltas

    Example:
        >>> import pandas as pd
        >>> pairs_df = pd.read_csv('chembl_pairs_long.csv')
        >>> X, y = batch_embed_pairs(pairs_df)
        >>> X.shape, y.shape
        ((10000, 4096), (10000,))
    """
    embeddings = []
    targets = []

    for idx, row in pairs_df.iterrows():
        emb = embed_molecule_and_edit(
            row['mol_a'],
            row['edit_from'],
            row['edit_to'],
            radius=radius,
            n_bits=n_bits,
            combine=combine
        )

        if emb is not None:
            embeddings.append(emb)
            targets.append(row['delta'])

    X = np.array(embeddings)
    y = np.array(targets)

    return X, y


if __name__ == '__main__':
    # Example usage
    print("Edit Embedding Example")
    print("=" * 70)

    # Example transformation: methyl to ethyl
    mol = "c1ccccc1"  # benzene
    edit_from = "C[H]"  # -CH3
    edit_to = "CC[H]"   # -CH2CH3

    print(f"Molecule: {mol}")
    print(f"Edit: {edit_from} -> {edit_to}")
    print()

    # Embed
    emb = embed_molecule_and_edit(mol, edit_from, edit_to, combine='concat')

    print(f"Embedding shape: {emb.shape}")
    print(f"Molecule part sum: {emb[:2048].sum():.2f}")
    print(f"Edit part sum: {emb[2048:].sum():.2f}")
    print()

    print("This embedding can be fed to a neural network:")
    print("  Input: F(molecule, edit)")
    print("  Output: predicted property change (delta)")

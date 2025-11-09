"""Small molecule specific edit implementation."""

from typing import Optional, List
from rdkit import Chem
from src.core.edit import Edit
from src.utils.chemistry import smiles_to_mol, mol_to_smiles, apply_transformation


class SmallMoleculeEdit(Edit):
    """
    Edit specific to small molecules.

    Extends base Edit with RDKit-specific functionality for
    applying transformations to molecules.
    """

    def apply(self, smiles: str) -> List[str]:
        """
        Apply this edit to a molecule.

        Args:
            smiles: SMILES string of the molecule to edit

        Returns:
            List of SMILES strings for products (may be multiple if pattern matches multiple times)
        """
        mol = smiles_to_mol(smiles)
        if mol is None:
            return []

        products = apply_transformation(mol, self.from_smarts, self.to_smarts)

        # Convert back to SMILES
        product_smiles = []
        for prod in products:
            smi = mol_to_smiles(prod)
            if smi:
                product_smiles.append(smi)

        return product_smiles

    def is_applicable(self, smiles: str) -> bool:
        """
        Check if this edit can be applied to a molecule.

        Args:
            smiles: SMILES string

        Returns:
            True if the from_smarts pattern is found in the molecule
        """
        mol = smiles_to_mol(smiles)
        if mol is None:
            return False

        pattern = Chem.MolFromSmarts(self.from_smarts)
        if pattern is None:
            return False

        return mol.HasSubstructMatch(pattern)

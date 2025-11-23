"""
Vocabulary of common molecular edits with canonical names.

This maps edit_smiles (reaction SMILES) to their common names
in medicinal chemistry.
"""

# Canonical names for common edits
EDIT_NAMES = {
    # Alkylation
    "[H]>>C": "methylation",
    "[H]>>CC": "ethylation",
    "[H]>>C(C)C": "isopropylation",
    "[H]>>C(C)(C)C": "tert-butylation",

    # Homologation (chain extension)
    "C>>CC": "homologation_C1_to_C2",
    "CC>>CCC": "homologation_C2_to_C3",
    "C>>C(C)C": "branching_methyl_to_isopropyl",

    # Halogenation
    "[H]>>F": "fluorination",
    "[H]>>Cl": "chlorination",
    "[H]>>Br": "bromination",
    "[H]>>I": "iodination",

    # Halogen exchange
    "F>>Cl": "F_to_Cl_exchange",
    "Cl>>Br": "Cl_to_Br_exchange",
    "Br>>I": "Br_to_I_exchange",
    "F>>Br": "F_to_Br_exchange",
    "Cl>>F": "Cl_to_F_exchange",

    # Heteroatom swaps
    "O>>N": "OH_to_NH2",
    "N>>O": "NH2_to_OH",
    "O>>S": "O_to_S_bioisostere",
    "S>>O": "S_to_O_bioisostere",
    "C>>N": "aza_substitution",
    "N>>C": "deaza_substitution",

    # Functional group interconversions
    "C>>O": "hydroxylation",
    "O>>[H]": "dehydroxylation",
    "C>>C(=O)": "carbonyl_formation",
    "C(=O)>>C": "carbonyl_reduction",

    # Common medicinal chemistry modifications
    "C(F)(F)F>>C": "CF3_removal",
    "C>>C(F)(F)F": "CF3_addition",
    "c1ccccc1>>c1ccncc1": "phenyl_to_pyridyl",
    "c1ccncc1>>c1ccccc1": "pyridyl_to_phenyl",

    # Dehalogenation
    "F>>[H]": "defluorination",
    "Cl>>[H]": "dechlorination",
    "Br>>[H]": "debromination",
    "I>>[H]": "deiodination",
}


def get_edit_name(edit_smiles: str) -> str:
    """
    Get the canonical name for an edit.

    Args:
        edit_smiles: Reaction SMILES (e.g., "C>>CC")

    Returns:
        Canonical name if known, else the edit_smiles itself

    """
    return EDIT_NAMES.get(edit_smiles, edit_smiles)


def is_known_edit(edit_smiles: str) -> bool:
    """Check if edit has a canonical name."""
    return edit_smiles in EDIT_NAMES


def get_all_known_edits():
    """Get all known edits and their names."""
    return EDIT_NAMES.copy()


def categorize_edit(edit_smiles: str) -> str:
    """
    Categorize edit into broad classes.

    Returns:
        Category: alkylation, halogenation, heteroatom_swap, etc.
    """
    name = get_edit_name(edit_smiles)

    if "methylation" in name or "ethylation" in name:
        return "alkylation"
    elif "homologation" in name or "branching" in name:
        return "chain_extension"
    elif "fluorination" in name or "chlorination" in name:
        return "halogenation"
    elif "defluorination" in name or "dechlorination" in name:
        return "dehalogenation"
    elif "exchange" in name:
        return "halogen_exchange"
    elif "bioisostere" in name or "to_pyridyl" in name:
        return "bioisosteric_replacement"
    elif "aza" in name or "deaza" in name:
        return "heteroatom_substitution"
    elif "OH_to" in name or "NH2_to" in name:
        return "functional_group_interconversion"
    else:
        return "other"


if __name__ == '__main__':
    # Demo
    test_edits = [
        "C>>CC",
        "F>>Cl",
        "[H]>>C",
        "O>>N",
        "unknown>>transformation"
    ]

    print("Edit Vocabulary Demo")
    print("=" * 60)

    for edit in test_edits:
        name = get_edit_name(edit)
        category = categorize_edit(edit)
        known = "✓" if is_known_edit(edit) else "✗"

        print(f"{known} {edit:30} → {name:30} [{category}]")

    print()
    print(f"Total known edits: {len(EDIT_NAMES)}")

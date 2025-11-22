"""
Visualization functions for molecular edits and transformations.

This module provides visualization utilities for displaying molecules,
their edits, and the resulting transformed molecules.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import io
import numpy as np
from typing import List, Tuple, Optional


def visualize_molecule_edits(
    source_smiles: str,
    edit_names: List[str],
    result_smiles_list: List[str],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 6),
    mol_size: Tuple[int, int] = (300, 300),
    arrow_color: str = '#2E86AB',
    arrow_width: float = 2.0,
    font_size: int = 10,
    title_font_size: int = 14
) -> plt.Figure:
    """
    Visualize a source molecule with arrows pointing to result molecules after edits.

    Args:
        source_smiles: SMILES string of the source molecule
        edit_names: List of edit names (labels for arrows)
        result_smiles_list: List of SMILES strings for result molecules
        title: Optional title for the figure
        figsize: Figure size (width, height)
        mol_size: Size for individual molecule images
        arrow_color: Color for the arrows
        arrow_width: Width of the arrows
        font_size: Font size for edit labels
        title_font_size: Font size for the main title

    Returns:
        matplotlib.figure.Figure: The generated figure

    """
    # Validate inputs
    if len(edit_names) != len(result_smiles_list):
        raise ValueError(f"Number of edit names ({len(edit_names)}) must match "
                        f"number of result molecules ({len(result_smiles_list)})")

    n_edits = len(edit_names)

    # Convert SMILES to RDKit molecules
    source_mol = Chem.MolFromSmiles(source_smiles)
    if source_mol is None:
        raise ValueError(f"Invalid source SMILES: {source_smiles}")

    result_mols = []
    for smiles in result_smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid result SMILES: {smiles}")
        result_mols.append(mol)

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=title_font_size, fontweight='bold', y=0.98)

    # Generate molecule images
    source_img = Draw.MolToImage(source_mol, size=mol_size)
    result_imgs = [Draw.MolToImage(mol, size=mol_size) for mol in result_mols]

    # Calculate positions
    # Source molecule on the left
    source_x = 15
    source_y = 50

    # Result molecules distributed on the right
    result_x = 75
    if n_edits == 1:
        result_y_positions = [50]
    else:
        # Distribute vertically
        spacing = 80 / (n_edits + 1)
        result_y_positions = [10 + spacing * (i + 1) for i in range(n_edits)]

    # Place source molecule
    _add_image_to_axis(ax, source_img, source_x, source_y, width=15)

    # Place result molecules and arrows
    for i, (result_img, edit_name, result_y) in enumerate(
        zip(result_imgs, edit_names, result_y_positions)
    ):
        # Place result molecule
        _add_image_to_axis(ax, result_img, result_x, result_y, width=15)

        # Draw arrow from source to result
        arrow = FancyArrowPatch(
            (source_x + 8, source_y),
            (result_x - 8, result_y),
            arrowstyle='-|>',
            mutation_scale=25,
            linewidth=arrow_width,
            color=arrow_color,
            alpha=0.7,
            zorder=1
        )
        ax.add_patch(arrow)

        # Add edit label on arrow
        mid_x = (source_x + 8 + result_x - 8) / 2
        mid_y = (source_y + result_y) / 2

        # Format edit name for display (replace underscores with spaces, capitalize)
        display_name = edit_name.replace('_', ' ').title()

        # Add white background box for text
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor=arrow_color, linewidth=1.5, alpha=0.9)
        ax.text(mid_x, mid_y, display_name, ha='center', va='center',
                fontsize=font_size, fontweight='bold', bbox=bbox_props, zorder=2)

    plt.tight_layout()
    return fig


def _add_image_to_axis(ax, img, x, y, width=10):
    """
    Helper function to add a PIL image to a matplotlib axis at specific coordinates.

    Args:
        ax: matplotlib axis
        img: PIL Image
        x: x-coordinate (center)
        y: y-coordinate (center)
        width: width in axis coordinates
    """
    # Convert PIL image to array
    img_array = np.array(img)

    # Calculate height to maintain aspect ratio
    aspect_ratio = img_array.shape[0] / img_array.shape[1]
    height = width * aspect_ratio

    # Calculate extent (left, right, bottom, top)
    extent = [x - width/2, x + width/2, y - height/2, y + height/2]

    # Add image to axis
    ax.imshow(img_array, extent=extent, aspect='auto', zorder=0)


def visualize_edit_distribution(
    edit_counts: dict,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 6),
    color: str = '#A23B72'
) -> plt.Figure:
    """
    Visualize the distribution of edits in the dataset.

    Args:
        edit_counts: Dictionary mapping edit names to counts
        top_n: Number of top edits to display
        figsize: Figure size
        color: Bar color

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Sort and get top N
    sorted_edits = sorted(edit_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    edit_names = [name.replace('_', ' ').title() for name, _ in sorted_edits]
    counts = [count for _, count in sorted_edits]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create horizontal bar chart
    y_pos = np.arange(len(edit_names))
    ax.barh(y_pos, counts, color=color, alpha=0.8)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels(edit_names)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title(f'Top {top_n} Most Frequent Edits', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add count labels
    for i, count in enumerate(counts):
        ax.text(count, i, f'  {count:,}', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def visualize_property_change_distribution(
    deltas: List[float],
    property_name: str,
    edit_name: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    color: str = '#F18F01'
) -> plt.Figure:
    """
    Visualize the distribution of property changes for a specific edit.

    Args:
        deltas: List of property change values
        property_name: Name of the property
        edit_name: Optional name of the edit
        figsize: Figure size
        color: Histogram color

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create histogram
    n, bins, patches = ax.hist(deltas, bins=50, color=color, alpha=0.7, edgecolor='black')

    # Add vertical line at mean
    mean_delta = np.mean(deltas)
    ax.axvline(mean_delta, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_delta:.2f}')

    # Customize
    title = f'Distribution of {property_name.replace("_", " ").title()} Changes'
    if edit_name:
        title += f'\n({edit_name.replace("_", " ").title()})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Δ{property_name}', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_multiple_edits_grid(
    source_smiles_list: List[str],
    edit_names_list: List[List[str]],
    result_smiles_list: List[List[str]],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 18)
) -> plt.Figure:
    """
    Create a grid visualization showing multiple source molecules with their edits.

    Args:
        source_smiles_list: List of source molecule SMILES
        edit_names_list: List of lists of edit names (one list per source molecule)
        result_smiles_list: List of lists of result SMILES (one list per source molecule)
        titles: Optional list of titles for each subplot
        figsize: Overall figure size

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    n_molecules = len(source_smiles_list)

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)

    for i in range(n_molecules):
        # Create subplot for each molecule
        ax = fig.add_subplot(n_molecules, 1, i + 1)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

        # Get data for this molecule
        source_smiles = source_smiles_list[i]
        edit_names = edit_names_list[i]
        result_smiles = result_smiles_list[i]

        # Set subplot title
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12, fontweight='bold', pad=10)

        # Validate
        n_edits = len(edit_names)
        source_mol = Chem.MolFromSmiles(source_smiles)
        result_mols = [Chem.MolFromSmiles(s) for s in result_smiles]

        # Generate images
        source_img = Draw.MolToImage(source_mol, size=(250, 250))
        result_imgs = [Draw.MolToImage(mol, size=(250, 250)) for mol in result_mols]

        # Position source
        source_x = 15
        source_y = 50

        # Position results
        result_x = 75
        if n_edits == 1:
            result_y_positions = [50]
        else:
            spacing = 80 / (n_edits + 1)
            result_y_positions = [10 + spacing * (j + 1) for j in range(n_edits)]

        # Add source
        _add_image_to_axis(ax, source_img, source_x, source_y, width=15)

        # Add results and arrows
        for j, (result_img, edit_name, result_y) in enumerate(
            zip(result_imgs, edit_names, result_y_positions)
        ):
            _add_image_to_axis(ax, result_img, result_x, result_y, width=15)

            # Arrow
            arrow = FancyArrowPatch(
                (source_x + 8, source_y),
                (result_x - 8, result_y),
                arrowstyle='-|>',
                mutation_scale=20,
                linewidth=1.5,
                color='#2E86AB',
                alpha=0.7,
                zorder=1
            )
            ax.add_patch(arrow)

            # Label
            mid_x = (source_x + 8 + result_x - 8) / 2
            mid_y = (source_y + result_y) / 2
            display_name = edit_name.replace('_', ' ').title()
            bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white',
                            edgecolor='#2E86AB', linewidth=1, alpha=0.9)
            ax.text(mid_x, mid_y, display_name, ha='center', va='center',
                   fontsize=8, fontweight='bold', bbox=bbox_props, zorder=2)

    plt.tight_layout()
    return fig

"""
Structural encoders for antibody 3D structures.

These encoders take 3D coordinates of antibody structures and produce
per-residue structural embeddings that can be combined with sequence
embeddings for structure-aware mutation effect prediction.

Available encoders:
- GVPEncoder: Geometric Vector Perceptron GNN
- SE3TransformerEncoder: SE(3)-equivariant transformer
- EquiformerEncoder: Equivariant graph transformer (SOTA)

All encoders follow the StructuralEncoder interface defined in base.py.
"""

try:
    from .gvp import GVPEncoder
except ImportError:
    GVPEncoder = None

try:
    from .se3_transformer import SE3TransformerEncoder
except ImportError:
    SE3TransformerEncoder = None

try:
    from .equiformer import EquiformerEncoder
except ImportError:
    EquiformerEncoder = None

__all__ = [
    'GVPEncoder',
    'SE3TransformerEncoder',
    'EquiformerEncoder',
]

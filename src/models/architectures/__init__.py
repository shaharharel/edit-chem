"""Neural network architectures for multi-task learning."""

from .multi_head import SharedBackbone, TaskHead, MultiTaskHead

__all__ = [
    'SharedBackbone',
    'TaskHead',
    'MultiTaskHead',
]

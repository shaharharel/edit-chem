"""Edit abstraction - represents atomic molecular transformations."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Edit:
    """
    Represents an atomic edit/transformation.

    For small molecules: substitution of one substructure with another
    (e.g., Cl → F, COOH → COOMe)

    Attributes:
        edit_id: Unique identifier for this edit
        from_smarts: SMARTS pattern of the substructure being replaced
        to_smarts: SMARTS pattern of the replacement substructure
        context_smarts: Optional SMARTS for the attachment context
        edit_type: Type of edit (substitution, extension, deletion, etc.)
    """
    edit_id: str
    from_smarts: str
    to_smarts: str
    context_smarts: Optional[str] = None
    edit_type: str = "substitution"

    def __hash__(self):
        """Hash based on the chemical transformation."""
        return hash((self.from_smarts, self.to_smarts, self.context_smarts))

    def __eq__(self, other):
        """Two edits are equal if they represent the same transformation."""
        if not isinstance(other, Edit):
            return False
        return (self.from_smarts == other.from_smarts and
                self.to_smarts == other.to_smarts and
                self.context_smarts == other.context_smarts)

    def reverse(self) -> 'Edit':
        """
        Return the reverse edit.

        If this edit is A → B, the reverse is B → A.
        Useful for bidirectional edit banks.
        """
        return Edit(
            edit_id=f"{self.edit_id}_rev",
            from_smarts=self.to_smarts,
            to_smarts=self.from_smarts,
            context_smarts=self.context_smarts,
            edit_type=self.edit_type
        )

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'edit_id': self.edit_id,
            'from_smarts': self.from_smarts,
            'to_smarts': self.to_smarts,
            'context_smarts': self.context_smarts,
            'edit_type': self.edit_type
        }

    @classmethod
    def from_dict(cls, data):
        """Create Edit from dictionary."""
        return cls(**data)

    def __repr__(self):
        return f"Edit({self.from_smarts} → {self.to_smarts})"

"""
Model factory for RNA experiments.

Creates RNA embedders and models that are compatible with the
existing edit-chem model architectures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict, Optional, List

import torch
import torch.nn as nn

from src.models import PropertyPredictor, EditEffectPredictor
from src.embedding.rna import (
    RNAEmbedder,
    NucleotideEmbedder,
    RNAEditEmbedder,
    StructuredRNAEditEmbedder
)


def create_rna_embedder(
    embedder_type: str,
    trainable: bool = False,
    device: str = 'auto',
    **kwargs
) -> RNAEmbedder:
    """
    Create RNA sequence embedder.

    Args:
        embedder_type: Type of embedder:
            - 'nucleotide': Simple k-mer based (fast, interpretable)
            - 'rnafm': RNA-FM pretrained (requires fm package)
            - 'rnabert': RNABERT pretrained (requires transformers)
        trainable: Whether to allow fine-tuning (only for pretrained)
        device: Device for computation ('auto', 'cuda', 'cpu')
        **kwargs: Additional embedder-specific arguments

    Returns:
        RNAEmbedder instance
    """
    # Auto-detect device
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {device}")

    if embedder_type == 'nucleotide':
        return NucleotideEmbedder(
            include_onehot=kwargs.get('include_onehot', True),
            include_kmers=kwargs.get('include_kmers', True),
            kmer_sizes=kwargs.get('kmer_sizes', [3, 4]),
            include_stats=kwargs.get('include_stats', True),
            include_structure=kwargs.get('include_structure', False),
            include_positional=kwargs.get('include_positional', True),
            num_position_bins=kwargs.get('num_position_bins', 10)
        )

    elif embedder_type == 'rnafm':
        try:
            from src.embedding.rna.rnafm import RNAFMEmbedder
        except ImportError as e:
            raise ImportError(
                "RNA-FM not installed. Install with:\n"
                "pip install git+https://github.com/ml4bio/RNA-FM.git\n"
                f"Original error: {e}"
            )

        return RNAFMEmbedder(
            model_name=kwargs.get('model_name', 'rna_fm_t12'),
            pooling=kwargs.get('pooling', 'mean'),
            device=device,
            batch_size=kwargs.get('batch_size', 32),
            trainable=trainable
        )

    elif embedder_type == 'rnabert':
        try:
            from src.embedding.rna.rnabert import RNABERTEmbedder
        except ImportError as e:
            raise ImportError(
                "transformers not installed. Install with:\n"
                "pip install transformers torch\n"
                f"Original error: {e}"
            )

        return RNABERTEmbedder(
            model_name=kwargs.get('model_name', 'multimolecule/rnabert'),
            pooling=kwargs.get('pooling', 'mean'),
            device=device,
            batch_size=kwargs.get('batch_size', 32),
            trainable=trainable,
            max_length=kwargs.get('max_length', 440)
        )

    else:
        raise ValueError(
            f"Unknown embedder type: {embedder_type}. "
            f"Available: 'nucleotide', 'rnafm', 'rnabert'"
        )


def create_edit_embedder(
    rna_embedder: RNAEmbedder,
    use_local_context: bool = False,
    context_window: int = 25,
    include_edit_features: bool = False
) -> RNAEditEmbedder:
    """
    Create RNA edit embedder.

    Args:
        rna_embedder: Base RNA embedder
        use_local_context: Use local context around edit
        context_window: Size of context window
        include_edit_features: Include additional edit features

    Returns:
        RNAEditEmbedder instance
    """
    return RNAEditEmbedder(
        rna_embedder=rna_embedder,
        use_local_context=use_local_context,
        context_window=context_window,
        include_edit_features=include_edit_features
    )


def create_structured_edit_embedder(
    rnafm_embedder,
    mutation_type_dim: int = 64,
    mutation_effect_dim: int = 256,
    position_dim: int = 64,
    local_context_dim: int = 256,
    attention_context_dim: int = 128,
    fusion_hidden_dims: List[int] = None,
    output_dim: int = 256,
    window_size: int = 10,
    dropout: float = 0.1
) -> StructuredRNAEditEmbedder:
    """
    Create StructuredRNAEditEmbedder for rich edit representations.

    Args:
        rnafm_embedder: RNAFMEmbedder instance (required)
        mutation_type_dim: Dimension of mutation type embedding
        mutation_effect_dim: Dimension of mutation effect embedding
        position_dim: Dimension of position encoding
        local_context_dim: Dimension of local context
        attention_context_dim: Dimension of attention context
        fusion_hidden_dims: Hidden dims for fusion MLP
        output_dim: Final output dimension
        window_size: Local context window size
        dropout: Dropout probability

    Returns:
        StructuredRNAEditEmbedder instance
    """
    if StructuredRNAEditEmbedder is None:
        raise ImportError(
            "StructuredRNAEditEmbedder not available. "
            "Make sure RNA-FM is installed."
        )

    if fusion_hidden_dims is None:
        fusion_hidden_dims = [512, 384]

    return StructuredRNAEditEmbedder(
        rnafm_embedder=rnafm_embedder,
        mutation_type_dim=mutation_type_dim,
        mutation_effect_dim=mutation_effect_dim,
        position_dim=position_dim,
        local_context_dim=local_context_dim,
        attention_context_dim=attention_context_dim,
        fusion_hidden_dims=fusion_hidden_dims,
        output_dim=output_dim,
        window_size=window_size,
        dropout=dropout
    )


class StructuredEditEffectModel(nn.Module):
    """
    End-to-end model for edit effect prediction using StructuredRNAEditEmbedder.

    This model:
    1. Takes sequence A, edit position, from/to nucleotides
    2. Uses StructuredRNAEditEmbedder to create rich edit embedding
    3. Passes through prediction head to output Δ

    The entire model is trainable end-to-end.
    """

    def __init__(
        self,
        structured_embedder: StructuredRNAEditEmbedder,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        n_tasks: int = 1
    ):
        super().__init__()

        self.structured_embedder = structured_embedder
        self.n_tasks = n_tasks

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Prediction head
        input_dim = structured_embedder.embedding_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, n_tasks))

        self.prediction_head = nn.Sequential(*layers)

    def forward(
        self,
        sequences,
        edit_positions,
        edit_from,
        edit_to
    ):
        """
        Forward pass.

        Args:
            sequences: RNA sequence(s)
            edit_positions: Position(s) of edit
            edit_from: Original nucleotide(s)
            edit_to: New nucleotide(s)

        Returns:
            Predicted Δ values [batch, n_tasks]
        """
        # Get structured edit embedding
        edit_emb = self.structured_embedder(
            sequences, edit_positions, edit_from, edit_to
        )

        # Predict Δ
        output = self.prediction_head(edit_emb)

        if self.n_tasks == 1:
            return output.squeeze(-1)
        return output

    def predict(self, sequences, edit_positions, edit_from, edit_to):
        """Predict method for evaluation."""
        self.eval()
        with torch.no_grad():
            return self.forward(sequences, edit_positions, edit_from, edit_to)


def create_models(
    config,
    train_data: Dict,
    embedder: RNAEmbedder
) -> Dict:
    """
    Create models for RNA experiments.

    Uses the SAME model architectures as small molecules:
    - PropertyPredictor: f(sequence) → property
    - EditEffectPredictor: g(sequence, edit) → Δproperty

    Args:
        config: RNAExperimentConfig
        train_data: Dict with training data per property
        embedder: RNA embedder instance

    Returns:
        Dict of model configurations
    """
    models = {}
    task_names = list(train_data.keys())

    for method_config in config.methods:
        method_name = method_config['name']
        method_type = method_config['type']

        if method_type == 'baseline_property':
            # Baseline: predict property from single sequence
            model = PropertyPredictor(
                embedder=embedder,
                task_names=task_names,
                hidden_dims=method_config.get('hidden_dims'),
                head_hidden_dims=method_config.get('head_hidden_dims'),
                dropout=method_config.get('dropout', 0.2),
                learning_rate=method_config.get('lr', 0.001),
                batch_size=method_config.get('batch_size', 32),
                max_epochs=method_config.get('max_epochs', 50)
            )

            models[method_name] = {
                'type': 'baseline_property',
                'embedder': embedder,
                'model': model,
                'config': method_config
            }

        elif method_type == 'edit_framework':
            # Edit framework: predict Δ from (sequence, edit)
            edit_embedder = create_edit_embedder(
                rna_embedder=embedder,
                use_local_context=method_config.get('use_local_context', False),
                context_window=method_config.get('context_window', 25),
                include_edit_features=method_config.get('include_edit_features', False)
            )

            model = EditEffectPredictor(
                mol_embedder=embedder,  # Note: uses mol_embedder interface
                edit_embedder=edit_embedder,
                task_names=task_names,
                hidden_dims=method_config.get('hidden_dims'),
                head_hidden_dims=method_config.get('head_hidden_dims'),
                dropout=method_config.get('dropout', 0.2),
                learning_rate=method_config.get('lr', 0.001),
                gnn_learning_rate=method_config.get('gnn_lr', 1e-5),
                batch_size=method_config.get('batch_size', 32),
                max_epochs=method_config.get('max_epochs', 50),
                trainable_edit_embeddings=method_config.get('trainable_edit_embeddings', True),
                trainable_edit_hidden_dims=method_config.get('trainable_edit_dims', [512, 256]),
                trainable_edit_use_fragments=method_config.get('use_edit_fragments', False)
            )

            models[method_name] = {
                'type': 'edit_framework',
                'rna_embedder': embedder,
                'edit_embedder': edit_embedder,
                'model': model,
                'config': method_config
            }

        elif method_type == 'structured_edit_framework':
            # Structured edit framework: uses StructuredRNAEditEmbedder
            # Requires RNA-FM embedder
            try:
                from src.embedding.rna.rnafm import RNAFMEmbedder
            except ImportError:
                raise ImportError(
                    "structured_edit_framework requires RNA-FM. "
                    "Install with: pip install git+https://github.com/ml4bio/RNA-FM.git"
                )

            # Create or use existing RNA-FM embedder
            rnafm_embedder = create_rna_embedder(
                embedder_type='rnafm',
                trainable=False,  # Keep RNA-FM frozen
                device=method_config.get('device', 'auto'),
                model_name=method_config.get('rnafm_model', 'rna_fm_t12'),
                pooling=method_config.get('rnafm_pooling', 'mean')
            )

            # Create structured edit embedder
            structured_embedder = create_structured_edit_embedder(
                rnafm_embedder=rnafm_embedder,
                mutation_type_dim=method_config.get('mutation_type_dim', 64),
                mutation_effect_dim=method_config.get('mutation_effect_dim', 256),
                position_dim=method_config.get('position_dim', 64),
                local_context_dim=method_config.get('local_context_dim', 256),
                attention_context_dim=method_config.get('attention_context_dim', 128),
                fusion_hidden_dims=method_config.get('fusion_hidden_dims', [512, 384]),
                output_dim=method_config.get('output_dim', 256),
                window_size=method_config.get('window_size', 10),
                dropout=method_config.get('embedder_dropout', 0.1)
            )

            # Create end-to-end model
            model = StructuredEditEffectModel(
                structured_embedder=structured_embedder,
                hidden_dims=method_config.get('head_hidden_dims', [128, 64]),
                dropout=method_config.get('dropout', 0.1),
                n_tasks=len(task_names)
            )

            models[method_name] = {
                'type': 'structured_edit_framework',
                'rnafm_embedder': rnafm_embedder,
                'structured_embedder': structured_embedder,
                'model': model,
                'config': method_config
            }

        else:
            raise ValueError(f"Unknown method type: {method_type}")

    return models


def create_embedder_from_config(config) -> RNAEmbedder:
    """
    Create embedder from experiment config.

    Args:
        config: RNAExperimentConfig

    Returns:
        RNAEmbedder instance
    """
    if config.embedder_type == 'nucleotide':
        return create_rna_embedder(
            embedder_type='nucleotide',
            kmer_sizes=config.nucleotide_kmer_sizes,
            include_structure=config.nucleotide_include_structure
        )

    elif config.embedder_type == 'rnafm':
        return create_rna_embedder(
            embedder_type='rnafm',
            trainable=config.trainable_embedder,
            device=config.embedder_device,
            model_name=config.rnafm_model,
            pooling=config.rnafm_pooling
        )

    elif config.embedder_type == 'rnabert':
        return create_rna_embedder(
            embedder_type='rnabert',
            trainable=config.trainable_embedder,
            device=config.embedder_device,
            model_name=config.rnabert_model,
            pooling=config.rnabert_pooling
        )

    else:
        raise ValueError(f"Unknown embedder type: {config.embedder_type}")

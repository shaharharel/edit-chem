"""
Factory functions for creating antibody embedders, edit embedders, and predictors.

Provides a unified interface to instantiate various model components
based on configuration dictionaries.
"""

from typing import Dict, Any, Optional, List
import torch


def create_embedder(
    embedder_type: str,
    device: str = 'auto',
    trainable: bool = False,
    **kwargs,
):
    """
    Create an antibody sequence embedder.

    Args:
        embedder_type: Type of embedder ('igt5', 'igbert', 'antiberta2', 'ablang2', 'balm', 'balm_paired')
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        trainable: Whether to allow fine-tuning
        **kwargs: Additional arguments for the embedder

    Returns:
        AntibodyEmbedder instance
    """
    embedder_type = embedder_type.lower()

    if embedder_type == 'igt5':
        from src.embedding.antibody import IgT5Embedder
        return IgT5Embedder(
            model_name=kwargs.get('model_name', 'Exscientia/IgT5'),
            trainable=trainable,
            device=device,
            pooling=kwargs.get('pooling', 'mean'),
        )

    elif embedder_type == 'igbert':
        from src.embedding.antibody import IgBertEmbedder
        return IgBertEmbedder(
            model_name=kwargs.get('model_name', 'Exscientia/IgBert'),
            trainable=trainable,
            device=device,
            pooling=kwargs.get('pooling', 'mean'),
        )

    elif embedder_type == 'antiberta2':
        from src.embedding.antibody import AntiBERTa2Embedder
        return AntiBERTa2Embedder(
            model_name=kwargs.get('model_name', 'alchemab/antiberta2'),
            trainable=trainable,
            device=device,
            pooling=kwargs.get('pooling', 'mean'),
        )

    elif embedder_type == 'ablang2':
        from src.embedding.antibody import AbLang2Embedder
        return AbLang2Embedder(
            model_name=kwargs.get('model_name', 'qilowoq/AbLang2'),
            trainable=trainable,
            device=device,
            pooling=kwargs.get('pooling', 'mean'),
        )

    elif embedder_type == 'balm':
        from src.embedding.antibody import BALMEmbedder
        return BALMEmbedder(
            model_name=kwargs.get('model_name', 'beam-labs/BALM'),
            trainable=trainable,
            device=device,
            pooling=kwargs.get('pooling', 'mean'),
        )

    elif embedder_type == 'balm_paired':
        from src.embedding.antibody import BALMPairedEmbedder
        return BALMPairedEmbedder(
            model_name=kwargs.get('model_name', 'briney/BALM-paired'),
            trainable=trainable,
            device=device,
            pooling=kwargs.get('pooling', 'mean'),
        )

    else:
        raise ValueError(
            f"Unknown embedder type: {embedder_type}. "
            f"Available: igt5, igbert, antiberta2, ablang2, balm, balm_paired"
        )


def create_structural_encoder(
    encoder_type: str,
    device: str = 'auto',
    **kwargs,
):
    """
    Create a structural encoder for antibody structures.

    Args:
        encoder_type: Type of encoder ('gvp', 'se3', 'equiformer')
        device: Device to use
        **kwargs: Additional arguments for the encoder

    Returns:
        StructuralEncoder instance
    """
    encoder_type = encoder_type.lower()

    if encoder_type == 'gvp':
        from src.embedding.antibody.structural import GVPEncoder
        return GVPEncoder(
            node_dim=kwargs.get('node_dim', 128),
            edge_dim=kwargs.get('edge_dim', 32),
            num_layers=kwargs.get('num_layers', 3),
            device=device,
        )

    elif encoder_type == 'se3':
        from src.embedding.antibody.structural import SE3TransformerEncoder
        return SE3TransformerEncoder(
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_layers=kwargs.get('num_layers', 4),
            num_heads=kwargs.get('num_heads', 8),
            device=device,
        )

    elif encoder_type == 'equiformer':
        from src.embedding.antibody.structural import EquiformerEncoder
        return EquiformerEncoder(
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_layers=kwargs.get('num_layers', 4),
            num_heads=kwargs.get('num_heads', 8),
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown structural encoder: {encoder_type}. "
            f"Available: gvp, se3, equiformer"
        )


def create_edit_embedder(
    ab_embedder,
    embedder_type: str = 'simple',
    structural_encoder=None,
    **kwargs,
):
    """
    Create an antibody edit embedder.

    Args:
        ab_embedder: Base antibody embedder
        embedder_type: Type of edit embedder ('simple', 'structured')
        structural_encoder: Optional structural encoder
        **kwargs: Additional arguments for the edit embedder

    Returns:
        AntibodyEditEmbedder or StructuredAntibodyEditEmbedder instance
    """
    embedder_type = embedder_type.lower()

    if embedder_type == 'simple':
        from src.embedding.antibody import AntibodyEditEmbedder
        return AntibodyEditEmbedder(
            base_embedder=ab_embedder,
            output_dim=kwargs.get('output_dim', 320),
            dropout=kwargs.get('embedder_dropout', 0.1),
        )

    elif embedder_type == 'structured':
        from src.embedding.antibody import StructuredAntibodyEditEmbedder

        return StructuredAntibodyEditEmbedder(
            base_embedder=ab_embedder,
            structural_encoder=structural_encoder,
            identity_dim=kwargs.get('identity_dim', 32),
            location_dim=kwargs.get('location_dim', 64),
            context_dim=kwargs.get('context_dim', 128),
            structure_dim=kwargs.get('structure_dim', 64),
            fusion_hidden_dims=kwargs.get('fusion_hidden_dims', [512, 384]),
            output_dim=kwargs.get('output_dim', 320),
            window_size=kwargs.get('window_size', 5),
            aggregation=kwargs.get('aggregation', 'mean'),
            dropout=kwargs.get('embedder_dropout', 0.1),
        )

    else:
        raise ValueError(
            f"Unknown edit embedder type: {embedder_type}. "
            f"Available: simple, structured"
        )


def create_predictor(
    ab_embedder,
    edit_embedder,
    method_config: Dict[str, Any],
    device: str = 'auto',
):
    """
    Create an antibody effect predictor based on method configuration.

    Args:
        ab_embedder: Antibody embedder
        edit_embedder: Edit embedder
        method_config: Configuration dictionary for the method
        device: Device to use

    Returns:
        AntibodyEffectPredictor instance
    """
    from src.models.predictors import AntibodyEffectPredictor

    return AntibodyEffectPredictor(
        ab_embedder=ab_embedder,
        edit_embedder=edit_embedder,
        hidden_dims=method_config.get('hidden_dims'),
        head_hidden_dims=method_config.get('head_hidden_dims'),
        dropout=method_config.get('dropout', 0.2),
        learning_rate=method_config.get('lr', 1e-3),
        embedder_learning_rate=method_config.get('embedder_lr', 1e-5),
        batch_size=method_config.get('batch_size', 32),
        max_epochs=method_config.get('max_epochs', 50),
        device=device,
        task_names=method_config.get('task_names'),
        task_weights=method_config.get('task_weights'),
    )


def setup_experiment(config: 'AntibodyExperimentConfig') -> Dict[str, Any]:
    """
    Set up all components for an experiment based on configuration.

    Args:
        config: AntibodyExperimentConfig instance

    Returns:
        Dict with 'embedder', 'structural_encoder', 'edit_embedder', 'dataset'
    """
    # Create base embedder
    embedder = create_embedder(
        embedder_type=config.embedder_type,
        device=config.embedder_device,
        trainable=config.trainable_embedder,
        model_name=getattr(config, f'{config.embedder_type}_model', None),
        pooling=config.pooling,
    )

    # Create structural encoder if needed
    structural_encoder = None
    if config.use_structure and config.structure_dir:
        structural_encoder = create_structural_encoder(
            encoder_type=config.structural_encoder,
            device=config.embedder_device,
        )

    # Create edit embedder
    edit_embedder = create_edit_embedder(
        ab_embedder=embedder,
        embedder_type=config.edit_embedder_type,
        structural_encoder=structural_encoder,
        window_size=getattr(config, 'context_window', 5),
        aggregation=getattr(config, 'aggregation', 'mean'),
    )

    # Load dataset
    from src.data.antibody import AbPairDataset

    dataset = AbPairDataset.from_loader(
        loader_name=config.dataset_type,
        data_dir=config.data_dir,
        **config.dataset_kwargs,
    )

    return {
        'embedder': embedder,
        'structural_encoder': structural_encoder,
        'edit_embedder': edit_embedder,
        'dataset': dataset,
    }

from pathlib import Path
from typing import Literal, Optional, Union

from torch.nn import Module

from transformers import AutoConfig
from speculators.base import (
    SpeculatorModel,
    SpeculatorConfig,
    DraftModelConfig,
    DraftModelType,
    TokenProposalConfig,
    TokenProposalType,
    AlgorithmType,
    VerifierConfig,
)
import logging

__all__ = [
    "SpecDecodeLibraryFormats",
    "convert_to_speculators",
    "detect_model_format",
    "from_eagle2_format",
    "from_eagle3_format",
    "from_eagle_format",
    "from_hass_format",
]


SpecDecodeLibraryFormats = Literal["speculators", "eagle", "eagle2", "eagle3", "hass"]
logger = logging.getLogger(__name__)

def detect_model_format(
    source: Union[str, Path, Module],  # noqa: ARG001
    config: Optional[Union[str, Path, dict, SpeculatorConfig]],  # noqa: ARG001
) -> SpecDecodeLibraryFormats:
    """
    Detect the model format based on the source and config.

    :param source: A local directory containing a speculators config and model files,
        a local directory or file containing a model from another supported library,
        or a model instance from another supported library.
    :param config: The path to the config file or a config object.
        If not provided, the config will be loaded from the source if needed.
    :return: The detected model format.
    """
    raise NotImplementedError("Model format detection is not implemented yet.")


def convert_to_speculators(
    source: Union[str, Path, Module],
    config: Optional[Union[str, Path, dict]],
    verifier: Optional[Union[str, Path, Module]],
    format_: Optional[SpecDecodeLibraryFormats],
    **kwargs,
) -> SpeculatorModel:
    """
    Convert a model from a specific format to the speculators library format.

    :param source: The path (str, Path) to the model or the model itself (Module).
    :param config: The path (str, Path) to the config file or the config itself (dict).
    :param verifier: The HuggingFace model ID, local directory,
        or a PyTorch module representing the verifier model.
        If not provided, the verifier model will be loaded from the source,
        if available. If set to None, the verifier model will not be loaded.
    :param format_: The format of the model to convert from.
        If not provided, the format will be detected automatically.
    :param kwargs: Additional keyword arguments for loading the model.
        This is still in process and more named arguments will be added to the
        function signature rather than keeping them in kwargs as needed.
    :return: The converted speculator model.
    :raises ValueError: If the format is not supported.
    """
    format_ = format_ or detect_model_format(source, config)
    if format_ == "eagle":
        return from_eagle_format(source, config, verifier, **kwargs)
    elif format_ == "hass":
        return from_hass_format(source, config, verifier, **kwargs)
    elif format_ == "eagle3":
        return from_eagle3_format(source, config, verifier, **kwargs)
    else:
        raise ValueError(f"Unsupported model format: {format_}")


def from_eagle_format(
    source: Union[str, Path, Module],  # noqa: ARG001
    config: Optional[Union[str, Path, dict]] = None,  # noqa: ARG001
    verifier: Optional[Union[str, Path, Module]] = None,  # noqa: ARG001
    **kwargs,  # noqa: ARG001
) -> SpeculatorModel:
    """
    Convert a model from the Eagle repo v1 format to the speculators library format.

    :param source: The path (str, Path) to the Eagle repo model
        or the model itself (Module).
    :param config: The path (str, Path) to the Eagle repo config file
        or the config itself (dict).
    :param verifier: The HuggingFace model ID, local directory,
        or a PyTorch module representing the verifier model.
        If not provided, the verifier model will be loaded from the source,
        if available. If set to None, the verifier model will not be loaded.
    :param kwargs: Additional keyword arguments for loading the model.
        This is still in process and more named arguments will be added to the
        function signature rather than keeping them in kwargs as needed.
    :return: The converted speculator model.
    """
    # Load config if it's not already a dict
    if not isinstance(config, dict):
        config = AutoConfig.from_pretrained(source if config is None else config)

    # Build draft model config
    draft_model_config = DraftModelConfig(
        type_=DraftModelType.LLAMA_EAGLE,
        inputs="model.layers[-1].*",
        config=config,
    )

    # Set proposal config
    proposal_config = TokenProposalConfig(type_=TokenProposalType.GREEDY)

    # Construct verifier config
    verifier_config = VerifierConfig(
        architecture=config.get("architectures"),
        model=getattr(verifier, "source", None) if verifier else None,
    )

    # Build full speculator config
    speculator_config = SpeculatorConfig(
        speculators_algorithm=AlgorithmType.EAGLE,
        draft_model=draft_model_config,
        proposal_methods=proposal_config,
        default_proposal_method=proposal_config.type_,
        verifier=verifier_config,
    )

    speculator_model = SpeculatorModel.from_config(speculator_config)
    logger.info(
        "Returning speculator model from config. "
        "Note: weights have not been loaded from disk yet."
        )
    return speculator_model

def from_eagle2_format(
    source: Union[str, Path, Module],  # noqa: ARG001
    config: Optional[Union[str, Path, dict]] = None,  # noqa: ARG001
    verifier: Optional[Union[str, Path, Module]] = None,  # noqa: ARG001
    **kwargs,  # noqa: ARG001
) -> SpeculatorModel:
    """
    Convert a model from the Eagle repo v2 format to the speculators library format.

    :param source: The path (str, Path) to the Eagle repo model
        or the model itself (Module).
    :param config: The path (str, Path) to the Eagle repo config file
        or the config itself (dict).
    :param verifier: The HuggingFace model ID, local directory,
        or a PyTorch module representing the verifier model.
        If not provided, the verifier model will be loaded from the source,
        if available. If set to None, the verifier model will not be loaded.
    :param kwargs: Additional keyword arguments for loading the model.
        This is still in process and more named arguments will be added to the
        function signature rather than keeping them in kwargs as needed.
    :return: The converted speculator model.
    """
    raise NotImplementedError("Eagle v2 format conversion is not implemented yet.")


def from_eagle3_format(
    source: Union[str, Path, Module],  # noqa: ARG001
    config: Optional[Union[str, Path, dict]] = None,  # noqa: ARG001
    verifier: Optional[Union[str, Path, Module]] = None,  # noqa: ARG001
    **kwargs,  # noqa: ARG001
) -> SpeculatorModel:
    """
    Convert a model from the Eagle repo v3 format to the speculators library format.

    :param source: The path (str, Path) to the Eagle repo model
        or the model itself (Module).
    :param config: The path (str, Path) to the Eagle repo config file
        or the config itself (dict).
    :param verifier: The HuggingFace model ID, local directory,
        or a PyTorch module representing the verifier model.
        If not provided, the verifier model will be loaded from the source,
        if available. If set to None, the verifier model will not be loaded.
    :param kwargs: Additional keyword arguments for loading the model.
        This is still in process and more named arguments will be added to the
        function signature rather than keeping them in kwargs as needed.
    :return: The converted speculator model.
    """
    raise NotImplementedError("Eagle v3 format conversion is not implemented yet.")


def from_hass_format(
    source: Union[str, Path, Module],  # noqa: ARG001
    config: Optional[Union[str, Path, dict]] = None,  # noqa: ARG001
    verifier: Optional[Union[str, Path, Module]] = None,  # noqa: ARG001
    **kwargs,  # noqa: ARG001
) -> SpeculatorModel:
    """
    Convert a model from the Hass repo format to the speculators library format.

    :param source: The path (str, Path) to the Hass repo model
        or the model itself (Module).
    :param config: The path (str, Path) to the Hass repo config file
        or the config itself (dict).
    :param verifier: The HuggingFace model ID, local directory,
        or a PyTorch module representing the verifier model.
        If not provided, the verifier model will be loaded from the source,
        if available. If set to None, the verifier model will not be loaded.
    :param kwargs: Additional keyword arguments for loading the model.
        This is still in process and more named arguments will be added to the
        function signature rather than keeping them in kwargs as needed.
    :return: The converted speculator model.
    """
    raise NotImplementedError("Hass format conversion is not implemented yet.")

from .eagle import EagleSpeculator, EagleSpeculatorConfig
from .independent import IndependentSpeculatorConfig
from .mlp import MLPSpeculatorConfig
from .transformer import TransformerSpeculatorConfig

__all__ = [
    "EagleSpeculator",
    "EagleSpeculatorConfig",
    "IndependentSpeculatorConfig",
    "MLPSpeculatorConfig",
    "TransformerSpeculatorConfig",
]

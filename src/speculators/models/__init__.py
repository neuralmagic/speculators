from .independent import IndependentSpeculatorConfig
from .llama_eagle import LlamaEagleSpeculator
from .mlp import MLPSpeculatorConfig
from .transformer import TransformerSpeculatorConfig

__all__ = [
    "IndependentSpeculatorConfig",
    "LlamaEagleSpeculator",
    "MLPSpeculatorConfig",
    "TransformerSpeculatorConfig",
]

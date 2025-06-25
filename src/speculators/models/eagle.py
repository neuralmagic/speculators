"""
Eagle model implementation for EAGLE1 and HASS speculator models.

This module provides a unified implementation for both EAGLE1 and HASS variants
through configurable parameters.

Classes:
    EagleSpeculatorConfig: Configuration for EAGLE/HASS models
    EagleSpeculator: Model implementation for EAGLE/HASS speculators
"""

from typing import Literal, Optional, Union

import torch
from pydantic import Field
from torch import nn
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from speculators.config import SpeculatorModelConfig

__all__ = [
    "EagleSpeculator",
    "EagleSpeculatorConfig",
]


@SpeculatorModelConfig.register("eagle")
class EagleSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration class for EAGLE1 and HASS speculator models.

    This unified configuration supports both EAGLE1 and HASS variants through
    configurable parameters, allowing a single model implementation to handle
    both architectures.
    """

    # Model identification
    speculators_model_type: Literal["eagle"] = "eagle"
    architectures: list[str] = Field(
        default=["EagleSpeculator"],
        description=(
            "List of model architectures that can be used with the "
            "model pretrained weights."
        ),
    )
    transformer_layer_architecture: str = Field(
        default="LlamaDecoderLayer",
        description=(
            "The architecture of the transformer layer to use. "
            "Typically 'LlamaDecoderLayer' for Eagle 1, Eagle 2, and HASS."
        ),
    )
    transformer_layer_config: PretrainedConfig = Field(
        default_factory=LlamaConfig,
        description=(
            "Configuration for the transformer layer to use in the "
            "Eagle model architecture. This must be a PretrainedConfig that matches "
            "the config required by the transformer_layer_architecture."
        ),
    )
    layernorms: bool = Field(
        default=False,
        description=(
            "Whether to use additional layernorms in the model architecture, "
            "specifically the layernorm after the verifier's hidden state, "
            "after the fusion layer, and before the LM head. "
            "For Eagle, Eagle 1, and HASS, this is False."
        ),
    )
    fusion_bias: bool = Field(
        default=False,
        description=(
            "Whether to add a bias to the fusion (fc) layer that is applied to the "
            "concat of the input embeddings and input hidden state. "
            "For Eagle and Eagle 2, this is False, while for HASS it is True."
        ),
    )

    def to_dict(self):
        """Convert to dictionary, handling the transformer_layer_config specially."""
        output = super().to_dict()
        # Convert transformer_layer_config to dict if it's a PretrainedConfig
        if hasattr(self.transformer_layer_config, "to_dict"):
            output["transformer_layer_config"] = self.transformer_layer_config.to_dict()
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Create from dictionary, handling the transformer_layer_config specially."""
        # Convert transformer_layer_config dict to LlamaConfig if needed
        if "transformer_layer_config" in config_dict and isinstance(
            config_dict["transformer_layer_config"], dict
        ):
            from transformers import LlamaConfig

            config_dict = config_dict.copy()
            config_dict["transformer_layer_config"] = LlamaConfig(
                **config_dict["transformer_layer_config"]
            )
        return super().from_dict(config_dict, **kwargs)


class EagleSpeculator(PreTrainedModel, GenerationMixin):
    """
    Eagle speculator model for speculative decoding.

    This model implements EAGLE1, EAGLE2, and HASS variants through configuration.
    The key differences between variants are:
    - EAGLE1/2: layernorms=False, fusion_bias=False (EAGLE2 same architecture)
    - HASS: layernorms=False, fusion_bias=True
    - TTT variant: layernorms=True, fusion_bias varies
    """

    config_class = EagleSpeculatorConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config: EagleSpeculatorConfig):
        super().__init__(config)
        self.config = config

        # Extract dimensions from transformer config for cleaner access
        self.vocab_size = config.transformer_layer_config.vocab_size
        self.hidden_size = config.transformer_layer_config.hidden_size
        self.padding_idx = config.transformer_layer_config.pad_token_id

        # Initialize model components
        self._init_embeddings()
        self._init_fusion_layer()
        self._init_decoder_layers()
        self._init_output_layer()

        # Initialize extra components if needed
        if config.layernorms:
            self._init_extra_layernorms()

        # Initialize weights and apply final processing
        self.post_init()

    def _init_embeddings(self):
        """Initialize embedding layer and rotary embeddings."""
        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.hidden_size, self.padding_idx
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            config=self.config.transformer_layer_config
        )

    def _init_fusion_layer(self):
        """Initialize fusion layer that combines embeddings and hidden states."""
        self.fc = nn.Linear(
            2 * self.hidden_size,  # Concatenated embeddings + hidden states
            self.hidden_size,
            bias=self.config.fusion_bias,
        )

    def _init_decoder_layers(self):
        """Initialize decoder layers and optionally modify input layernorm."""
        # Eagle speculators always use a single decoder layer
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(self.config.transformer_layer_config, layer_idx=0)]
        )

        # Replace input layernorm with Identity for standard Eagle/HASS
        if not self.config.layernorms:
            self.layers[0].input_layernorm = nn.Identity()

    def _init_output_layer(self):
        """Initialize output projection layer."""
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def _init_extra_layernorms(self):
        """Initialize extra layernorms for TTT variant."""
        eps = self.config.transformer_layer_config.rms_norm_eps

        # Applied after input embeddings
        self.post_embedding_layernorm = LlamaRMSNorm(self.hidden_size, eps=eps)

        # Applied before LM head (acts as final layer norm)
        self.pre_lm_head_layernorm = LlamaRMSNorm(self.hidden_size, eps=eps)

    def get_input_embeddings(self):
        """Get input embeddings layer."""
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set input embeddings layer."""
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,  # noqa: ARG002
        return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithPast]:
        """
        Forward pass of the Eagle speculator.

        Args:
            input_ids: Input token IDs from the verifier [batch_size, seq_length]
            hidden_states: Hidden states from the verifier
                [batch_size, seq_length, hidden_size]
            attention_mask: Attention mask [batch_size, seq_length]
            position_ids: Position IDs for rotary embeddings
            past_key_values: Past key values for KV caching
            use_cache: Whether to return past key values
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a ModelOutput instead of tuple

        Returns:
            Logits for next token prediction [batch_size, seq_length, vocab_size]
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Get input embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        # Apply post-embedding layernorm if configured (TTT variant)
        if hasattr(self, "post_embedding_layernorm"):
            inputs_embeds = self.post_embedding_layernorm(inputs_embeds)

        # Fusion: concatenate embeddings and hidden states, then project
        hidden_states = self.fc(torch.cat([inputs_embeds, hidden_states], dim=-1))

        # Prepare inputs for decoder layer
        hidden_states, attention_mask, position_ids = self._prepare_decoder_inputs(
            hidden_states, attention_mask, position_ids, past_key_values
        )

        # Get rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Pass through the single decoder layer
        layer_outputs = self.layers[0](
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values[0] if past_key_values else None,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=(cos, sin),
        )

        hidden_states = layer_outputs[0]

        # Apply pre-LM head layernorm if configured (TTT variant)
        if hasattr(self, "pre_lm_head_layernorm"):
            hidden_states = self.pre_lm_head_layernorm(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return logits

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=layer_outputs[1] if use_cache else None,
            hidden_states=None,  # Not collecting hidden states for now
            attentions=None,  # Not collecting attentions for now
        )

    def _prepare_decoder_inputs(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]],
    ) -> tuple[torch.FloatTensor, torch.Tensor, torch.LongTensor]:
        """
        Prepare inputs for the decoder layer.

        This method handles:
        - Creating position IDs if not provided
        - Converting 2D attention mask to 4D causal mask
        - Ensuring all inputs have correct shapes and types
        """
        batch_size, seq_length = hidden_states.shape[:2]

        # Create position IDs if not provided
        if position_ids is None:
            device = hidden_states.device
            position_ids = (
                torch.arange(seq_length, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Prepare 4D causal attention mask if needed
        if attention_mask is not None and attention_mask.dim() == 2:  # noqa: PLR2004
            past_key_values_length = (
                past_key_values[0][0].shape[2] if past_key_values else 0
            )
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                past_key_values_length,
                sliding_window=getattr(self.config, "sliding_window", None),
            )

        return hidden_states, attention_mask, position_ids

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,  # noqa: ARG002
        **kwargs,
    ) -> dict:
        """
        Prepare inputs for generation step.

        This method is used by generate() and contrastive search.
        """
        # If past_key_values are provided, only use the last token
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # Position IDs
        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            # Create position IDs based on attention mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # Hidden states must be provided for Eagle
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError("Eagle speculator requires hidden_states from verifier")

        return {
            "input_ids": input_ids,
            "hidden_states": hidden_states,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):  # noqa: C901, PLR0912
        """
        Load pretrained model from HuggingFace hub or local path.

        This method handles:
        - Auto-detecting model configuration from checkpoint
        - Loading weights with proper mapping
        - Handling missing lm_head weights gracefully
        """
        from pathlib import Path

        from safetensors.torch import load_file as safe_load_file

        # First, load the configuration
        config = kwargs.pop("config", None)
        if config is None:
            # Try to load our custom config first
            try:
                config = EagleSpeculatorConfig.from_pretrained(
                    pretrained_model_name_or_path
                )
            except Exception:  # noqa: BLE001
                # Fall back to creating config from LlamaConfig
                from transformers import LlamaConfig

                llama_config = LlamaConfig.from_pretrained(
                    pretrained_model_name_or_path
                )
                # Ensure single layer for Eagle
                llama_config.num_hidden_layers = 1
                config = EagleSpeculatorConfig(transformer_layer_config=llama_config)

        # Create the model
        model = cls(config)

        # Try to load weights
        if Path(pretrained_model_name_or_path).is_dir():
            # Local directory
            model_file = None
            for filename in ["model.safetensors", "pytorch_model.bin"]:
                filepath = Path(pretrained_model_name_or_path) / filename
                if filepath.exists():
                    model_file = filepath
                    break

            if model_file:
                # Load the state dict
                if str(model_file).endswith(".safetensors"):
                    state_dict = safe_load_file(model_file)
                else:
                    state_dict = torch.load(model_file, map_location="cpu")

                # Handle different naming conventions
                renamed_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key
                    # Map TTT naming to our naming
                    if key == "embed_layernorm.weight":
                        new_key = "post_embedding_layernorm.weight"
                    elif key == "hidden_layernorm.weight":
                        # TTT uses hidden_layernorm differently - skip it
                        continue
                    elif key == "lm_head_layernorm.weight":
                        new_key = "pre_lm_head_layernorm.weight"
                    renamed_state_dict[new_key] = value

                # Load with strict=False to handle missing lm_head
                missing_keys, unexpected_keys = model.load_state_dict(
                    renamed_state_dict, strict=False
                )

                # Filter out expected missing keys
                expected_missing = ["lm_head.weight"]
                critical_missing = [
                    k
                    for k in missing_keys
                    if not any(em in k for em in expected_missing)
                ]

                if critical_missing:
                    import warnings

                    warnings.warn(
                        f"Critical missing keys: {critical_missing}", stacklevel=2
                    )

                return model

        # For HuggingFace hub, use parent class method with our config
        return super().from_pretrained(
            pretrained_model_name_or_path, *args, config=config, **kwargs
        )

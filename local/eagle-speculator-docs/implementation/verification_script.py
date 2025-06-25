#!/usr/bin/env python3
"""
Verification script for EagleSpeculator model implementation.

This script tests the EagleSpeculator model with different checkpoint configurations:
- EAGLE 1: yuhuili/EAGLE-LLaMA3.1-Instruct-8B (no layernorms)
- HASS/TTT: nm-testing/Eagle_Speculator_Llama_3_1_8B_TTT (with layernorms)

The script performs the following steps:
1. Create appropriate configuration
2. Initialize model
3. Load weights and report mismatches
4. Execute forward pass
5. Save the model using save_pretrained
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import LlamaConfig, AutoTokenizer
from safetensors import safe_open

# Add the speculators package to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from speculators.models.eagle import EagleSpeculatorConfig, EagleSpeculator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure the speculators package is properly installed and EagleSpeculator is implemented.")
    sys.exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


def analyze_checkpoint(checkpoint_path: str) -> Dict[str, bool]:
    """
    Analyze checkpoint to determine its configuration.
    
    Returns:
        Dict with detected features
    """
    features = {
        "has_extra_layernorms": False,
        "has_fusion_bias": False,
        "num_layers": 1,
    }
    
    weight_names = set()
    
    # Try to get weight names from HuggingFace hub or local path
    try:
        from huggingface_hub import hf_hub_download
        
        # Try to download a single file first
        try:
            model_file = hf_hub_download(checkpoint_path, "model.safetensors")
            with safe_open(model_file, framework="pt") as f:
                weight_names = set(f.keys())
        except:
            # Try pytorch format
            try:
                model_file = hf_hub_download(checkpoint_path, "pytorch_model.bin")
                weights = torch.load(model_file, map_location='cpu')
                weight_names = set(weights.keys())
            except:
                # Try index files
                for index_name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
                    try:
                        index_file = hf_hub_download(checkpoint_path, index_name)
                        with open(index_file, 'r') as f:
                            index = json.load(f)
                            weight_names = set(index['weight_map'].keys())
                            break
                    except:
                        continue
    except:
        # Fall back to local path checking
        if os.path.isdir(checkpoint_path):
            # Check for safetensors format
            safetensors_index = os.path.join(checkpoint_path, "model.safetensors.index.json")
            pytorch_index = os.path.join(checkpoint_path, "pytorch_model.bin.index.json")
            
            if os.path.exists(safetensors_index):
                with open(safetensors_index, 'r') as f:
                    index = json.load(f)
                    weight_names = set(index['weight_map'].keys())
            elif os.path.exists(pytorch_index):
                with open(pytorch_index, 'r') as f:
                    index = json.load(f)
                    weight_names = set(index['weight_map'].keys())
            else:
                # Try single file
                if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
                    with safe_open(os.path.join(checkpoint_path, "model.safetensors"), framework="pt") as f:
                        weight_names = set(f.keys())
                elif os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
                    weights = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location='cpu')
                    weight_names = set(weights.keys())
    
    # Check for extra layernorms (multiple naming conventions)
    layernorm_indicators = [
        "post_embedding_layernorm", "pre_lm_head_layernorm",  # Our naming
        "embed_layernorm", "hidden_layernorm", "lm_head_layernorm"  # TTT naming
    ]
    if any(indicator in name for name in weight_names for indicator in layernorm_indicators):
        features["has_extra_layernorms"] = True
    
    # Check for fusion bias
    if "fc.bias" in weight_names:
        features["has_fusion_bias"] = True
    
    # Count decoder layers
    layer_count = sum(1 for name in weight_names if "layers." in name and ".self_attn.q_proj" in name)
    if layer_count > 0:
        features["num_layers"] = layer_count
    
    return features


def create_config_for_checkpoint(checkpoint_path: str, base_model_name: str = "meta-llama/Llama-3.1-8B") -> EagleSpeculatorConfig:
    """
    Create appropriate EagleSpeculatorConfig based on checkpoint analysis.
    
    Args:
        checkpoint_path: Path to the checkpoint
        base_model_name: Name of the base LLaMA model
        
    Returns:
        EagleSpeculatorConfig instance
    """
    print_section("Step 1: Creating Configuration")
    
    # Try to load the checkpoint's own config first
    try:
        checkpoint_config = LlamaConfig.from_pretrained(checkpoint_path)
        print(f"Loaded config from checkpoint: {checkpoint_path}")
        print(f"  - hidden_size: {checkpoint_config.hidden_size}")
        print(f"  - vocab_size: {checkpoint_config.vocab_size}")
        print(f"  - num_hidden_layers: {checkpoint_config.num_hidden_layers}")
        
        # Use checkpoint config as base
        base_config = checkpoint_config
    except Exception as e:
        print(f"Could not load config from checkpoint: {e}")
        # Fall back to base model config
        print(f"\nLoading base config from: {base_model_name}")
        base_config = LlamaConfig.from_pretrained(base_model_name)
    
    # Analyze checkpoint
    print(f"\nAnalyzing checkpoint weights...")
    features = analyze_checkpoint(checkpoint_path)
    print(f"Detected features:")
    for key, value in features.items():
        print(f"  - {key}: {value}")
    
    # Eagle models typically have 1 layer
    if base_config.num_hidden_layers != 1:
        print(f"  Note: Adjusting num_hidden_layers from {base_config.num_hidden_layers} to 1 for Eagle")
        base_config.num_hidden_layers = 1
    
    # Create config based on detected features
    config = EagleSpeculatorConfig(
        transformer_layer_config=base_config,
        layernorms=features["has_extra_layernorms"],
        fusion_bias=features["has_fusion_bias"],
    )
    
    print(f"\nConfiguration created:")
    print(f"  - layernorms: {config.layernorms}")
    print(f"  - fusion_bias: {config.fusion_bias}")
    print(f"  - hidden_size: {config.transformer_layer_config.hidden_size}")
    print(f"  - vocab_size: {config.transformer_layer_config.vocab_size}")
    print(f"  - num_hidden_layers: {config.transformer_layer_config.num_hidden_layers}")
    
    return config


def initialize_model(config: EagleSpeculatorConfig) -> EagleSpeculator:
    """Initialize EagleSpeculator model with given configuration."""
    print_section("Step 2: Initializing Model")
    
    try:
        model = EagleSpeculator(config)
        print("Model initialized successfully")
        
        # Print model structure
        print("\nModel structure:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        # List main components
        print("\nMain components:")
        for name, module in model.named_children():
            param_count = sum(p.numel() for p in module.parameters())
            print(f"  - {name}: {module.__class__.__name__} ({param_count:,} params)")
        
        return model
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise


def load_checkpoint_weights(
    model: EagleSpeculator, 
    checkpoint_path: str
) -> Dict[str, List[str]]:
    """
    Load weights from checkpoint using from_pretrained.
    
    Returns:
        Dict with loading results
    """
    print_section("Step 3: Loading Checkpoint Weights")
    
    print(f"Loading from: {checkpoint_path}")
    
    try:
        # Load using from_pretrained - this should handle all weight mapping automatically
        result = model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu"),
            strict=False
        )
        
        missing_keys = result.missing_keys
        unexpected_keys = result.unexpected_keys
        
        # Filter out expected missing keys
        expected_missing = ['lm_head.weight']  # lm_head comes from verifier
        critical_missing = [k for k in missing_keys if k not in expected_missing]
        
        print(f"\nWeight loading summary:")
        print(f"  - Missing keys: {len(missing_keys)}")
        print(f"  - Unexpected keys: {len(unexpected_keys)}")
        print(f"  - Critical missing keys: {len(critical_missing)}")
        
        if critical_missing:
            print(f"\nCritical missing keys:")
            for key in critical_missing[:10]:
                print(f"  - {key}")
            if len(critical_missing) > 10:
                print(f"  ... and {len(critical_missing) - 10} more")
        
        if unexpected_keys:
            print(f"\nUnexpected keys:")
            for key in unexpected_keys[:10]:
                print(f"  - {key}")
            if len(unexpected_keys) > 10:
                print(f"  ... and {len(unexpected_keys) - 10} more")
        
        # Check if loading was successful
        if len(critical_missing) == 0:
            print("\n✓ Weight loading successful (ignoring expected missing lm_head)")
        else:
            print("\n✗ Weight loading failed - critical weights missing")
        
        return {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "critical_missing": critical_missing,
            "success": len(critical_missing) == 0
        }
        
    except FileNotFoundError:
        # Try HuggingFace from_pretrained
        print("Trying HuggingFace from_pretrained...")
        try:
            loaded_model = EagleSpeculator.from_pretrained(checkpoint_path)
            print("✓ Loaded successfully using from_pretrained")
            # Copy loaded weights to our model
            model.load_state_dict(loaded_model.state_dict(), strict=False)
            return {
                "missing_keys": [],
                "unexpected_keys": [],
                "critical_missing": [],
                "success": True
            }
        except Exception as e:
            print(f"✗ Failed to load using from_pretrained: {e}")
            import traceback
            traceback.print_exc()
            return {
                "missing_keys": list(model.state_dict().keys()),
                "unexpected_keys": [],
                "critical_missing": list(model.state_dict().keys()),
                "success": False
            }
    except Exception as e:
        print(f"Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return {
            "missing_keys": list(model.state_dict().keys()),
            "unexpected_keys": [],
            "critical_missing": list(model.state_dict().keys()),
            "success": False
        }


def execute_forward_pass(model: EagleSpeculator, config: EagleSpeculatorConfig, device: str = "cpu") -> bool:
    """Execute a forward pass through the model."""
    print_section("Step 4: Forward Pass Test")
    
    try:
        model = model.to(device)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        seq_length = 10
        hidden_size = config.transformer_layer_config.hidden_size
        vocab_size = config.transformer_layer_config.vocab_size
        
        print(f"Creating dummy inputs:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Sequence length: {seq_length}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Vocab size: {vocab_size}")
        print(f"  - Device: {device}")
        
        # Random input tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        
        # Random hidden states from verifier
        hidden_states = torch.randn(batch_size, seq_length, hidden_size).to(device)
        
        # Optional attention mask
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool).to(device)
        
        print("\nExecuting forward pass...")
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Extract logits from output
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output
        
        print(f"✓ Forward pass successful!")
        print(f"  Output shape: {logits.shape}")
        print(f"  Expected shape: ({batch_size}, {seq_length}, {vocab_size})")
        
        # Verify output shape
        assert logits.shape == (batch_size, seq_length, vocab_size), \
            f"Output shape mismatch: got {logits.shape}, expected ({batch_size}, {seq_length}, {vocab_size})"
        
        # Check for NaN or Inf
        if torch.isnan(logits).any():
            print("✗ WARNING: Output contains NaN values")
            return False
        if torch.isinf(logits).any():
            print("✗ WARNING: Output contains Inf values")
            return False
        
        # Basic statistics
        print(f"\nOutput statistics:")
        print(f"  - Mean: {logits.mean().item():.4f}")
        print(f"  - Std: {logits.std().item():.4f}")
        print(f"  - Min: {logits.min().item():.4f}")
        print(f"  - Max: {logits.max().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_model(model: EagleSpeculator, output_path: str) -> bool:
    """Save the model using save_pretrained."""
    print_section("Step 5: Saving Model")
    
    try:
        print(f"Saving model to: {output_path}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save model
        model.save_pretrained(output_path)
        print("✓ Model saved successfully!")
        
        # Verify saved files
        saved_files = os.listdir(output_path)
        print(f"\nSaved files:")
        for file in saved_files:
            file_path = os.path.join(output_path, file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  - {file} ({size:.2f} MB)")
        
        # Check for expected files
        expected_files = ['config.json']
        missing_files = [f for f in expected_files if not any(f in file for file in saved_files)]
        
        if missing_files:
            print(f"✗ WARNING: Expected files missing: {missing_files}")
            return False
        
        # Verify config can be loaded
        try:
            loaded_config = EagleSpeculatorConfig.from_pretrained(output_path)
            print("✓ Config verification passed")
        except Exception as e:
            print(f"✗ Config verification failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to save model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify EagleSpeculator implementation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path or name of the checkpoint to test"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base LLaMA model for configuration"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_output",
        help="Directory to save the test model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on"
    )
    
    args = parser.parse_args()
    
    print(f"EagleSpeculator Verification Script")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Base model: {args.base_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Track success of each step
    results = {
        "config_creation": False,
        "model_init": False,
        "weight_loading": False,
        "forward_pass": False,
        "model_saving": False
    }
    
    try:
        # Step 1: Create configuration
        config = create_config_for_checkpoint(args.checkpoint, args.base_model)
        results["config_creation"] = True
        
        # Step 2: Initialize model
        model = initialize_model(config)
        results["model_init"] = True
        
        # Step 3: Load weights
        load_result = load_checkpoint_weights(model, args.checkpoint)
        results["weight_loading"] = load_result["success"]
        
        # Step 4: Forward pass
        if results["weight_loading"]:
            results["forward_pass"] = execute_forward_pass(model, config, args.device)
        else:
            print("\nSkipping forward pass due to weight loading failure")
        
        # Step 5: Save model
        if results["forward_pass"]:
            output_path = os.path.join(args.output_dir, f"eagle_test_{Path(args.checkpoint).name}")
            results["model_saving"] = save_model(model, output_path)
        else:
            print("\nSkipping model saving due to forward pass failure")
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print_section("Verification Summary")
    all_passed = True
    for step, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{step:.<30} {status}")
        if not success:
            all_passed = False
    
    total_passed = sum(results.values())
    total_steps = len(results)
    
    print(f"\nOverall: {total_passed}/{total_steps} steps passed")
    
    if all_passed:
        print("\n🎉 All verification steps passed! The implementation is working correctly.")
    else:
        print("\n❌ Some verification steps failed. Please check the implementation.")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
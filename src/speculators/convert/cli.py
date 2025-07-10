"""
Unified CLI interface for checkpoint conversion.
"""

from typing import Annotated

import typer

from speculators.convert.eagle.eagle_converter import EagleConverter

app = typer.Typer(
    help="Convert speculator checkpoints to the standardized speculators format.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def convert(
    input_path: Annotated[
        str,
        typer.Argument(help="Path to checkpoint (local path or HuggingFace model ID)"),
    ],
    output_path: Annotated[
        str,
        typer.Argument(help="Output directory for the converted checkpoint"),
    ],
    base_model: Annotated[
        str,
        typer.Argument(help="Base model name/path (e.g., meta-llama/Llama-3.1-8B)"),
    ],
    # Model type flags (mutually exclusive)
    eagle: Annotated[
        bool,
        typer.Option(
            "--eagle",
            help="Convert Eagle/HASS checkpoint",
        ),
    ] = False,
    # Model-specific options
    layernorms: Annotated[
        bool,
        typer.Option(
            "--layernorms",
            help="Enable extra layernorms (Eagle/HASS only)",
        ),
    ] = False,
    fusion_bias: Annotated[
        bool,
        typer.Option(
            "--fusion-bias",
            help="Enable fusion bias (Eagle/HASS only)",
        ),
    ] = False,
    # General options
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Validate the converted checkpoint",
        ),
    ] = False,
):
    """
    Convert speculator checkpoints to speculators format.

    Examples::

        # Convert Eagle checkpoint
        speculators convert --eagle yuhuili/EAGLE-LLaMA3.1-Instruct-8B \\
            ./eagle-converted meta-llama/Llama-3.1-8B-Instruct

        # Convert Eagle with layernorms enabled
        speculators convert --eagle nm-testing/Eagle_TTT ./ttt-converted \\
            meta-llama/Llama-3.1-8B-Instruct --layernorms

        # Convert Eagle with fusion bias enabled
        speculators convert --eagle ./checkpoint ./converted \\
            meta-llama/Llama-3.1-8B --fusion-bias
    """
    # Determine which converter to use
    if eagle:
        converter = EagleConverter()
        try:
            converter.convert(
                input_path,
                output_path,
                base_model,
                fusion_bias=fusion_bias,
                layernorms=layernorms,
                validate=validate,
            )
        except Exception as e:
            typer.echo(f"✗ Conversion failed: {e}", err=True)
            raise typer.Exit(1) from e
    else:
        typer.echo("Error: Please specify a model type (e.g., --eagle)", err=True)
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

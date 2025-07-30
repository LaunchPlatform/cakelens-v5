import logging
import pathlib
import sys
from typing import Optional

import click
import torch

from .detect import Detector
from .model import Model


def setup_logging(verbose: bool) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@click.command()
@click.argument("video_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to the model checkpoint file (if not provided, will load from Hugging Face Hub)",
    required=False,
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Batch size for inference (default: 1)",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    help="Device to run inference on (auto-detected if not specified)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
@click.option(
    "--output",
    type=click.Path(path_type=pathlib.Path),
    help="Output file path for results (JSON format)",
)
def detect(
    video_path: pathlib.Path,
    model_path: Optional[pathlib.Path],
    batch_size: int,
    device: Optional[str],
    verbose: bool,
    output: Optional[pathlib.Path],
) -> None:
    """
    Detect AI-generated content in video files.

    VIDEO_PATH: Path to the video file to analyze
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load the model
        model = Model()

        # Load checkpoint if provided
        if model_path is not None and model_path.exists():
            map_location = device if device is not None else "cpu"
            checkpoint = torch.load(model_path, map_location=map_location)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info("Model loaded successfully")
        else:
            logger.info("Loading model from Hugging Face Hub...")
            model.load_from_huggingface_hub(device=device)
            logger.info("Model loaded successfully from Hugging Face Hub")

        # Create detector
        detector = Detector(
            model=model,
            batch_size=batch_size,
            device=device,
        )

        # Run detection
        verdict = detector.detect(video_path)

        # Display results
        click.echo("\n" + "=" * 50)
        click.echo("DETECTION RESULTS")
        click.echo("=" * 50)
        click.echo(f"Video: {verdict.video_filepath}")
        click.echo(f"Frame count: {verdict.frame_count:,}")
        click.echo("\nPredictions:")

        from .data_types import Label

        for label, prob in zip(Label, verdict.predictions):
            percentage = prob * 100.0
            click.echo(f"  {label.value:15s}: {percentage:6.2f}%")

        # Save results to file if requested
        if output:
            import json

            results = {
                "video_filepath": str(verdict.video_filepath),
                "frame_count": verdict.frame_count,
                "predictions": {
                    label.value: prob * 100.0
                    for label, prob in zip(Label, verdict.predictions)
                },
            }

            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            click.echo(f"\nResults saved to: {output}")

    except Exception as e:
        logger.error("Detection failed: %s", e)
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    detect()

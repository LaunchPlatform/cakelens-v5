import pathlib
from unittest.mock import Mock
from unittest.mock import patch

import click.testing
import pytest

from cakelens.cli import detect


def test_detect_command_help():
    """Test that the detect command shows help correctly."""
    runner = click.testing.CliRunner()
    result = runner.invoke(detect, ["--help"])
    assert result.exit_code == 0
    assert "Detect AI-generated content in video files" in result.output
    assert "VIDEO_PATH" in result.output


def test_detect_command_missing_video_arg():
    """Test that the detect command fails with missing video argument."""
    runner = click.testing.CliRunner()
    result = runner.invoke(detect, [])
    assert result.exit_code != 0


@patch("cakelens.cli.Model")
@patch("cakelens.cli.Detector")
@patch("torch.load")
def test_detect_command_success_with_local_model(
    mock_torch_load, mock_detector_class, mock_model_class
):
    """Test successful detection run with local model file."""
    # Mock the model and detector
    mock_model = Mock()
    mock_model_class.return_value = mock_model

    mock_checkpoint = {"model_state_dict": {}}
    mock_torch_load.return_value = mock_checkpoint

    mock_detector = Mock()
    mock_verdict = Mock()
    mock_verdict.video_filepath = pathlib.Path("test.mp4")
    mock_verdict.frame_count = 1000
    mock_verdict.predictions = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
    ]
    mock_detector.detect.return_value = mock_verdict
    mock_detector_class.return_value = mock_detector

    # Create a temporary video file
    test_video = pathlib.Path("test_video.mp4")
    test_video.touch()
    test_model = pathlib.Path("test_model.pt")
    test_model.touch()

    try:
        runner = click.testing.CliRunner()
        result = runner.invoke(
            detect,
            [
                str(test_video),
                "--model-path",
                str(test_model),
                "--batch-size",
                "1",
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert "DETECTION RESULTS" in result.output
        assert "test.mp4" in result.output

    finally:
        # Clean up
        if test_video.exists():
            test_video.unlink()
        if test_model.exists():
            test_model.unlink()


def test_detect_command_nonexistent_video():
    """Test that the detect command fails with nonexistent video file."""
    runner = click.testing.CliRunner()
    result = runner.invoke(detect, ["nonexistent.mp4", "--model-path", "model.pt"])
    assert result.exit_code != 0


@patch("cakelens.cli.Model")
@patch("cakelens.cli.Detector")
def test_detect_command_huggingface_hub(mock_detector_class, mock_model_class):
    """Test successful detection run with Hugging Face Hub loading."""
    # Mock the model and detector
    mock_model = Mock()
    mock_model_class.return_value = mock_model
    mock_model.load_from_huggingface_hub.return_value = None

    mock_detector = Mock()
    mock_verdict = Mock()
    mock_verdict.video_filepath = pathlib.Path("test.mp4")
    mock_verdict.frame_count = 1000
    mock_verdict.predictions = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
    ]
    mock_detector.detect.return_value = mock_verdict
    mock_detector_class.return_value = mock_detector

    # Create a temporary video file
    test_video = pathlib.Path("test_video.mp4")
    test_video.touch()

    try:
        runner = click.testing.CliRunner()
        result = runner.invoke(
            detect, [str(test_video), "--batch-size", "1", "--verbose"]
        )

        assert result.exit_code == 0
        assert "DETECTION RESULTS" in result.output
        assert "test.mp4" in result.output
        mock_model.load_from_huggingface_hub.assert_called_once()

    finally:
        # Clean up
        if test_video.exists():
            test_video.unlink()


@patch("cakelens.cli.Model")
@patch("cakelens.cli.Detector")
def test_detect_command_with_device_option(mock_detector_class, mock_model_class):
    """Test detection with device option specified."""
    # Mock the model and detector
    mock_model = Mock()
    mock_model_class.return_value = mock_model
    mock_model.load_from_huggingface_hub.return_value = None

    mock_detector = Mock()
    mock_verdict = Mock()
    mock_verdict.video_filepath = pathlib.Path("test.mp4")
    mock_verdict.frame_count = 1000
    mock_verdict.predictions = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
    ]
    mock_detector.detect.return_value = mock_verdict
    mock_detector_class.return_value = mock_detector

    # Create a temporary video file
    test_video = pathlib.Path("test_video.mp4")
    test_video.touch()

    try:
        runner = click.testing.CliRunner()
        result = runner.invoke(
            detect, [str(test_video), "--device", "cpu", "--batch-size", "2"]
        )

        assert result.exit_code == 0
        assert "DETECTION RESULTS" in result.output
        mock_model.load_from_huggingface_hub.assert_called_once_with(device="cpu")

    finally:
        # Clean up
        if test_video.exists():
            test_video.unlink()


@patch("cakelens.cli.Model")
@patch("cakelens.cli.Detector")
def test_detect_command_with_output_option(mock_detector_class, mock_model_class):
    """Test detection with output file option."""
    # Mock the model and detector
    mock_model = Mock()
    mock_model_class.return_value = mock_model
    mock_model.load_from_huggingface_hub.return_value = None

    mock_detector = Mock()
    mock_verdict = Mock()
    mock_verdict.video_filepath = pathlib.Path("test.mp4")
    mock_verdict.frame_count = 1000
    mock_verdict.predictions = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
    ]
    mock_detector.detect.return_value = mock_verdict
    mock_detector_class.return_value = mock_detector

    # Create a temporary video file
    test_video = pathlib.Path("test_video.mp4")
    test_video.touch()
    output_file = pathlib.Path("test_output.json")

    try:
        runner = click.testing.CliRunner()
        result = runner.invoke(detect, [str(test_video), "--output", str(output_file)])

        assert result.exit_code == 0
        assert "DETECTION RESULTS" in result.output
        assert "Results saved to:" in result.output
        assert output_file.exists()

    finally:
        # Clean up
        if test_video.exists():
            test_video.unlink()
        if output_file.exists():
            output_file.unlink()

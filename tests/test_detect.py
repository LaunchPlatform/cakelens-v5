import logging
import pathlib

import pytest

from cakelens.detect import Detector
from cakelens.model import Model


@pytest.fixture
def model() -> Model:
    model = Model()
    model.load_from_huggingface_hub()
    return model


@pytest.fixture
def detector(model: Model) -> Detector:
    return Detector(model, batch_size=2)


@pytest.fixture
def fixtures_folder() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "video_filename, expected",
    [
        ("ai-gen00.mp4", 81.9341),
        ("non-ai-gen00.mp4", 30.695459),
    ],
)
def test_detect(
    detector: Detector,
    fixtures_folder: pathlib.Path,
    video_filename: str,
    expected: float,
):
    video_filepath = fixtures_folder / video_filename
    verdict = detector.detect(video_filepath=video_filepath)
    assert (verdict.predictions[0] * 100) == pytest.approx(expected)

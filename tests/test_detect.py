import pathlib

import pytest

from cakelens.detect import detect_device
from cakelens.detect import Detector
from cakelens.model import Model


@pytest.fixture
def device() -> str:
    return detect_device()


@pytest.fixture
def model(device: str) -> Model:
    model = Model()
    model.load_from_huggingface_hub(device=device)
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
        ("ai-gen00.mp4", 81.93103671073914),
        ("non-ai-gen00.mp4", 30.68598210811615),
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

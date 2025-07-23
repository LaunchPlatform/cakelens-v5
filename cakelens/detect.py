import logging
import os
import pathlib

import torch
from torch.utils.data import DataLoader
from torchcodec.decoders import VideoDecoder

from . import constants
from .data_types import Frameset
from .data_types import Label
from .data_types import Verdict
from .datasets import VideoDataset
from .model import make_transformer
from .model import Model
from .utils import format_percentage_values

logger = logging.getLogger(__name__)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Detector:
    def __init__(
        self,
        model: Model,
        batch_size: int = 1,
        num_workers: int | None = None,
        device: str | None = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = os.cpu_count()
        self.device = device
        if self.device is None:
            self.device = detect_device()
        self.model.to(self.device)

    def detect(self, video_filepath: pathlib.Path) -> Verdict:
        logger.info("Running detection for %s", video_filepath)
        decoder = VideoDecoder(video_filepath)
        framesets = [
            Frameset(
                video_filepath=video_filepath,
                index=index,
            )
            for index, _ in enumerate(
                range(0, decoder.metadata.num_frames, constants.FRAMESET_COUNT)
            )
        ]

        transform = make_transformer()
        dataset = VideoDataset(
            framesets=framesets,
            frame_count=constants.FRAMESET_COUNT,
            frame_width=constants.WINDOW_WIDTH,
            frame_height=constants.WINDOW_HEIGHT,
            transform=transform,
        )

        logger.info(
            "Start evaluating with device=%s, frameset_count=%s, batch_size=%d, num_workers=%d",
            self.device,
            f"{len(dataset):,}",
            self.batch_size,
            self.num_workers,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device=self.device,
        )

        pred_rows = []
        count = 0
        for x in dataloader:
            logits = self.model(x)
            preds = logits.sigmoid()
            pred_rows.append(preds)
            for row in preds:
                logger.info(
                    "[%s] %14s: %s",
                    count,
                    "Predictions",
                    format_percentage_values(row.tolist()),
                )
                count += 1

        pred_mean = torch.vstack(pred_rows).mean(dim=0)
        logger.info("Mean predictions: %s", pred_mean)
        logger.info("Verdict:")
        for label, prob in zip(Label, pred_mean):
            logger.info("%10s: %.2f%%", label.value, (prob * 100.0).item())
        logger.info("Done")
        return Verdict(
            video_filepath=video_filepath,
            frame_count=decoder.metadata.num_frames,
            predictions=pred_mean.tolist(),
        )

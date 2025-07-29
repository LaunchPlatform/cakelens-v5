import logging

import torch
from torch.nn import functional
from torch.profiler import record_function
from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder

from . import constants
from .data_types import Frameset


logger = logging.getLogger(__name__)


class VideoDataset(Dataset):
    def __init__(
        self,
        framesets: list[Frameset],
        decoder: VideoDecoder,
        frame_count: int,
        frame_width: int | None = None,
        frame_height: int | None = None,
        transform=None,
        target_transform=None,
        transfer_to_device: str | None = None,
    ):
        self.framesets = framesets
        self.decoder = decoder
        self.frame_count = frame_count
        self.frame_width = frame_width or constants.WINDOW_WIDTH
        self.frame_height = frame_height or constants.WINDOW_HEIGHT
        self.transform = transform
        self.target_transform = target_transform
        self.transfer_to_device = transfer_to_device

    def __len__(self):
        return len(self.framesets)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frameset = self.framesets[idx]
        logger.debug(
            "Reading video frameset %s, transfer_to_device=%s",
            frameset.index,
            self.transfer_to_device,
        )
        frameset_data = read_frames(
            decoder=self.decoder,
            index=frameset.index,
            frame_count=self.frame_count,
            target_width=self.frame_width,
            target_height=self.frame_height,
        )
        if self.transform:
            with record_function("video_transform"):
                frameset_data = self.transform(frameset_data)
        if self.transfer_to_device is not None:
            frameset_data = frameset_data.to(self.transfer_to_device)
        return frameset_data


def crop_video(
    x: torch.Tensor,
    crop_pos: tuple[int, int] = (0, 0),
    target_frame_count: int = constants.FRAMESET_COUNT,
    target_width: int = constants.WINDOW_WIDTH,
    target_height: int = constants.WINDOW_HEIGHT,
):
    frame_count, channels, height, width = x.shape
    crop_x, crop_y = crop_pos
    cropped = x[
        :,
        :target_frame_count,
        crop_y : crop_y + target_height,
        crop_x : crop_x + target_width,
    ]
    cropped_height = cropped.size(2)
    cropped_width = cropped.size(3)

    # Calculate padding amounts (0 if no padding needed)
    pad_height = target_height - cropped_height  # Pad bottom
    half_pad_height = pad_height // 2
    pad_width = target_width - cropped_width  # Pad right
    half_pad_width = pad_width // 2
    pad_frame = target_frame_count - frame_count
    half_pad_frame = pad_frame // 2

    # Pad with zeros on bottom and right
    padded_x = functional.pad(
        cropped,
        (
            # (pad width left, width right)
            half_pad_width,
            pad_width - half_pad_width,
            # (pad height top, height bottom)
            half_pad_height,
            pad_height - half_pad_height,
            # pad nothing for RGB channels
            0,
            0,
            # (pad beginning frames, pad ending frames)
            half_pad_frame,
            pad_frame - half_pad_frame,
        ),
        mode="constant",
        value=0,
    )
    return padded_x


def read_frames(
    decoder: VideoDecoder,
    index: int,
    crop_pos: tuple[int, int] = (0, 0),
    frame_count: int = constants.FRAMESET_COUNT,
    target_width: int = constants.WINDOW_WIDTH,
    target_height: int = constants.WINDOW_HEIGHT,
) -> torch.Tensor:
    begin = index * frame_count
    end = min((index + 1) * frame_count, decoder.metadata.num_frames)
    frames = decoder[begin:end]
    return crop_video(
        frames.float() / 255.0,
        crop_pos=crop_pos,
        target_frame_count=frame_count,
        target_width=target_width,
        target_height=target_height,
    )

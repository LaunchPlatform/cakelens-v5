import functools
import logging
import typing
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import transforms

from . import constants
from .data_types import Label
from .utils import make_repr_attrs

logger = logging.getLogger(__name__)

AffineInstanceNorm3d = functools.partial(nn.InstanceNorm3d, affine=True)


class StackChannels:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Stack frames RGB Channels into the depth dimension
        _, _, width, height = x.shape
        return x.reshape(1, -1, width, height)


class SkipConnection(nn.Module):
    def __init__(self, main: nn.Module, bypass: nn.Module):
        super().__init__()
        self.add_module("main", main)
        self.add_module("bypass", bypass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.bypass(x)


class Checkpointed(nn.Module):
    def __init__(self, module: nn.Module, enabled: bool, use_reentrant: bool = True):
        super().__init__()
        self.use_reentrant = use_reentrant
        self.enabled = enabled
        self.add_module("main", module)

    def extra_repr(self) -> str:
        return make_repr_attrs(
            [
                ("use_reentrant", self.use_reentrant),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.main(x)
        return checkpoint(self.main, x, use_reentrant=self.use_reentrant)


class ZeroPad(nn.Module):
    def __init__(self, pad: tuple[int, ...]):
        super().__init__()
        self.pad = pad

    def extra_repr(self) -> str:
        return make_repr_attrs(
            [
                ("pad", self.pad),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.pad(x, self.pad, mode="constant", value=0)


class Crop(nn.Module):
    def __init__(self, crop: tuple[tuple[int | None, int | None], ...]):
        super().__init__()
        self.crop = crop
        self._slices = tuple(slice(begin, end) for begin, end in self.crop)

    def extra_repr(self) -> str:
        return make_repr_attrs(
            [
                ("crop", self.crop),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[self._slices]


class SpaceTimeConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        channels_step: int = 32,
        non_linear_op: typing.Type[nn.Module] = nn.LeakyReLU,
        norm_op: typing.Type[nn.Module] | None = nn.InstanceNorm3d,
        conv_cls: typing.Type[nn.Module] = nn.Conv3d,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.channels_step = channels_step
        self.norm_op = norm_op
        self.non_linear_op = non_linear_op
        self.conv_cls = conv_cls
        # Input: (N, in_channels, 11, 170, 170)
        modules = OrderedDict()
        prev_channels = in_channels

        for idx in range(self.depth):
            space_channels = prev_channels + self.channels_step
            time_channels = prev_channels + self.channels_step * 2
            modules[f"space_conv{idx}"] = self.conv_cls(
                in_channels=prev_channels,
                out_channels=space_channels,
                kernel_size=(1, 3, 3),
                padding="same",
            )
            modules[f"space_relu{idx}"] = self.non_linear_op()
            if self.norm_op is not None:
                modules[f"space_norm{idx}"] = self.norm_op(space_channels)
            modules[f"time_conv{idx}"] = self.conv_cls(
                in_channels=space_channels,
                out_channels=time_channels,
                kernel_size=(3, 1, 1),
                padding="same",
            )
            modules[f"time_relu{idx}"] = self.non_linear_op()
            if self.norm_op is not None:
                modules[f"time_norm{idx}"] = self.norm_op(time_channels)
            prev_channels = time_channels
        # Output: (N, in_channels + depth * channels_step * 2, 11, 170, 170)
        to_pad_channel: int = self.out_channels - in_channels
        if to_pad_channel:
            half_to_pad_channel: int = to_pad_channel // 2
            padding_module = ZeroPad(
                (
                    (
                        0,
                        0,
                    )
                    * 3
                )
                + (
                    half_to_pad_channel,
                    to_pad_channel - half_to_pad_channel,
                )
            )
        else:
            padding_module = nn.Identity()
        self.add_module(
            "seq",
            SkipConnection(
                main=nn.Sequential(modules),
                bypass=padding_module,
            ),
        )

    @property
    def out_channels(self) -> int:
        return self.in_channels + (self.depth * self.channels_step * 2)

    def extra_repr(self) -> str:
        return make_repr_attrs(
            [
                ("in_channels", self.in_channels),
                ("out_channels", self.out_channels),
                ("depth", self.depth),
                ("channels_step", self.channels_step),
                ("non_linear_op", self.non_linear_op),
                ("norm_op", self.norm_op),
                ("conv_cls", self.conv_cls),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        kernel_size: int | tuple[int, int, int],
        channels_step: int = 32,
        non_linear_op: typing.Type[nn.Module] = nn.LeakyReLU,
        norm_op: typing.Type[nn.Module] | None = nn.InstanceNorm3d,
        conv_cls: typing.Type[nn.Module] = nn.Conv3d,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.kernel_size = kernel_size
        self.channels_step = channels_step
        self.norm_op = norm_op
        self.non_linear_op = non_linear_op
        self.conv_cls = conv_cls
        # Input: (N, in_channels, 11, 170, 170)
        modules = OrderedDict()
        prev_channels = in_channels

        for idx in range(self.depth):
            current_channels = prev_channels + self.channels_step
            modules[f"space_conv{idx}"] = self.conv_cls(
                in_channels=prev_channels,
                out_channels=current_channels,
                kernel_size=self.kernel_size,
                padding="same",
            )
            modules[f"space_relu{idx}"] = self.non_linear_op()
            if self.norm_op is not None:
                modules[f"space_norm{idx}"] = self.norm_op(current_channels)
            prev_channels = current_channels
        # Output: (N, in_channels + depth * channels_step , 11, 170, 170)
        to_pad_channel: int = self.out_channels - in_channels
        if to_pad_channel:
            half_to_pad_channel: int = to_pad_channel // 2
            padding_module = ZeroPad(
                (
                    (
                        0,
                        0,
                    )
                    * 3
                )
                + (
                    half_to_pad_channel,
                    to_pad_channel - half_to_pad_channel,
                )
            )
        else:
            padding_module = nn.Identity()
        self.add_module(
            "seq",
            SkipConnection(
                main=nn.Sequential(modules),
                bypass=padding_module,
            ),
        )

    @property
    def out_channels(self) -> int:
        return self.in_channels + (self.depth * self.channels_step)

    def extra_repr(self) -> str:
        return make_repr_attrs(
            [
                ("in_channels", self.in_channels),
                ("out_channels", self.out_channels),
                ("depth", self.depth),
                ("kernel_size", self.kernel_size),
                ("channels_step", self.channels_step),
                ("non_linear_op", self.non_linear_op),
                ("norm_op", self.norm_op),
                ("conv_cls", self.conv_cls),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ConvGroup(nn.Module):
    def __init__(
        self,
        in_channels: int,
        block_count: int,
        block_cls: typing.Type[nn.Module] | None = SpaceTimeConvBlock,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.block_count = block_count
        self.block_cls = block_cls
        self.checkpoint = checkpoint
        modules = OrderedDict()
        prev_channels = in_channels
        for idx in range(self.block_count):
            block = self.block_cls(
                in_channels=prev_channels,
            )
            modules[f"block{idx}"] = block
            prev_channels = block.out_channels
        self.out_channels = prev_channels
        self.add_module(
            "seq", Checkpointed(nn.Sequential(modules), enabled=self.checkpoint)
        )

    def extra_repr(self) -> str:
        return make_repr_attrs(
            [
                ("in_channels", self.in_channels),
                ("out_channels", self.out_channels),
                ("block_cls", self.block_cls),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class FullyConnected(nn.Module):
    def __init__(
        self,
        non_linear_op: typing.Type[nn.Module] = nn.LeakyReLU,
    ):
        self.non_linear_op = non_linear_op
        super().__init__()
        self.add_module(
            "seq",
            nn.Sequential(
                OrderedDict(
                    [
                        #
                        ("maxpool", nn.MaxPool3d(kernel_size=5)),
                        # Output: (N, C, 1, 32, 32)
                        #
                        # flatten
                        ("flatten", nn.Flatten(start_dim=1)),
                        # fully connected l0
                        ("linear0", nn.LazyLinear(out_features=512)),
                        ("relu0", self.non_linear_op()),
                        # fully connected l1
                        ("linear1", nn.LazyLinear(out_features=256)),
                        ("relu1", self.non_linear_op()),
                        # fully connected l2
                        ("linear2", nn.LazyLinear(out_features=128)),
                        ("relu2", self.non_linear_op()),
                        # fully connected l4
                        ("relu3", nn.LazyLinear(out_features=len(Label))),
                    ]
                )
            ),
        )

    def extra_repr(self) -> str:
        return make_repr_attrs(
            [
                ("non_linear_op", self.non_linear_op),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class Model(nn.Module):
    def __init__(
        self,
        initial_channels: int = 64,
        spacetime_block_count: int = 6,
        spacetime_block_depth: int = 1,
        channels_step: int = 32,
        checkpoint: bool = False,
        non_linear_op: typing.Type[nn.Module] = nn.LeakyReLU,
        norm_op: typing.Type[nn.Module] | None = AffineInstanceNorm3d,
        conv_cls: typing.Type[nn.Module] = nn.Conv3d,
    ):
        super().__init__()
        self.initial_channels = initial_channels
        self.spacetime_block_count = spacetime_block_count
        self.spacetime_block_depth = spacetime_block_depth
        self.channels_step = channels_step
        self.checkpoint = checkpoint
        self.non_linear_op = non_linear_op
        self.norm_op = norm_op
        self.conv_cls = conv_cls
        modules = OrderedDict(
            [
                # Input (N, 1, 11 * 3, 512, 512)
                #
                # C0. 3x3x3 cube scanning across WxHxC for each frame independently
                (
                    "input_conv0",
                    self.conv_cls(
                        in_channels=1,
                        out_channels=self.initial_channels,
                        kernel_size=3,
                        stride=3,
                    ),
                ),
                ("input_relu0", self.non_linear_op()),
                *(
                    (("input_norm0", self.norm_op(self.initial_channels)),)
                    if self.norm_op is not None
                    else ()
                ),
                (
                    "spacetime_conv_group",
                    ConvGroup(
                        in_channels=self.initial_channels,
                        block_count=self.spacetime_block_count,
                        block_cls=functools.partial(
                            SpaceTimeConvBlock,
                            channels_step=self.channels_step,
                            depth=self.spacetime_block_depth,
                            non_linear_op=self.non_linear_op,
                            norm_op=self.norm_op,
                            conv_cls=self.conv_cls,
                        ),
                        checkpoint=self.checkpoint,
                    ),
                ),
            ]
            # Output: (N, 64, 9, 170, 170)
        )

        modules["fully_connected"] = FullyConnected(non_linear_op=self.non_linear_op)
        self.add_module(
            "seq",
            nn.Sequential(modules),
        )

    def extra_repr(self) -> str:
        return make_repr_attrs(
            [
                ("initial_channels", self.initial_channels),
                ("channels_step", self.channels_step),
                ("non_linear_op", self.non_linear_op),
                ("norm_op", self.norm_op),
                ("conv_cls", self.conv_cls),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)

    def load_from_huggingface_hub(self, device: str | None = None):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error(
                "You need to install huggingface-hub in order to load the model from huggingface-hub"
            )
            raise RuntimeError()
        model_path = hf_hub_download(
            repo_id=constants.HUGGINGFACE_HUB_REPO_ID,
            filename=constants.HUGGINGFACE_HUB_REPO_FILENAME,
        )
        # TODO: flatten the model_state_dict in huggingface hub to make it much eaiser?
        self.load_state_dict(
            torch.load(model_path, map_location=device)["model_state_dict"]
        )


def make_transformer() -> typing.Callable:
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=constants.NORMALIZATION_MEAN, std=constants.NORMALIZATION_STD
            ),
            StackChannels(),
        ]
    )

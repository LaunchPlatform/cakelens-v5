import dataclasses
import enum
import pathlib


@enum.unique
class Label(enum.Enum):
    AI_GEN = "AI_GEN"
    ANIME_2D = "ANIME_2D"
    ANIME_3D = "ANIME_3D"
    VIDEO_GAME = "VIDEO_GAME"
    # Video Generators
    KLING = "KLING"
    HIGGSFIELD = "HIGGSFIELD"
    WAN = "WAN"
    MIDJOURNEY = "MIDJOURNEY"
    HAILUO = "HAILUO"
    RAY = "RAY"
    VEO = "VEO"
    RUNWAY = "RUNWAY"
    SORA = "SORA"
    CHATGPT = "CHATGPT"
    PIKA = "PIKA"
    HUNYUAN = "HUNYUAN"
    VIDU = "VIDU"


@dataclasses.dataclass(frozen=True)
class Frameset:
    index: int


@dataclasses.dataclass(frozen=True)
class Verdict:
    video_filepath: pathlib
    frame_count: int
    predictions: list[float]

"""Shared domain types for the bone-age assessment pipeline."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

Sex = Literal["boy", "girl"]
BoneName = Literal[
    "DIPFifth",
    "DIPThird",
    "DIPFirst",
    "MCPFifth",
    "MCPThird",
    "MCPFirst",
    "MIPFifth",
    "MIPThird",
    "PIPFifth",
    "PIPThird",
    "PIPFirst",
    "Radius",
    "Ulna",
]
Box: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """One detector output converted to a framework-independent representation."""

    box: Box
    confidence: float
    class_id: int


@dataclass(frozen=True, slots=True)
class BoneRegion:
    """A required anatomical region selected from detector outputs."""

    bone_name: BoneName
    box: Box


@dataclass(frozen=True, slots=True)
class Assessment:
    """Maturity stage and RUS-CHN score for one anatomical region."""

    stage: int
    score: int

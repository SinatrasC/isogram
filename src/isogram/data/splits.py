from __future__ import annotations

from enum import StrEnum


class Split(StrEnum):
    ALL = "all"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def filename(self) -> str:
        return f"{self.value}.csv"

from dataclasses import dataclass, field
from typing import Any, Dict

from count_model.count_data import CountData


@dataclass()
class LinkCountData:
    count: int
    total: int
    source_total : int = 0

    def __add__(self, other: "LinkCountData") -> "LinkCountData":
        return LinkCountData(self.count + other.count, self.total + other.total)

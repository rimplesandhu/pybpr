from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from count_model.count_data import CountData


@dataclass()
class LinkCountData:
    count: int
    total: int

    def __add__(self, other: "LinkCountData") -> "LinkCountData":
        return LinkCountData(self.count + other.count, self.total + other.total)

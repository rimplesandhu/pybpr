from dataclasses import dataclass, field
from typing import Any, Dict

from count_model.count_data import CountData


@dataclass(slots=True)
class SourceData:
    total: int = 0
    destination_counts: Dict[Any, int] = field(default_factory=lambda : {})

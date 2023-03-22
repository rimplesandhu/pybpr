from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class CountData:
    count: int = 0
    prior: float = 1e-3

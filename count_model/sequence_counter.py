from abc import ABC, abstractmethod
from typing import Any, Iterator, Tuple
from count_model.count_model import CountModel
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData


class SequenceCounter(ABC):

    @abstractmethod
    def observe_sequence(
        self,
        sources: Iterator,
        destination,
    ) -> None:
        pass

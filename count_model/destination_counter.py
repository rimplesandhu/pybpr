from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Iterator, Tuple
from count_model.count_model import CountModel
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData
from count_model.sequence_counter import SequenceCounter


class DestinationCounter:
    _link_counter: LinkCounter

    def __init__(
        self,
        link_counter: LinkCounter,
    ) -> None:
        super().__init__()
        self._link_counter = link_counter

    def observe_sequence(
        self,
        sequence: Iterator,
        destination,
    ) -> None:
        link_counter = self._link_counter

        for source, num in sequence:
            link_counter.observe_link(source, destination, num)

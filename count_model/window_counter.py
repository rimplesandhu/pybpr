from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, Tuple
from count_model.count_model import CountModel
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData
from count_model.sequence_counter import SequenceCounter


@dataclass(slots=True)
class WindowCounter(SequenceCounter):
    link_counter: LinkCounter
    window_size: int

    def observe_sequence(
        self,
        sequence: Iterator,
    ) -> None:
        link_counter = self.link_counter
        history = deque()

        for element, num in sequence:
            # observe all elements in window as sources to this element
            for source in history:
                link_counter.observe_link(source, element, num)

            # slide window over one element
            history.append(element)
            if len(history) > self.window_size:
                history.popleft()

    def get_sequence_weights(
            self,
            sequence: Iterator,
    ) -> Any:
        link_counter = self.link_counter
        dests = {}
        denomenator = 0
        for source, num in sequence:
            source_data = link_counter.get_source_data(source)
            denomenator += source_data.total
            for dest, num in source_data.destination_counts:
                dests[dest] = dests.get(dest, 0) + num
        return dests, denomenator
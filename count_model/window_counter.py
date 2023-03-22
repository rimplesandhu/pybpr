from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Iterator, Tuple
from count_model.count_model import CountModel
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData
from count_model.sequence_counter import SequenceCounter


class WindowCounter(SequenceCounter):
    _link_counter: LinkCounter
    _window_size: int

    def __init__(
        self,
        link_counter: LinkCounter,
        window_size: int,
    ) -> None:
        super().__init__()
        self._link_counter = link_counter
        self._window_size = window_size

    def observe_sequence(
        self,
        sequence: Iterator,
    ) -> None:
        link_counter = self._link_counter
        history = deque()

        for element, num in sequence:
            # observe all elements in window as sources to this element
            for source in history:
                link_counter.observe_link(source, element, num)

            # slide window over one element
            history.append(element)
            if len(history) > self._window_size:
                history.popleft()

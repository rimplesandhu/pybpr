from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, Tuple
from count_model.count_model import CountModel
from count_model.link_count_data import LinkCountData
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData
from count_model.sequence_counter import SequenceCounter


@dataclass(slots=True)
class PermutationCounter(SequenceCounter):
    link_counter: LinkCounter

    def observe_sequence(
        self,
        sequence: Iterator,
    ) -> None:
        link_counter = self.link_counter
        elements = tuple(sequence)
        for source in elements:
            for dest in elements:
                link_counter.observe_link(source, dest)

    def get_sequence_weights(
            self,
            sequence: Iterator,
    ) -> Any:
        link_counter = self.link_counter
        dests = {}
        denomenator = 0
        for source in sequence:
            source_data = link_counter.get_source_data(source)
            denomenator += source_data.total
            for dest, num in source_data.destination_counts:
                dests[dest] = dests.get(dest, 0) + num
        return dests, denomenator
    
    def get_link_counts(
            self,
            sequence:Iterator,
            dest,
    ) -> Any:
        return (self.link_counter.get_link_count(source, dest) for source in sequence)
    

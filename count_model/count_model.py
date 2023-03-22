from abc import ABC, abstractmethod
from typing import Any, Iterator, Tuple
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData


"""
e->d->c->b->a->x


d->c->b->a->x


c->b->a->x


b->a->x


a->x


x
"""

class CountModel(ABC):
    _link_counter: LinkCounter

    def __init__(
        self,
        link_counter: LinkCounter,
    ) -> None:
        super().__init__()
        self._link_counter = link_counter

    def get_link_weights(
        self,
        source,
        destinations: Iterator,
    ) -> Iterator[Tuple[Any, float]]:
        source_data = self._link_counter.get_source_data(source)
        destination_counts = source_data.destination_counts
        source_count = source_data.count
        return (
            (
                destination,
                self.get_link_weight(
                    source,
                    destination,
                    source_count,
                    destination_counts.get(destination, 0),
                ),
            )
            for destination in destinations
        )

    def get_destination_weights(
        self,
        source,
    ):
        return self.get_link_weights(
            source,
            self._link_counter.get_source_data(source).destination_counts.keys().__iter__(),
        )

    @abstractmethod
    def get_link_weight(
        self,
        source,
        destination,
        source_count: int,
        link_count: int,
    ) -> float:
        return None

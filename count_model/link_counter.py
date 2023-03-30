from dataclasses import dataclass, field
from typing import Any, Dict, Iterator
from count_model.link_count_data import LinkCountData

from count_model.source_data import SourceData


@dataclass(slots=True)
class LinkCounter:
    __link_map: Dict[Any, SourceData] = field(default_factory=lambda: {})

    def __init__(self) -> None:
        self.__link_map = {}

    def observe_link(
        self,
        source,
        destination,
        num: int = 1,
    ) -> None:
        self.observe_destination(
            destination,
            self.get_source_data(source),
            num,
        )

    def get_source_data(
        self,
        source,
    ) -> SourceData:
        # TODO: if this is called a lot to query unregistered sources we should add a method that does not add it to the link map
        link_map = self.__link_map
        source_data = link_map.get(source, None)
        if source_data is None:
            source_data = SourceData()
            link_map[source] = source_data
        return source_data

    def get_link_count(
        self,
        source,
        dest,
    ) -> LinkCountData:
        source_data = self.__link_map.get(source, SourceData(0, {}))
        return LinkCountData(
            source_data.destination_counts.get(dest, 0),
            source_data.total,
        )

    def observe_destination(
        self,
        destination,
        link_data: SourceData,
        num: int,
    ) -> None:
        link_data.total += num
        destination_counts = link_data.destination_counts
        destination_counts[destination] = num + destination_counts.get(destination, 0)

    def get_sequence_weights(
        self,
        sequence: Iterator,
    ) -> Any:
        dests = {}
        denomenator = 0
        for source, num in sequence:
            source_data = self.get_source_data(source)
            denomenator += source_data.total
            for dest, num in source_data.destination_counts:
                dests[dest] = dests.get(dest, 0) + num
        return dests, denomenator

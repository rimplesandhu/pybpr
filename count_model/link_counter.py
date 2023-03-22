from dataclasses import dataclass
from typing import Any, Dict

from count_model.source_data import SourceData


class LinkCounter:
    __slots__ = ("__link_map",)

    __link_map: Dict[Any, SourceData]

    def __init__(self) -> None:
        self.__link_map = {}

    def observe_link(
        self,
        source,
        destination,
        num: int,
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

    def observe_destination(
        self,
        destination,
        link_data: SourceData,
        num: int,
    ) -> None:
        link_data.count += num
        destination_counts = link_data.destination_counts
        destination_counts[destination] = num + destination_counts.get(destination, 0)

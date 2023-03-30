from abc import ABC, abstractmethod
from typing import Any, Iterator, Tuple
from count_model.count_model import CountModel
from count_model.link_counter import LinkCounter
from count_model.source_data import SourceData


class UniformPriorModel(CountModel):
    link_prior: float
    source_prior: float

    def __init__(
        self,
        link_counter: LinkCounter,
        link_prior: float,
        source_prior: float,
    ) -> None:
        super().__init__(link_counter)
        self.link_prior = link_prior
        self.source_prior = source_prior

    def get_link_weight(
        self,
        source,
        destination,
        source_count: int,
        link_count: int,
    ) -> float:
        return (link_count + self.link_prior) / (source_count + self.source_prior)

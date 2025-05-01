from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
import numpy as np


class SuccessiveHalving(HPOAlgorithm):
    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: int,
        max_budget: int,
        eta: int = 2,
    ) -> None:
        super().__init__(cs, total_budget, min_budget, max_budget)
        pass

    def ask(self) -> tuple[dict, float]:
        pass

    def tell(self, config: Configuration, result: float, budget: int) -> None:
        pass

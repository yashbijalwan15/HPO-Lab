from ConfigSpace import ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
import numpy as np


class SuccessiveHalving(HPOAlgorithm):
    def ask(self) -> tuple[dict, float]:
        pass

    def tell(self, config, val, budget) -> None:
        pass

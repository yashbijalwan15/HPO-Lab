from ConfigSpace import ConfigurationSpace, Configuration
from hpo_algorithm import HPOAlgorithm


class BayesianOptimisation(HPOAlgorithm):
    def ask(self) -> tuple[dict, float]:
        pass

    def tell(self, config: Configuration, result: float, budget) -> None:
        pass

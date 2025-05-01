from ConfigSpace import ConfigurationSpace, Configuration
from hpo_algorithm import HPOAlgorithm


class BayesianOptimisation(HPOAlgorithm):
    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: int,
        max_budget: int,
    ) -> None:
        super().__init__(cs, total_budget, min_budget, max_budget)
        pass

    def ask(self) -> tuple[dict, float]:
        pass

    def tell(self, config: Configuration, result: float, budget: int) -> None:
        pass

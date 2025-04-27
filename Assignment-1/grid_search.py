from hpo_algorithm import HPOAlgorithm
from ConfigSpace import ConfigurationSpace


class GridSearch(HPOAlgorithm):
    def ask(self) -> tuple[dict, float]:
        pass
    
    def tell(self) -> None:
        pass
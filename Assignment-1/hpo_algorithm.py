from abc import abstractmethod
from ConfigSpace import Configuration, ConfigurationSpace


class HPOAlgorithm:
    def __init__(self, cs: ConfigurationSpace, total_budget: int, min_budget: int, max_budget: int) -> None:
        self.cs: ConfigurationSpace = cs
        self.budget: int = total_budget
        self.min_budget: int = min_budget
        self.max_budget: int = max_budget

    @abstractmethod
    def ask(self) -> tuple[dict, float]:
        pass
    
    @abstractmethod
    def tell(self, config: Configuration, result: float, budget: int) -> None:
        pass

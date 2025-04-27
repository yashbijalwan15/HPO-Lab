from abc import abstractmethod
from ConfigSpace import Configuration, ConfigurationSpace


class HPOAlgorithm:    
    @abstractmethod
    def ask(self) -> tuple[dict, float]:
        pass
    
    @abstractmethod
    def tell(self, config, result) -> None:
        pass
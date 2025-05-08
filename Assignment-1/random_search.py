from ConfigSpace import ConfigurationSpace
from hpo_algorithm import HPOAlgorithm


class RandomSearch(HPOAlgorithm):
    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: int,
        max_budget: int,
    ) -> None:
        super().__init__(cs, total_budget, min_budget, max_budget)
        
        ratio = max_budget / min_budget
        n_init = int(total_budget  / ratio)
        self.configs = self.sample(n_init)
        self.evals = []
        self.idx = 0
    
    def ask(self) -> tuple[dict, float]:
        if len(self.evals) == len(self.configs):
            print(f"Configs Run: {len(self.configs)}")
            return (None, self.max_budget)
        
        self.idx += 1
        return (self.configs[self.idx - 1], self.max_budget)
    
    def tell(self, config: dict, result: float, budget: int) -> None:
        self.evals.append(result)

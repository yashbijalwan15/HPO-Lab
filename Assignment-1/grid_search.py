from ConfigSpace import ConfigurationSpace
from hpo_algorithm import HPOAlgorithm


class GridSearch(HPOAlgorithm):
    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: int,
        max_budget: int,
    ) -> None:
        super().__init__(cs, total_budget, min_budget, max_budget)
        
        ratio = max_budget / min_budget
        n_init = int(total_budget / ratio)
        
        self.configs = self.grid(num_steps=2)
        print(f"Total Configs: {len(self.configs)}")

        n_init = min(n_init, len(self.configs))
        self.configs = self.configs[:n_init]
        print(f"Configs Run: {len(self.configs)}")

        self.evals = []
        self.idx = 0
    
    def ask(self) -> tuple[dict, float]:
        if len(self.evals) == len(self.configs):
            return (None, self.max_budget)
        
        self.idx += 1
        return (self.configs[self.idx - 1], self.max_budget)
    
    def tell(self, config: dict, result: float, budget: int) -> None:
        self.evals.append(result)

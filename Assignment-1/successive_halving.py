from ConfigSpace import ConfigurationSpace
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
        
        self.eta = eta
        n_rounds = int(np.log(max_budget / min_budget) / np.log(self.eta)) + 1
        n_init = int(self.eta ** n_rounds)

        self.configs = self.sample(n_init)
        self.evals = []
        self.idx = 0
        self.curr_budget = min_budget
    
    def ask(self) -> tuple[dict, float]:
        if len(self.evals) == len(self.configs):
            if self.curr_budget == self.max_budget:
                return (None, self.max_budget)
            
            print(f"Iteration {int(np.emath.logn(self.eta, self.curr_budget / self.min_budget) + 1)}:")
            print(f"Configs: {len(self.configs)}, Budget: {self.curr_budget}")
            
            top_evals = np.argsort(self.evals)[: len(self.evals) // self.eta]
            self.configs = [self.configs[i] for i in top_evals]
            self.evals = []
            self.curr_budget *= self.eta
            self.curr_budget = min(self.curr_budget, self.max_budget)
            print(f"Configs left: {len(self.configs)}")

            self.idx = 0

            if len(self.configs) == 1:
                print(f"Iteration {int(np.ceil(np.emath.logn(self.eta, self.max_budget / self.min_budget)) + 1)}:")
                print(f"Configs: {len(self.configs)}, Budget: {self.max_budget}")
        
        self.idx += 1
        return (self.configs[self.idx - 1], self.curr_budget)

    def tell(self, config: dict, result: float, budget: int) -> None:
        self.evals.append(result)

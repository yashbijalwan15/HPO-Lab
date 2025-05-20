from ConfigSpace import ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
import numpy as np


class SuccessiveHalving(HPOAlgorithm):
    """
    Implements the Successive Halving algorithm for Hyperparameter Optimisation.

    Evaluates a large number of configurations with a small budget, then
    iteratively reduces the number of configurations while increasing the budget
    for the remaining ones.
    """
    
    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: int,
        max_budget: int,
        seed: int = None,
        eta: int = 2,
    ) -> None:
        """
        Initialises the SuccessiveHalving optimizer class.

        Args:
            cs (ConfigurationSpace): The hyperparameter configuration space.
            total_budget (int): Total evaluation budget.
            min_budget (int): Minimum budget per evaluation.
            max_budget (int): Maximum budget per evaluation.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            eta (int, optional): Halving rate. Defaults to 2.
        """

        super().__init__(cs, total_budget, min_budget, max_budget, seed)
        
        self.eta = eta
        ratio = max_budget / min_budget
        n_rounds = int(np.log(ratio) / np.log(self.eta)) + 1
        # n_init = int(self.eta ** n_rounds)
        n_init = int(total_budget * (n_rounds / self.eta + ratio / self.eta**n_rounds)**-1)

        self.configs = self.sample(n_init)
        self.evals = []
        self.idx = 0
        self.curr_budget = min_budget
    
    def ask(self) -> tuple[dict, float]:
        """
        Proposes the next hyperparameter configuration and budget to evaluate.

        Returns:
            tuple[dict, float]: A tuple containing a hyperparameter configuration 
                                and the corresponding budget.
        """

        if len(self.evals) == len(self.configs):
            if self.curr_budget == self.max_budget:
                print(f"Iteration {int(np.ceil(np.emath.logn(self.eta, self.max_budget / self.min_budget)) + 1)}:")
                print(f"Configs: {len(self.configs)}, Budget: {self.max_budget}")
                return (None, self.max_budget)
            
            print(f"Iteration {int(np.emath.logn(self.eta, self.curr_budget / self.min_budget) + 1)}:")
            print(f"Configs: {len(self.configs)}, Budget: {self.curr_budget}")
            
            top_evals = np.argsort(self.evals)[::-1][: len(self.evals) // self.eta]
            self.configs = [self.configs[i] for i in top_evals]
            self.evals = []
            self.curr_budget *= self.eta
            self.curr_budget = min(self.curr_budget, self.max_budget)
            print(f"Configs left: {len(self.configs)}")

            if len(self.configs) == 1:
                self.curr_budget = self.max_budget

            self.idx = 0
        
        self.idx += 1
        return (self.configs[self.idx - 1], self.curr_budget)

    def tell(self, result: float) -> None:
        """
        Reports the result of evaluating a configuration.

        Args:
            result (float): The performance result.
        """

        self.evals.append(result)

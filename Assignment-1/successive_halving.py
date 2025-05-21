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

        # Calculate the number of halving rounds
        ratio = max_budget / min_budget
        n_rounds = int(np.log(ratio) / np.log(self.eta)) + 1

        # Calculate total no. of configs to evaluate given the budget and halving schedule
        # n_init = int(self.eta ** n_rounds)
        n_init = int(total_budget * (n_rounds / self.eta + ratio / self.eta**n_rounds)**-1)

        # Randomly sample n_init configurations from the configspace
        self.configs = self.sample(n_init)
        self.evals = []
        self.idx = 0
        self.curr_budget = min_budget # initial budget
    
    def ask(self) -> tuple[dict, float]:
        """
        Proposes the next hyperparameter configuration and budget to evaluate.

        Returns:
            tuple[dict, float]: A tuple containing a hyperparameter configuration 
                                and the corresponding budget.
        """

        # If current round is finished (all configs evaluated)
        if len(self.evals) == len(self.configs):
            if self.curr_budget == self.max_budget:
                # Final round completed, return None
                print(f"Iteration {int(np.ceil(np.emath.logn(self.eta, self.max_budget / self.min_budget)) + 1)}:")
                print(f"Configs: {len(self.configs)}, Budget: {self.max_budget}")
                return (None, self.max_budget)
            
            # Print current iteration info
            print(f"Iteration {int(np.emath.logn(self.eta, self.curr_budget / self.min_budget) + 1)}:")
            print(f"Configs: {len(self.configs)}, Budget: {self.curr_budget}")
            
            # Select top-performing configs to move to next round
            top_evals = np.argsort(self.evals)[::-1][: len(self.evals) // self.eta]
            self.configs = [self.configs[i] for i in top_evals]

            # Reset evaluation list and increase the budget
            self.evals = []
            self.curr_budget *= self.eta
            self.curr_budget = min(self.curr_budget, self.max_budget)
            print(f"Configs left: {len(self.configs)}")

            # If only one config remains, evaluate it for max budget
            if len(self.configs) == 1:
                self.curr_budget = self.max_budget

            self.idx = 0 # Reset index for next round
        
        # Return next config and budget for evaluation
        self.idx += 1
        return (self.configs[self.idx - 1], self.curr_budget)

    def tell(self, result: float) -> None:
        """
        Reports the result of evaluating a configuration.

        Args:
            result (float): The performance result.
        """

        self.evals.append(result)

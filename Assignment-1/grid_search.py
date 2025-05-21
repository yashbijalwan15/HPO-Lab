from ConfigSpace import ConfigurationSpace
from hpo_algorithm import HPOAlgorithm


class GridSearch(HPOAlgorithm):
    """
    Implements the Grid Search algorithm for Hyperparameter Optimisation.

    Exhaustively evaluates configurations generated from a grid over the
    hyperparameter space.
    """

    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: int,
        max_budget: int,
        seed: int = None,
    ) -> None:
        """
        Initialises the GridSearch optimizer class.

        Args:
            cs (ConfigurationSpace): The hyperparameter configuration space.
            total_budget (int): Total evaluation budget.
            min_budget (int): Minimum budget per evaluation.
            max_budget (int): Maximum budget per evaluation.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """

        super().__init__(cs, total_budget, min_budget, max_budget, seed)
        
        # Calculate total no. of configs to evaluate
        ratio = max_budget / min_budget
        n_init = int(total_budget / ratio)
        
        # Generate grid of n_init configurations
        self.configs = self.grid(n_init, num_steps=2)
        print(f"Configs Run: {len(self.configs)}")

        self.evals = []
        self.idx = 0
    
    def ask(self) -> tuple[dict, float]:
        """
        Proposes the next hyperparameter configuration and budget to evaluate.

        Returns:
            tuple[dict, float]: A tuple containing a hyperparameter configuration 
                                and the corresponding budget.
        """

        # If all configs have been evaluated, return None
        if len(self.evals) == len(self.configs):
            return (None, self.max_budget)
        
        # Return next config and budget for evaluation
        self.idx += 1
        return (self.configs[self.idx - 1], self.max_budget)
    
    def tell(self, result: float) -> None:
        """
        Reports the result of evaluating a configuration.

        Args:
            result (float): The performance result.
        """

        self.evals.append(result)

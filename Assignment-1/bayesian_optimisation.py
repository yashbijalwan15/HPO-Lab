from ConfigSpace import ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm


class BayesianOptimisation(HPOAlgorithm):
    """
    Implements the Bayesian Optimisation algorithm for Hyperparameter Optimisation.

    Uses a Gaussian Process surrogate model and Expected Improvement (EI)
    acquisition function to iteratively select configurations to evaluate.
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
        Initialises the BayesianOptimisation class.

        Args:
            cs (ConfigurationSpace): The hyperparameter configuration space.
            total_budget (int): Total evaluation budget.
            min_budget (int): Minimum budget per evaluation.
            max_budget (int): Maximum budget per evaluation.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """

        super().__init__(cs, total_budget, min_budget, max_budget, seed)
        
        # Gaussian Process surrogate model
        self.gp = GaussianProcessRegressor()
        
        # Calculate total no. of configs to evaluate
        ratio = max_budget / min_budget
        self.n_init = int(total_budget / ratio)

        # Initialise with 5 random configurations
        self.configs = self.sample(5)
        self.evals = []
        self.idx = 0

        self.f_max = None # current best result
    
    def _transform_configs(self, configs: list[dict]):
        """
        Converts a list of configuration dictionaries into a 2D array for model input.

        Args:
            configs (list[dict]): List of hyperparameter configurations.

        Returns:
            np.ndarray: 2D array of vectorized configurations.
        """

        return np.array([self.vectorize(cfg) for cfg in configs])
    
    def _ei(self, mu, sigma):
        """
        Computes the Expected Improvement (EI) acquisition function.

        Args:
            mu (np.ndarray): Predicted means from the surrogate model.
            sigma (np.ndarray): Predicted standard deviations.

        Returns:
            np.ndarray: Acquisition function values for each input.
        """

        a = mu - self.f_max
        z = a / sigma

        return a * norm.cdf(z) + sigma * norm.pdf(z)

    def ask(self) -> tuple[dict, float]:
        """
        Proposes the next hyperparameter configuration and budget to evaluate.

        If enough initial configurations have been evaluated, fits a Gaussian Process
        and uses the Expected Improvement acquisition function to select a new configuration.

        Returns:
            tuple[dict, float]: A tuple containing a hyperparameter configuration 
                                and the corresponding budget.
        """

        # If all configs have been evaluated, return None
        if self.idx >= self.n_init:
            return None, self.max_budget
        
        # If all configs evaluated so far, train GP and choose next config using EI
        if len(self.evals) == len(self.configs):
            X = self._transform_configs(self.configs)
            X = np.nan_to_num(X, nan=-1)
            y = np.array(self.evals)

            # Train GP on past configs
            self.gp.fit(X, y)

            # Generate candidate configurations
            candidates = self.sample(100)
            X_candidates = self._transform_configs(candidates)
            X_candidates = np.nan_to_num(X_candidates, nan=-1)

            # Predict mean and std from GP
            mu, sigma = self.gp.predict(X_candidates, return_std=True)
            self.f_max = max(self.evals) # Update best-so-far

            # Compute EI and select best candidate
            acq_values = self._ei(mu, sigma)
            best_idx = np.argmax(acq_values)

            # Append best candidate to config list
            self.configs.append(candidates[best_idx])
        
        # Return next config and budget for evaluation
        self.idx += 1
        return self.configs[self.idx - 1].copy(), self.max_budget
    
    def tell(self, result: float) -> None:
        """
        Reports the result of evaluating a configuration.

        Args:
            result (float): The performance result.
        """

        self.evals.append(result)

from ConfigSpace import ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm


class BayesianOptimisation(HPOAlgorithm):
    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: int,
        max_budget: int,
    ) -> None:
        super().__init__(cs, total_budget, min_budget, max_budget)
        
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        
        ratio = max_budget / min_budget
        self.n_init = int(total_budget / ratio)
        self.configs = self.sample(5)
        self.evals = []
        self.idx = 0

        self.f_max = None
    
    def _transform_configs(self, configs):
        return np.array([self.vectorize(cfg) for cfg in configs])
    
    def _ei(self, mu, sigma):
        a = mu - self.f_max
        z = a / sigma

        return a * norm.cdf(z) + sigma * norm.pdf(z)

    def ask(self) -> tuple[dict, float]:
        if self.idx >= self.n_init:
            return None, self.max_budget
        
        if len(self.evals) == len(self.configs):
            X = self._transform_configs(self.configs)
            X = np.nan_to_num(X, nan=-1)
            y = np.array(self.evals)

            self.gp.fit(X, y)

            candidates = self.sample(100)
            X_candidates = self._transform_configs(candidates)
            X_candidates = np.nan_to_num(X_candidates, nan=-1)

            mu, sigma = self.gp.predict(X_candidates, return_std=True)
            self.f_max = max(self.evals)

            acq_values = self._ei(mu, sigma)

            best_idx = np.argmax(acq_values)
            self.configs.append(candidates[best_idx])
        
        self.idx += 1
        return self.configs[self.idx - 1].copy(), self.max_budget
    
    def tell(self, config: dict, result: float, budget: int) -> None:
        self.evals.append(result)

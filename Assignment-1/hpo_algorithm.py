from abc import abstractmethod
from ConfigSpace import Configuration, ConfigurationSpace, util
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np
import random
import itertools
from collections import deque


class HPOAlgorithm:
    def __init__(
        self, cs: ConfigurationSpace,
        total_budget: int,
        min_budget: int,
        max_budget: int
    ) -> None:
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

    def sample(self, size: int | None = None) -> list[Configuration] | Configuration:
        # TODO: Add config-id to each config to identify and plot later
        return self.cs.sample_configuration(size)
    
    def grid(self, n_vals: int = 5) -> list[dict]:
        def _get_param_grid(hp_name: str, n_vals: int = 5) -> tuple:
            param = self.cs[hp_name]
            if isinstance(param, (CategoricalHyperparameter)):
                return tuple(param.choices)
            
            elif isinstance(param, (OrdinalHyperparameter)):
                return tuple(param.sequence)
            
            elif isinstance(param, Constant):
                return (param.value,)
            
            elif isinstance(param, UniformFloatHyperparameter):
                if param.log:
                    lower, upper = np.log([param.lower, param.upper])
                    pts = np.exp(np.linspace(lower, upper, n_vals))
                else:
                    lower, upper = param.lower, param.upper
                    pts = np.linspace(lower, upper, n_vals)
                return tuple(pts)
            
            elif isinstance(param, UniformIntegerHyperparameter):
                if param.log:
                    lower, upper = np.log([param.lower, param.upper])
                    pts = np.exp(np.linspace(lower, upper, n_vals))
                else:
                    lower, upper = param.lower, param.upper
                    pts = np.linspace(lower, upper, n_vals)
                return tuple(np.round(pts).astype(int))
            
            raise TypeError(f"Unknown hyperparameter type {type(param)}")

        def _get_cartesian_product(param_grid: list[tuple], hp_names: list[str]) -> list[dict]:
            grid = []
            for values in list(itertools.product(*param_grid)):
                config_dict = dict(zip(hp_names, values))
                grid.append(config_dict)
            
            return grid

        param_grid = []
        hp_names = []

        for hp_name in self.cs.get_all_unconditional_hyperparameters():
            param_grid.append(_get_param_grid(hp_name, n_vals))
            hp_names.append(hp_name)
        
        grid = _get_cartesian_product(param_grid, hp_names)

        unchecked_grid = deque(grid)
        checked_grid = []
        
        while len(unchecked_grid) > 0:
            try:
                _ = Configuration(self.cs, unchecked_grid[0])     
                checked_grid.append(unchecked_grid[0])           
            
            except ValueError as e:
                assert (str(e)[:23] == "Active hyperparameter '" and
                    str(e)[-16:] == "' not specified!"), \
                "Caught exception contains unexpected message."
                values = []
                hp_names = []
                new_active_hp_names = []

                for hp_name in unchecked_grid[0]:
                    values.append((unchecked_grid[0][hp_name],))
                    hp_names.append(hp_name)

                    for new_hp_name in self.cs._children[hp_name]:
                        if (
                            new_hp_name not in new_active_hp_names and
                            new_hp_name not in unchecked_grid[0]
                        ):
                            all_cond_ = True
                            for cond in self.cs._parent_conditions_of[new_hp_name]:
                                if not cond.evaluate(unchecked_grid[0]):
                                    all_cond_ = False
                            if all_cond_:
                                new_active_hp_names.append(new_hp_name)
                
                for hp_name in new_active_hp_names:
                    values.append(_get_param_grid(hp_name, n_vals))
                    hp_names.append(hp_name)
                
                new_grid = _get_cartesian_product(values, hp_names)
                unchecked_grid += new_grid
            
            unchecked_grid.popleft()
        
        return checked_grid

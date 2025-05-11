from abc import abstractmethod
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

import numpy as np
import itertools


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
    def tell(self, config: dict, result: float, budget: int) -> None:
        pass
    
    def sample(self, size: int | None = None) -> list[dict] | dict:
        # TODO: Add config-id to each config to identify and plot later
        return self.cs.sample_configuration(size)
    
    def grid(self, num_steps: int = 5) -> list[dict]:
        def _get_param_grid(hp_name: str, num_steps: int = 5) -> tuple:
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
                    pts = np.exp(np.linspace(lower, upper, num_steps))
                else:
                    lower, upper = param.lower, param.upper
                    pts = np.linspace(lower, upper, num_steps)
                return tuple(pts)
            
            elif isinstance(param, UniformIntegerHyperparameter):
                if param.log:
                    lower, upper = np.log([param.lower, param.upper])
                    pts = np.exp(np.linspace(lower, upper, num_steps))
                else:
                    lower, upper = param.lower, param.upper
                    pts = np.linspace(lower, upper, num_steps)
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
            param_grid.append(_get_param_grid(hp_name, num_steps))
            hp_names.append(hp_name)
        
        unchecked_grid = _get_cartesian_product(param_grid, hp_names)

        checked_grid = []
        
        i = 0
        while i < len(unchecked_grid):
            try:
                _ = Configuration(self.cs, unchecked_grid[i])     
                checked_grid.append(unchecked_grid[i])           
            
            except ValueError:
                values = []
                hp_names = []
                new_active_hp_names = []

                for hp_name in unchecked_grid[i]:
                    values.append((unchecked_grid[i][hp_name],))
                    hp_names.append(hp_name)

                    for new_hp in self.cs.get_children_of(hp_name):
                        new_hp_name = new_hp.name
                        if (
                            new_hp_name not in new_active_hp_names and
                            new_hp_name not in unchecked_grid[i]
                        ):
                            all_cond_ = True
                            for cond in self.cs.get_parent_conditions_of(new_hp_name):
                                if not cond.evaluate(unchecked_grid[i]):
                                    all_cond_ = False
                            if all_cond_:
                                new_active_hp_names.append(new_hp_name)
                
                for hp_name in new_active_hp_names:
                    values.append(_get_param_grid(hp_name, num_steps))
                    hp_names.append(hp_name)
                
                new_grid = _get_cartesian_product(values, hp_names)
                unchecked_grid += new_grid
            
            i += 1

        return checked_grid

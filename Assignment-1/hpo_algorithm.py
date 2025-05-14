from abc import abstractmethod
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from ConfigSpace.conditions import (
    EqualsCondition,
    InCondition,
    GreaterThanCondition,
    LessThanCondition,
)

import numpy as np


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

        conditions = {}
        for condition in self.cs.get_conditions():
            conditions.setdefault(condition.child.name, []).append(condition)
        self.conditions: dict = conditions

    @abstractmethod
    def ask(self) -> tuple[dict, float]:
        pass
    
    @abstractmethod
    def tell(self, config: dict, result: float, budget: int) -> None:
        pass
    
    def is_satisfied(self, hp_name: str, config: dict) -> bool:
        for condition in self.conditions.get(hp_name, []):
            parent_value = config.get(condition.parent.name)

            if isinstance(condition, EqualsCondition):
                if parent_value != condition.value:
                    return False
            
            elif isinstance(condition, InCondition):
                if parent_value not in condition.values:
                    return False
            
            elif isinstance(condition, LessThanCondition):
                if parent_value is None or parent_value >= condition.value:
                    return False
            
            elif isinstance(condition, GreaterThanCondition):
                if parent_value is None or parent_value <= condition.value:
                    return False
            
            else:
                try:
                    if not condition.evaluate(config):
                        return False
                except:
                    continue
        
        return True

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
            grid = [{}]
            for values, hp_name in zip(param_grid, hp_names):
                new_grid = []
                for config in grid:
                    if self.is_satisfied(hp_name, config):
                        new_grid.extend({**config, hp_name: val} for val in values)
                    else:
                        new_grid.append(config.copy())
                grid = new_grid
            return grid

        param_grid = []
        hp_names = []

        for hp_name in self.cs.get_hyperparameter_names():
            param_grid.append(_get_param_grid(hp_name, num_steps))
            hp_names.append(hp_name)
        
        unchecked_grid = _get_cartesian_product(param_grid, hp_names)
        checked_grid = []
        
        for grid in unchecked_grid:
            try:
                _ = Configuration(self.cs, grid)     
                checked_grid.append(grid)
            except ValueError as e:
                print(e)
                continue
        
        return checked_grid

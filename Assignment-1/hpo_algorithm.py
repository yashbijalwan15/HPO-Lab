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
    """
    Base class for Hyperparameter Optimisation algorithms.
    """

    def __init__(
        self, cs: ConfigurationSpace,
        total_budget: int,
        min_budget: int,
        max_budget: int,
        seed: int = None,
    ) -> None:
        """
        Initialises the base HPO algorithm.

        Args:
            cs (ConfigurationSpace): The hyperparameter configuration space.
            total_budget (int): Total evaluation budget.
            min_budget (int): Minimum budget per evaluation.
            max_budget (int): Maximum budget per evaluation.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """

        self.cs: ConfigurationSpace = cs
        self.budget: int = total_budget
        self.min_budget: int = min_budget
        self.max_budget: int = max_budget
        self.seed: int = seed

        # Preprocess conditions for faster checking later
        conditions = {}
        for condition in self.cs.get_conditions():
            conditions.setdefault(condition.child.name, []).append(condition)
        self.conditions: dict = conditions

    @abstractmethod
    def ask(self) -> tuple[dict, float]:
        """
        Proposes the next hyperparameter configuration and budget to evaluate.

        Returns:
            tuple[dict, float]: A tuple containing a hyperparameter configuration 
                                and the corresponding budget.
        """

        pass
    
    @abstractmethod
    def tell(self, result: float) -> None:
        """
        Reports the result of evaluating a configuration.

        Args:
            result (float): The performance result.
        """

        pass
    
    def is_satisfied(self, hp_name: str, config: dict) -> bool:
        """
        Checks whether all conditional constraints for a hyperparameter are satisfied.

        Args:
            hp_name (str): The name of the hyperparameter.
            config (dict): Partial or complete hyperparameter configuration.

        Returns:
            bool: True if all conditions are satisfied, else False.
        """

        for condition in self.conditions.get(hp_name, []):
            parent_value = config.get(condition.parent.name)

            # Check for each type of condition
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
                    # Evaluate any other conditions
                    if not condition.evaluate(config):
                        return False
                except:
                    continue
        
        return True

    def vectorize(self, config: dict) -> list:
        """
        Converts a configuration dictionary to a numeric vector.

        Categorical and ordinal parameters are encoded using their index values.

        Args:
            config (dict): The configuration to vectorize.

        Returns:
            list: Numeric representation of the configuration.
        """

        values = []
        for hp_name in self.cs.get_hyperparameter_names():
            param = self.cs[hp_name]
            if hp_name not in config:
                val = -1 # Placeholder for missing value
            elif isinstance(param, (CategoricalHyperparameter)):
                val = param.choices.index(config[hp_name])
            elif isinstance(param, (OrdinalHyperparameter)):
                val = param.sequence.index(config[hp_name])
            else:
                val = config[hp_name]
            values.append(float(val)) # Convert to float for consistency
        
        return values

    def sample(self, size: int = 1) -> list[dict] | dict:
        """
        Randomly samples valid configurations from the configuration space.

        Args:
            size (int, optional): Number of configurations to sample. Defaults to 1.

        Returns:
            dict or list[dict]: A single sampled configuration or a list of sampled configurations.
        
        Raises:
            ValueError: If valid configurations cannot be sampled after many attempts.
        """

        rng = np.random.default_rng(seed=self.seed)
        hp_names = self.cs.get_hyperparameter_names()
        iteration = 0
        missing = size
        accepted_configurations = []
        
        while len(accepted_configurations) < size:
            for _ in range(missing):
                unsampled_hp = set(hp_names)
                config = {}
                progress = True
                
                while unsampled_hp and progress:
                    for hp_name in list(unsampled_hp):
                        progress = False
                        param = self.cs[hp_name]
                        
                        # Skip if condition not satisfied
                        if not self.is_satisfied(hp_name, config):
                            continue
                        
                        # Sample a value based on parameter type
                        if isinstance(param, (CategoricalHyperparameter)):
                            value = rng.choice(param.choices)
                        
                        elif isinstance(param, (OrdinalHyperparameter)):
                            value = rng.choice(param.sequence)
                                                
                        elif isinstance(param, Constant):
                            value = param.value
                        
                        elif isinstance(param, UniformFloatHyperparameter):
                            if param.log:
                                u = rng.uniform(np.log(param.lower), np.log(param.upper))
                                value = float(np.exp(u))
                            else:
                                value = float(rng.uniform(param.lower, param.upper))
                        
                        elif isinstance(param, UniformIntegerHyperparameter):
                            if param.log:
                                u = rng.uniform(np.log(param.lower), np.log(param.upper))
                                value = int(round(np.exp(u)))
                            else:
                                value = rng.integers(param.lower, param.upper + 1)
                        
                        else:
                            raise TypeError(f"Unknown hyperparameter type {type(param)}")

                        config[hp_name] = value
                        unsampled_hp.remove(hp_name)
                        progress = True
                
                # Try to validate and store configuration dictionary
                try:
                    _ = Configuration(self.cs, config)
                    if config not in accepted_configurations:
                        accepted_configurations.append(config)
                
                except ValueError:
                    iteration += 1

                    if iteration == size * 100:
                        raise ValueError(
                            "Cannot sample valid configuration for "
                            "%s" % self.cs)
            
            missing = size - len(accepted_configurations)
        
        if size == 1:
            return accepted_configurations[0]
        else:
            return accepted_configurations
    
    def grid(self, n_init: int, num_steps: int = 5) -> list[dict]:
        """
        Generates a grid of configurations based on discretized parameter values.
        
        The grid is constructed using a Cartesian product of values from each
        hyperparameter's domain.

        Args:
            n_init (int): Maximum number of configurations to return.
            num_steps (int, optional): Number of values to sample per parameter. Defaults to 5.

        Returns:
            list[dict]: A list of valid configurations from the grid.
        """

        def _get_param_grid(hp_name: str, num_steps: int = 5) -> tuple:
            # Generate grid points for one parameter
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
            # Build Cartesian product with filtering based on conditions
            grid = [{}]
            for values, hp_name in zip(param_grid, hp_names):
                new_grid = []
                for config in grid:
                    if self.is_satisfied(hp_name, config):
                        new_grid.extend({**config, hp_name: val} for val in values)
                    else:
                        new_grid.append(config.copy())

                # Limit size if it exceeds n_init
                if len(new_grid) > n_init:
                    new_grid = [new_grid[i] for i in rng.choice(len(new_grid), n_init, replace=False)]
                grid = new_grid
            return grid

        rng = np.random.default_rng(seed=self.seed)
        param_grid = []
        hp_names = []

        # Generate parameter value grids
        for hp_name in self.cs.get_hyperparameter_names():
            param_grid.append(_get_param_grid(hp_name, num_steps))
            hp_names.append(hp_name)
        
        # Build candidate configurations from the grid
        unchecked_grid = _get_cartesian_product(param_grid, hp_names)
        checked_grid = []
        
        # Try to validate and store configuration dictionary
        for grid in unchecked_grid:
            try:
                _ = Configuration(self.cs, grid)     
                checked_grid.append(grid)
            except ValueError as e:
                print(e)
                continue
        
        return checked_grid

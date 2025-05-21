from pathlib import Path
from random_search import RandomSearch
from bayesian_optimisation import BayesianOptimisation
from grid_search import GridSearch
from successive_halving import SuccessiveHalving
from yahpo_gym import BenchmarkSet, local_config
import pickle
import time


# Set paths for YAHPO benchmark data
parent_path = Path(__file__).parent
local_config.init_config()
local_config.set_data_path((parent_path / "data").resolve())

def run(optimiser_class, scenario, instance, fidelity_param, budget, metric, seed=None):
    """
    Runs the given HPO algorithm on a YAHPO benchmark scenario.

    Args:
        optimiser_class (class): The optimiser class to instantiate.
        scenario (str): The YAHPO Gym benchmark scenario name.
        instance (str): The specific instance of the scenario to optimize.
        fidelity_param (str): The fidelity parameter to control budget.
        budget (int): Total evaluation budget.
        metric (str): The target metric to optimise.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - best_config (dict): The configuration with the best metric observed.
            - best_result (float): The best metric value achieved.
            - count (int): The number of top-level configurations evaluated.
    """

    # Initialise benchmark environment
    bench = BenchmarkSet(scenario=scenario)
    bench.set_instance(value=instance)
    
    # Retrieve configuration space and fidelity parameter values
    cs = bench.get_opt_space(drop_fidelity_params=True)
    fidelity = bench.get_fidelity_space()[fidelity_param]

    # Instantiate the optimiser
    optimiser = optimiser_class(cs=cs, total_budget=budget, min_budget=fidelity.lower, max_budget=fidelity.upper, seed=seed)

    runs = []
    best_result = 0
    best_config = {}
    count = 0
    curr_budget = 0
    budget_levels = [curr_budget]

    # Main optimisation loop
    while curr_budget < budget:
        # Get the next configuration to evaluate
        config, _budget = optimiser.ask()
        if _budget not in budget_levels:
            budget_levels.append(_budget)
        
        # Exit loop if no more configs left to evaluate
        if config is None:
            print(f"Budget Used: {curr_budget:0.2f} / {budget}")
            break
            
        # Count how many configurations are evaluated at initial budget
        if len(budget_levels) == 2:
            count += 1
        
        # Evaluate the configuration on the benchmark
        config[fidelity_param] = _budget
        if scenario == 'rbv2_xgboost': config['repl'] = 10 # max value
        result = bench.objective_function(config)[0][metric]

        # Track the best result and config
        if result > best_result:
            best_result = result
            best_config = config

        # Update the optimiser with the result
        optimiser.tell(result)
        
        # Increment the budget
        config['start_time'] = curr_budget # required for DeepCAVE
        curr_budget += (budget_levels[-1] - budget_levels[-2]) / fidelity.lower
        config['end_time'] = curr_budget # required for DeepCAVE

        config[metric] = result
        runs.append(config) # Store run info
    
    if curr_budget >= budget:
        print(f"Budget Exceeded: {curr_budget:0.2f} / {budget}")

    print(f"Total Runs: {len(runs)}")
    
    # Save results to pickle file
    with open(
        (parent_path / f"results/pkl/{seed}/{optimiser_class.__name__}_{scenario}_{instance}_{total_budget}.pkl").resolve(), "wb"
    ) as f:
        pickle.dump(runs, f)

    return best_config, best_result, count

if __name__ == "__main__":
    total_budget = 10000

    seeds = [0, 42, 1234, 2025, 4321]
    runtimes = []

    for seed in seeds:
        for scenario, instance, fidelity_param, metric in [
            ("nb301", "cifar10", "epoch", "val_accuracy"),
            ("rbv2_xgboost", "16", "trainsize", "acc"),
        ]:
            for optimiser_class in [
                RandomSearch,
                BayesianOptimisation,
                GridSearch,
                SuccessiveHalving,
            ]:
                print(
                    f"Running {optimiser_class.__name__} on {scenario} {instance} with {fidelity_param} at seed={seed}"
                )
                start_time = time.time()
                config, result, total = run(optimiser_class, scenario, instance, fidelity_param, total_budget, metric, seed)
                print(f"Best Result: {result:.3f}")

                # Track runtime for each combination
                runtime = time.time() - start_time
                print(f"Run time: {runtime:.5f} s")
                runtimes.append([optimiser_class.__name__, scenario, seed, runtime, config, result, total])
    
    # Save all runtime results
    with open(
        (parent_path / f"results/pkl/runtimes_{total_budget}.pkl").resolve(), "wb"
    ) as f:
        pickle.dump(runtimes, f)

from pathlib import Path
from random_search import RandomSearch
from bayesian_optimisation import BayesianOptimisation
from grid_search import GridSearch
from successive_halving import SuccessiveHalving
from yahpo_gym import BenchmarkSet, local_config
import pickle
import time


parent_path = Path(__file__).parent
local_config.init_config()
local_config.set_data_path((parent_path / "data").resolve())

def run(optimiser_class, scenario, instance, fidelity_param, budget, metric, seed=None):
    """
    Run the specified optimiser on the given benchmark.

    Parameters:
    optimiser (str): The name of the optimiser to use.
    benchmark (str): The name of the benchmark to use.
    """

    bench = BenchmarkSet(scenario=scenario)
    bench.set_instance(value=instance)
    
    cs = bench.get_opt_space(drop_fidelity_params=True)
    fidelity = bench.get_fidelity_space()[fidelity_param]

    optimiser = optimiser_class(cs=cs, total_budget=budget, min_budget=fidelity.lower, max_budget=fidelity.upper, seed=seed)

    runs = []
    # Run the optimiser on the benchmark
    curr_budget = 0
    budget_levels = [curr_budget]
    while curr_budget < budget:
        # Get the next configuration to evaluate
        config, _budget = optimiser.ask()
        if _budget not in budget_levels:
            budget_levels.append(_budget)
        
        if config is None:
            print(f"Budget Used: {curr_budget:0.2f} / {budget}")
            break

        # Evaluate the configuration on the benchmark
        config[fidelity_param] = _budget
        if scenario == 'rbv2_xgboost': config['repl'] = 10 # max value
        result = bench.objective_function(config)[0][metric]

        # Update the optimiser with the result
        optimiser.tell(config, result, _budget)
        
        # Increment the budget
        config['start_time'] = curr_budget
        curr_budget += (budget_levels[-1] - budget_levels[-2]) / fidelity.lower
        config['end_time'] = curr_budget

        config[metric] = result
        runs.append(config)
    
    if curr_budget >= budget:
        print(f"Budget Exceeded: {curr_budget:0.2f} / {budget}")

    print(f"Total Runs: {len(runs)}")
    with open(
        (parent_path / f"results/pkl/{seed}/{optimiser_class.__name__}_{scenario}_{instance}_{total_budget}.pkl").resolve(), "wb"
    ) as f:
        pickle.dump(runs, f)


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
                run(optimiser_class, scenario, instance, fidelity_param, total_budget, metric, seed)

                runtime = time.time() - start_time
                print(f"Run time: {runtime:.5f} s")
                runtimes.append([optimiser_class.__name__, scenario, seed, runtime])
    
    with open(
        (parent_path / f"results/runtimes_{total_budget}.pkl").resolve(), "wb"
    ) as f:
        pickle.dump(runtimes, f)

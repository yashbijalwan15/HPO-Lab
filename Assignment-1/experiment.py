from random_search import RandomSearch
from bayesian_optimisation import BayesianOptimisation
from grid_search import GridSearch
from successive_halving import SuccessiveHalving
from yahpo_gym import BenchmarkSet
import pickle


def run(optimiser_class, scenario, instance, fidelity_param, budget, metric):
    """
    Run the specified optimiser on the given benchmark.

    Parameters:
    optimiser (str): The name of the optimiser to use.
    benchmark (str): The name of the benchmark to use.
    """
    bench = None

    optimiser = None

    runs = []
    # Run the optimiser on the benchmark
    cur_budget = 0
    while cur_budget < budget:
        # Get the next configuration to evaluate
        config, budget = optimiser.ask()

        # Evaluate the configuration on the benchmark
        result = None

        # Update the optimiser with the result
        optimiser.tell(config, result, budget)

        # Increment the budget
        cur_budget += None
        runs.append((config, result))

    with open(
        f"results/{optimiser_class.__name__}_{scenario}_{instance}.pkl", "wb"
    ) as f:
        pickle.dump(runs, f)


if __name__ == "__main__":
    total_budget = None

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
                f"Running {optimiser_class.__name__} on {scenario} {instance} with {fidelity_param}"
            )
            run(optimiser_class, scenario, instance, fidelity_param, total_budget, metric)

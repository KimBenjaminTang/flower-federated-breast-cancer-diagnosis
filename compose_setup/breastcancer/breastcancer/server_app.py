"""breast-cancer: A Flower / PyTorch app."""
import numpy as np
import os
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Parameters, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from .task import SimpleNN, get_parameters
from typing import Union, Optional, List, Tuple, Dict

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

class SaveModelStrategy(FedAvg):
    """custom strategy to save the model weights after each federated round (and final weight for testing)"""
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            weights_path = "./agg-weights"
            if(not os.path.exists(weights_path)):
                os.mkdir(weights_path)
            np.savez(f"{weights_path}/round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    try:
        num_rounds = context.run_config["num-server-rounds"]
        fraction_fit = context.run_config["fraction_fit"]
        fraction_evaluate = context.run_config["fraction_evaluate"]
        min_fit_clients = context.run_config["min_fit_clients"]
        min_evaluate_clients = context.run_config["min_evaluate_clients"]
        min_available_clients = context.run_config["min_available_clients"]
    except: #if no context is provided (e.g. during the simulation with pyproject.toml, use default values)
        num_rounds = 10
        fraction_fit = 1.0
        fraction_evaluate = 1.0
        min_fit_clients = 5
        min_evaluate_clients = 5
        min_available_clients = 5
    params = get_parameters(SimpleNN())

    # Create FedAvg strategy
    strategy = SaveModelStrategy( #FedAvg(
        fraction_fit=fraction_fit,  # Sample x% of available clients for training
        fraction_evaluate=fraction_evaluate,  # Sample x% of available clients for evaluation
        min_fit_clients=min_fit_clients,  # Never sample less than x clients for training
        min_evaluate_clients=min_evaluate_clients,  # Never sample less than x clients for evaluation
        min_available_clients=min_available_clients,  # Wait until all x clients are available
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=ndarrays_to_parameters(
            params
        ),  # Pass initial model parameters
    )

    # Configure the server for 10 rounds of training
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)

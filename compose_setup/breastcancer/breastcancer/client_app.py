"""breast-cancer: A Flower / PyTorch app."""

import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from .task import SimpleNN, get_parameters, load_breast_cancer_datasets, set_parameters, test, train

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
params = get_parameters(SimpleNN())
DEVICE = torch.device("cpu")

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    net = SimpleNN().to(DEVICE)

    # Note: each client gets a different trainloader/valloader, so each client will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    trainloader, valloader = load_breast_cancer_datasets(partition_id=partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client() to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)

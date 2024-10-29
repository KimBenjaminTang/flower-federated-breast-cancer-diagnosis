"""breast-cancer: A Flower / PyTorch app."""

from collections import OrderedDict

import pandas as pd
import numpy as np

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

seed = 42

BATCH_SIZE = 32

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        data (pd.DataFrame): DataFrame containing feature data.
        labels (pd.Series or np.array): Labels corresponding to the data.
        transform (callable, optional): Optional transform to apply to the data.
        """

        self.data = data.values  # Convert DataFrame to numpy array
        self.labels = labels.values  # Convert labels to numpy array
        self.transform = transform

    def __len__(self):
        # Return the size of the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Get features and label at the specified index
        features = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Apply transformations if provided
        if self.transform:
            features = self.transform(features)

        return features, label

def load_breast_cancer_datasets(partition_id: int):
    """Load the training and validation datasets for an institution as defined by the partition_id"""
    print("Loading breast cancer dataset")
    data_path = "./institution_data/"
    this_institution_data = pd.read_csv(f"{data_path}/data_institution_{partition_id}.csv")
    this_institution_labels = pd.read_csv(f"{data_path}/labels_institution_{partition_id}.csv")
    this_institution_labels = this_institution_labels.squeeze("columns")

    train_data, test_data, train_labels, test_labels = train_test_split(this_institution_data, this_institution_labels, test_size=0.2, random_state=seed)

    train_dataset = CustomDataset(data=train_data, labels=train_labels)
    test_dataset = CustomDataset(data=test_data, labels=test_labels)

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,shuffle=False)

    return trainloader, valloader

class SimpleNN(nn.Module):
    def __init__(self, n_inputs=30):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 32)  # First hidden layer
        self.fc2 = nn.Linear(32, 16)         # Second hidden layer
        self.output = nn.Linear(16, 1)       # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation for first layer
        x = F.relu(self.fc2(x))  # Activation for second layer
        x = torch.sigmoid(self.output(x))  # Sigmoid for binary classification
        return x


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = nn.BCELoss()  # Use BCELoss for binary classification
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()  # Set the model to training mode

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0

        for data, labels in trainloader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(data)

            # Compute the loss
            loss = criterion(outputs.squeeze(), labels.float())  # Ensure labels are float for BCELoss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()  

            # Compute metrics: loss, total, correct classifications
            epoch_loss += loss.item()
            total += labels.size(0)
            predictions = (outputs.squeeze() > 0.5).float()  # Apply threshold for binary classification
            correct += (predictions == labels).sum().item()

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = nn.BCELoss()  # Use BCELoss for binary classification
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for val_data, val_labels in testloader:  # Use the testloader passed as argument
            val_outputs = net(val_data)
            val_loss = criterion(val_outputs.squeeze(), val_labels.float())  # Use BCELoss
            
            loss += val_loss.item()
            
            # Calculate accuracy
            predictions = (val_outputs.squeeze() > 0.5).float()  # Convert probabilities to binary predictions
            correct += (predictions == val_labels).sum().item()
            total += val_labels.size(0)

    # Calculate average validation loss and accuracy for the epoch
    loss /= len(testloader)
    print(f"correct: {correct}, total: {total}")
    val_accuracy = 100 * correct / total
    return loss, val_accuracy

# Define functions to get and set torch model parameters
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

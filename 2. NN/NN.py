import sys

from ucimlrepo import fetch_ucirepo

# Fetch the Ecoli dataset
ecoli = fetch_ucirepo(id=39)

# Extract features (X) and labels (y)
X = ecoli.data.features
y = ecoli.data.targets

# Convert the dataset to a pandas DataFrame
import pandas as pd

dataset = pd.concat([X, y], axis=1)

# Filter the dataset to include only the classes 'cp' and 'im'
filtered_dataset = dataset[dataset['class'].isin(['cp', 'im'])]

# Convert the class labels to binary values ('cp' -> 0, 'im' -> 1)
filtered_dataset.loc[:, 'class'] = filtered_dataset['class'].map({'cp': 0, 'im': 1})

# Prepare features (X) and labels (Y)
X_filtered = filtered_dataset.drop(columns=['class']).values
Y_filtered = filtered_dataset['class'].values

# Verify the shape of the dataset
print("Shape of filtered features:", X_filtered.shape)
print("Shape of filtered labels:", Y_filtered.shape)

import numpy as np


# Activation function and its derivative (Sigmoid for binary classification)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Initialize weights with random values
def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    return weights_input_hidden, weights_hidden_output


# Forward propagation: Compute the outputs
def forward_propagation(X, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output


# Backpropagation: Update the weights
def backpropagation(X, y, hidden_output, output, weights_input_hidden, weights_hidden_output, learning_rate):
    # Output layer error and delta
    output_error = y - output
    output_delta = output_error * sigmoid_prime(output)

    # Hidden layer error and delta
    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_prime(hidden_output)

    # Update weights
    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

    return weights_input_hidden, weights_hidden_output


# Loss function: Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Training loop
def train_mlp(X, y, input_size, hidden_size, output_size, epochs, learning_rate):
    # Initialize weights
    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        hidden_output, output = forward_propagation(X, weights_input_hidden, weights_hidden_output)

        # Calculate loss
        loss = mean_squared_error(y, output)

        # Backward pass (update weights)
        weights_input_hidden, weights_hidden_output = backpropagation(
            X, y, hidden_output, output, weights_input_hidden, weights_hidden_output, learning_rate
        )

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.3f}')
    return weights_input_hidden, weights_hidden_output


# Prepare the filtered dataset for training
X_filtered = X_filtered.astype(np.float32)
Y_filtered = Y_filtered.reshape(-1, 1).astype(np.float32)

# Set the network architecture
input_size = X_filtered.shape[1]  # 7 input features
hidden_size = 4  # 4 hidden neurons
output_size = 1  # Binary output

# Train the MLP from scratch
trained_weights_input_hidden, trained_weights_hidden_output = train_mlp(
    X_filtered, Y_filtered, input_size, hidden_size, output_size, epochs=1001, learning_rate=0.01
)

import torch
import torch.nn as nn
import torch.optim as optim

# Prepare the data (converted to torch tensors)
X_torch = torch.tensor(X_filtered, dtype=torch.float32)
Y_torch = torch.tensor(Y_filtered, dtype=torch.float32).view(-1, 1)


# Define the MLP architecture using PyTorch
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))  # Hidden layer with sigmoid activation
        x = self.sigmoid(self.fc2(x))  # Output layer with sigmoid activation
        return x


# Set the architecture
input_size = X_filtered.shape[1]  # 7 features
hidden_size = 4  # 4 hidden neurons (same as before)
output_size = 1  # Binary output

# Instantiate the model
model = MLP(input_size, hidden_size, output_size)

# Define the loss function and the optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# Training loop for the PyTorch MLP
epochs = 1000
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X_torch)

    # Compute loss
    loss = criterion(y_pred, Y_torch)

    # Zero gradients, backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.2f}')

# Final model evaluation after training
with torch.no_grad():
    y_pred = model(X_torch)
    predictions = (y_pred > 0.5).float()  # Convert probabilities to binary outputs (0 or 1)
    accuracy = (predictions == Y_torch).sum().item() / Y_torch.size(0) * 100
    print(f'Final accuracy: {accuracy:.2f}%')

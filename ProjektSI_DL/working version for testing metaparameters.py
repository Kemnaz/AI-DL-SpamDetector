import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import numpy as np

def train_and_evaluate_model(X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, num_layers, activation_function, learning_rate):
    # Define the model architecture
    class SpamDetector(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, activation_function):
            super(SpamDetector, self).__init__()
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            if activation_function == 'ReLU':
                layers.append(nn.ReLU())
            elif activation_function == 'Sigmoid':
                layers.append(nn.Sigmoid())
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                if activation_function == 'ReLU':
                    layers.append(nn.ReLU())
                elif activation_function == 'Sigmoid':
                    layers.append(nn.Sigmoid())
            layers.append(nn.Linear(hidden_size, output_size))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.astype(int))  # Convert to NumPy array first
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values.astype(int))  # Convert to NumPy array first

    # Instantiate the model
    model = SpamDetector(input_size, hidden_size, output_size, num_layers, activation_function)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        outputs = torch.sigmoid(outputs)  # Apply sigmoid activation function
        loss = criterion(outputs.squeeze(), y_train_tensor.float())
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        outputs = torch.sigmoid(outputs)  # Apply sigmoid activation function
        predicted = (outputs.squeeze() > 0.5).int()
        accuracy = (predicted == y_test_tensor).float().mean().item()

    return accuracy


# Load the SpamBase dataset
spambase = fetch_openml(data_id=44)
X = spambase.data
y = spambase.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define hyperparameters to compare
input_size = X_train.shape[1]
output_size = 1  # Binary classification (spam or not spam)
hidden_size = 64
num_layers_list = [1, 2, 3]  # Number of layers
activation_functions = ['ReLU', 'Sigmoid']  # Activation functions
learning_rates = [0.001, 0.01, 0.1]  # Learning rates

# Perform grid search to compare different hyperparameters
results = []
for num_layers in num_layers_list:
    for activation_function in activation_functions:
        for learning_rate in learning_rates:
            accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test, input_size, hidden_size, output_size, num_layers, activation_function, learning_rate)
            results.append({
                'num_layers': num_layers,
                'activation_function': activation_function,
                'learning_rate': learning_rate,
                'accuracy': accuracy
            })

# Print results
for result in results:
    print(f"Num Layers: {result['num_layers']}, Activation Function: {result['activation_function']}, Learning Rate: {result['learning_rate']}, Accuracy: {result['accuracy']}")

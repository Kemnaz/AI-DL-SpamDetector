import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
import numpy as np

# Load the SpamBase dataset
spambase = fetch_openml(data_id=44)
X = spambase.data
y = spambase.target

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y.astype(int))


# Define the model architecture
class SpamDetector(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(SpamDetector, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        out = torch.sigmoid(self.output_layer(out))
        return out


# Define list of parameters to test
hidden_sizes_values = [(32,), (64,), (128,), (32, 32), (64, 64), (128, 128)]
activation_functions = [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]

learning_rate = 0.001
num_epochs = 10

# Iterate over parameter combinations
for hidden_sizes in hidden_sizes_values:
    for activation_function in activation_functions:
        print(
            f"\nTesting model with hidden_sizes={hidden_sizes}, activation_function={activation_function.__class__.__name__}")

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Instantiate the model
        input_size = X_train.shape[1]
        output_size = 1  # Binary classification (spam or not spam)
        model = SpamDetector(input_size, hidden_sizes, output_size, activation_function)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train.float())
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            predicted = (outputs.squeeze() > 0.5).int()
            accuracy = (predicted == y_test).float().mean()
            print(f'Accuracy on test set: {accuracy.item() * 100:.2f}%')

# Given the air pollution dataset, please use an MLP (multilayer perceptrons) model to predict the
# level of pollutant PM2.5. Specifically, your MLP model needs
# 1. Two hidden layers with ReLU activation functions.
# 2. Use features SO2, PM10, CO, O3_8, airPressure, sunHours, highTemperature,
# lowHumidity, season to predict PM2.5
# 3. Pre-process the dataset, such as checking if there is any missing value in the dataset,
# normalizing the continuous values (z-score), one-hot encoding discrete values.
# 4. Measure your model performance in terms of MSE (mean squared error) and MAE
# (mean absolute error).
# 5. Assume PM2.5 >= 30 is high (unhealthy). Please define a classifier to predict days that are
# healthy and unhealthy.

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv("airQualityData.csv")

print(data.describe())

# Check to see if there are any missing values. If there are,
# they will be printed to the console.

print(data.isnull().sum())

# If any missing values exist, we must drop them

data = data.dropna()

# Our model is only going to focus on these important features.
# We have two types: continuous and discrete.

# Continuous are numbers, and discrete will be one-hot encoded
# and the categories added as individual features with a binary value associated with them.

cont_features = ['SO2', 'PM10', 'CO', 'O3_8', 'airPressure', 'sunHours', 'highTemperature', 'lowHumidity']
disc_features = ['season']

# Our target is PM25

target = 'PM25'

# We make sure to encode our data that way any categorical features
# are one-hot encoded and added as individual features.

data = pd.get_dummies(data, columns=disc_features)

# To normalize the data, we use the following code

scaler = StandardScaler()
data[cont_features] = scaler.fit_transform(data[cont_features])

# We can now split our data into training and testing sets

X = data.drop(columns=[target, 'date', 'cityname', 'citycode', 'year', 'month', 'day', 'longitude', 'latitude'])
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Verify the shape of our training data

print(X_train.shape, y_train.shape)

# Convert our data to float32

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Convert our data to tensors

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define an MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x


# Initialize the model and optimizer
model = MLP(X_train.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
num_epochs = 50
batch_size = 16
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Predict on the test set
with torch.no_grad():
    y_pred = model(X_test)

# Calculate MSE and MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Define a binary classifier
threshold = 30
y_pred_binary = (y_pred >= threshold).float()
y_test_binary = (y_test >= threshold).float()

# Calculate classification accuracy
accuracy = (y_pred_binary == y_test_binary).sum().item() / len(y_test)
print(f'Classification Accuracy: {accuracy}')
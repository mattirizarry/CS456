import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import shap

# Load the dataset
df = pd.read_csv('data.csv')

# Select relevant features
cont_features = ['AQI', 'SO2', 'NO2', 'PM10', 'CO', 'O3_8h', 'PM2.5', 'PRE.2020', 'WS.ex', 'P.mean', 'WS.mean.2min', 'T.mean', 'VP.mean', 'RHU.mean', 'SSD', 'P.min', 'T.min', 'P.max', 'T.max', 'WS.max', 'RHU.min']

# Extract target variable
pm25_target = df['PM2.5']

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(df[cont_features])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, pm25_target, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train.values).view(-1, 1)  # Reshape to make it a column vector
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test.values).view(-1, 1)    # Reshape to make it a column vector

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class LSTM_CNN_Model(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, cnn_out_channels, cnn_kernel_size, output_size):
        super(LSTM_CNN_Model, self).__init__()

        # LSTM path
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)

        # CNN path
        self.cnn = nn.Conv1d(1, cnn_out_channels, kernel_size=cnn_kernel_size, stride=1)

        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size + cnn_out_channels, output_size)

    def forward(self, x):
        # LSTM path
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step

        # CNN path
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.squeeze(2)

        # Concatenate LSTM and CNN outputs
        combined_out = torch.cat((lstm_out, cnn_out), dim=1)

        # Final fully connected layer
        out = self.fc(combined_out)

        return out

# Instantiate the model with correct input size

input_size = 21
lstm_hidden_size = 64
lstm_num_layers = 2
cnn_out_channels = 32
cnn_kernel_size = 3
output_size = 1
model = LSTM_CNN_Model(input_size, lstm_hidden_size, lstm_num_layers, cnn_out_channels, cnn_kernel_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss() # Predict PM25 with MSE
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, pm25_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), pm25_targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    total_loss = 0
    for inputs, pm25_targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), pm25_targets)
        total_loss += loss.item()

average_loss = total_loss / len(test_loader)
print(f'Test Average Loss: {average_loss:.4f}')

# Call the DeepExplainer on our model and data
explainer = shap.DeepExplainer(model, df)

# Get SHAP values for a specific example
shap_values = explainer.shap_values(df)

# Summary Plot
shap.summary_plot(shap_values, features=df, feature_names=cont_features)
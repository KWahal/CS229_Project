import numpy as np
import pandas as pd
import utils as utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define a custom PyTorch dataset
class CustomDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = self.input_data[idx]
        target = self.target_data[idx]
        return input_seq, target

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output

def get_rnn(df):
    # Assuming you have a multivariate DataFrame called 'df' with input features 'input_data'
    # and target variable 'target_data'

    # Preprocess the data
    df = utils.get_auction_type_df(df)
    df_resampled = utils.resample_data(df)
    df = df_resampled.drop(['date', 'Auction high rate %', 'Maturity date'], axis=1)

    input_data = df.values[:, 1:].astype(float)
    target_data = df_resampled['Auction high rate %'].values.astype(float)

    # Create the dataset and data loader
    dataset = CustomDataset(input_data, target_data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Set the device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the input size, hidden size, and output size of the RNN
    input_size = len(df.columns) - 1  # Number of input features
    hidden_size = 64
    output_size = 1  # Number of output classes (1 for regression)
    model = RNN(input_size, hidden_size, output_size).to(device)

    # Set up the loss function, optimizer, and other hyperparameters
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch_input, batch_target in data_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_input)
            loss = loss_fn(output, batch_target.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    # Use the trained model for predictions
    model.eval()
    with torch.no_grad():
        test_input = torch.tensor(test_df[['feature1', 'feature2', 'feature3']].values, dtype=torch.float32)
        predictions = model(test_input.to(device)).squeeze().cpu().numpy()

get_rnn('four_week')
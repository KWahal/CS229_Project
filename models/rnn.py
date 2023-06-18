import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import utils
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

def get_rnn(df):
    df = utils.get_auction_type_df(df)
    train_data, test_data = utils.split_train_test(df, 0.7, split_xy=False)
    test_dates = test_data['date'].tolist()
    train_data = train_data.drop(['date', 'Maturity date'], axis=1)
    test_data = test_data.drop(['date', 'Maturity date'], axis=1)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(train_data)

    sc2 = StandardScaler()
    y_train_scaled = sc2.fit_transform(train_data[['Auction high rate %']])

    hops = 3
    num_train_data = train_data.shape[0]
    num_cols = train_data.shape[1]
    X_train = []
    y_train = []
    for i in range(hops, num_train_data):
        X_train.append(X_train_scaled[i-hops:i])
        y_train.append(y_train_scaled[i][0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train_shape = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(hops, num_cols)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train_shape, y_train, epochs=10, batch_size=32)

    df_train_last = train_data.iloc[-hops:]
    full_df = pd.concat((df_train_last, test_data), axis=0)
    full_df = sc.transform(full_df)

    X_train_shape_pred = []
    num_test_data = full_df.shape[0]
    for i in range(hops, num_test_data):
        X_train_shape_pred.append(full_df[i-hops:i])
    X_train_shape_pred = np.array(X_train_shape_pred)
    print(X_train_shape_pred.shape)

    y_test = model.predict(X_train_shape_pred)
    y_final_pred = sc2.inverse_transform(y_test)

    final_open_pred = pd.DataFrame(y_final_pred)
    final_open_pred.columns = ['final_open_pred']

    final = pd.concat((final_open_pred, test_data.reset_index(drop=True)), axis=1)
    final.insert(0, 'Date', test_dates, True)
    print(final)

    plt.plot(final['Date'], final['Auction high rate %'], label='actual', color='red')
    plt.plot(final['Date'], final['final_open_pred'], label='predicted', color='blue')
    plt.legend()
    # plt.show()
    plt.savefig('images/rnn.png')

""" import numpy as np
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
        output = self.fc(output[:, -1])
        return output

def get_rnn(df):
    # Assuming you have a multivariate DataFrame called 'df' with input features 'input_data'
    # and target variable 'target_data'

    # Preprocess the data
    df = utils.get_auction_type_df(df)
    df_resampled = utils.resample_data(df)
    train_data, test_data = utils.split_train_test(df_resampled, 0.8, split_xy=False)

    input_data = train_data.drop(['date', 'Maturity date', 'Auction high rate %'], axis=1).values.astype(float)
    target_data = train_data['Auction high rate %'].values.astype(float)

    # Create the dataset and data loader
    trim = input_data.shape[0] % 32
    input_data = input_data[trim:]
    target_data = target_data[trim:]
    test_data = test_data[:32]

    dataset = CustomDataset(input_data, target_data)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Set the device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the input size, hidden size, and output size of the RNN
    input_size = input_data.shape[1]  # Number of input features
    hidden_size = 32

    output_size = input_data.shape[0]  # Number of output classes (1 for regression)
    model = RNN(input_size, hidden_size, output_size).to(device)

    # Set up the loss function, optimizer, and other hyperparameters
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.000001)
    num_epochs = 10

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for batch_input, batch_target in data_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_input.unsqueeze(1))
            loss = loss_fn(output, batch_target.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    # Use the trained model for predictions
    model.eval()
    print(model.parameters())
    with torch.no_grad():
        test_input = torch.tensor(test_data.drop(['date', 'Maturity date', 'Auction high rate %'], axis=1).values.astype(float), dtype=torch.float32)
        predictions = model(test_input.to(device)).squeeze().cpu().numpy()

    print(predictions) """

get_rnn('thirteen_week')
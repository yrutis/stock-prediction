import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn


# %%
df = pd.read_csv("stock_norm_cleaned.csv")

#%%
df.shape

# %%
# change type of "column" or row
pd.to_numeric(df.iloc[1,])
all_data = df.iloc[1,]

# %%

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.clf()
plt.title('Month vs Stock')
plt.ylabel('Value')
plt.xlabel('Months')
plt.plot(all_data)
plt.show()

# %%
# split dataset
test_data_size = 50

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]
print(len(train_data))
print(len(test_data))

# %%
train_data_normalized = torch.FloatTensor(train_data).view(-1)
test_data_normalized = torch.FloatTensor(test_data).view(-1)

# calculate back to interests with the following formula:
# e.g. 2nd number = -0.021552 = -2.1552e-02 = -2.1552*10^-2 = -0.021552

# %%

# split the training and test set in samples of the form:
# x = [t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10, t_11, t_12]
# y = [t_13]
# meaning: predict the 13th

train_window = 12


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
test_inout_seq = create_inout_sequences(test_data_normalized, train_window)
print("first training data point: {}".format(train_inout_seq[:1]))
print("first test data point: {}".format(test_inout_seq[:1]))

# %%

# building the neural network model with input size 1 (number of input features = 1 stock)
# output is 1, we want to predict 1 number

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)

# %%%

# Training the model

epochs = 300

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


#%%

# TODO
# optimize training loop to use dataloaders for utilizing GPU during training (instead of loop)
# use validation set
# visualize
# use multiple stocks to learn next interest (t_n+1)
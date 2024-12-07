# https://www.geeksforgeeks.org/long-short-term-memory-networks-using-pytorch/
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__
        self.hidden_dim = hidden_dim
        self.num_layers= num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with 0's
        

model = LSTMModel()
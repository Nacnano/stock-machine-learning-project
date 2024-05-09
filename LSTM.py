import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=512, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, 2, batch_first=True, bias=False)
        self.drop_out = nn.Dropout(p=0.2)
        self.layer_norm = nn.LayerNorm(self.hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size, bias=False)

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq)
        lstm_norm = self.layer_norm(self.drop_out(lstm_out))
        predictions = self.linear(lstm_norm[:,-1,:])
        return predictions

    def reset_hidden_state(self):
        pass

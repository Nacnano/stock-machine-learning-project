import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset

from tqdm.notebook import tqdm

import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class LSTMdataset1(Dataset) :
    def __init__(self, input_dataframe, output_dataframe, input_size = 1) :
        self.input_dataframe = input_dataframe
        self.output_dataframe = output_dataframe
        self.is_preprocessed = False
        self.input_size = input_size

    def Preprocess(self) :
        self.pre_input = []
        self.pre_output = []
        for i in range(self.__len__()) :
            input,output = self.__getitem__(i)
            self.pre_input.append(input)
            self.pre_output.append(output)
        self.is_preprocessed = True

    def __len__(self) :
        return len(self.input_dataframe) - self.input_size + 1
        
    def __getitem__(self, idx) :
        if(self.is_preprocessed) :
            return self.pre_input[idx],self.pre_output[idx]
    
        sub_iput_frame = self.input_dataframe.iloc[idx:idx + self.input_size]
        input_close = torch.log(torch.tensor(sub_iput_frame["Return_1d"].values, dtype=torch.float32) + 1)
        input_open = torch.log(torch.tensor(sub_iput_frame["Open_Return"].values, dtype=torch.float32) + 1)
        input_high = torch.log(torch.tensor(sub_iput_frame["High_Return"], dtype=torch.float32) + 1)
        input_low = torch.log(torch.tensor(sub_iput_frame["Low_Return"], dtype=torch.float32) + 1)

        output = torch.log(torch.tensor(self.output_dataframe.iloc[idx + self.input_size - 1], dtype=torch.float32) + 1)

        return torch.stack((input_open,input_high,input_low,input_close)),output
    
# Model
class LSTMBlock(nn.Module) :
    def __init__(self, input_size, output_size) :
        super().__init__()
        self.LSTM = nn.LSTM(input_size,output_size,1,batch_first=True,bias=True)
        self.Dropout = nn.Dropout(p=0.2)
        self.LayerNorm = nn.LayerNorm(output_size)
    
    def forward(self, input) :
        h1,_ = self.LSTM(input)
        h2 = self.Dropout(h1)
        output = self.LayerNorm(h2)
        return output

class StockLSTM1(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.hidden_size = 64
        self.batch_size = 32
        self.layer1 = LSTMBlock(4,self.hidden_size)
        self.layer2 = LSTMBlock(self.hidden_size,self.hidden_size)
        self.layer3 = nn.Linear(self.hidden_size,1,bias=True)

    def forward(self, input) :
        input = torch.multiply(input,25)
        h1 = self.layer1(input)
        h2 = self.layer2(h1)
        output = self.layer3(h2[:,-1,:])
        output = torch.multiply(output,0.04)
        return output

# Get output function
def LSTM1_stock_predict(model, ticker, input_dataframe, output_dataframe) :
    model.eval()

    X_dataframe = input_dataframe.loc[ticker]
    y_dataframe = output_dataframe.loc[ticker]

    dataset = LSTMdataset1(X_dataframe, y_dataframe, 8)
    dataset.Preprocess()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=False)

    # input size = 8, the prediction is discard
    predict_res = y_dataframe.iloc[0:7].values.tolist()
    target_res = y_dataframe.iloc[0:7].values.tolist()
    
    for batch_X, batch_y in dataloader:
        batch_X = torch.transpose(batch_X[:,:,:],1,2).to(device)
        batch_y = batch_y[:,None].to(device)
        
        y_pred = model(batch_X)

        predict_res += y_pred.to(torch.device("cpu")).reshape(-1).tolist()
        target_res += batch_y.to(torch.device("cpu")).reshape(-1).tolist()

    return np.array(predict_res),np.array(target_res)


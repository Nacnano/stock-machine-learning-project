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
pd.options.mode.chained_assignment = None 

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float32)

# Dataset
class GRUdataset2(Dataset) :
    def __init__(self, input_dataframe, output_dataframe, input_size = 1) :
        self.input_dataframe = input_dataframe
        self.output_dataframe = output_dataframe
        self.is_preprocessed = False
        self.input_size = input_size

        self.input_dataframe["RSI"] = self.input_dataframe["RSI"] / 100
        self.input_dataframe["MACD"] = self.input_dataframe["MACD"]
        self.input_dataframe["Return_1d"] = np.log(self.input_dataframe["Return_1d"] + 1) * 25
        self.input_dataframe["Return_5d"] = np.log(self.input_dataframe["Return_5d"] + 1) * 25
        self.input_dataframe["Return_10d"] = np.log(self.input_dataframe["Return_10d"] + 1) * 25
        self.input_dataframe["Return_21d"] = np.log(self.input_dataframe["Return_21d"] + 1) * 25
        self.input_dataframe["Return_42d"] = np.log(self.input_dataframe["Return_42d"] + 1) * 25
        self.input_dataframe["Return_63d"] = np.log(self.input_dataframe["Return_63d"] + 1) * 25
        self.input_dataframe["Momentum_5"] = np.log(self.input_dataframe["Momentum_5"] + 1) * 25
        self.input_dataframe["Momentum_10"] = np.log(self.input_dataframe["Momentum_10"] + 1) * 25
        self.input_dataframe["Momentum_5_10"] = np.log(self.input_dataframe["Momentum_5_10"] + 1) * 25
        self.input_dataframe["Momentum_21"] = np.log(self.input_dataframe["Momentum_21"] + 1) * 25
        self.input_dataframe["Momentum_5_21"] = np.log(self.input_dataframe["Momentum_5_21"] + 1) * 25
        self.input_dataframe["Momentum_42"] = np.log(self.input_dataframe["Momentum_42"] + 1) * 25
        self.input_dataframe["Momentum_5_42"] = np.log(self.input_dataframe["Momentum_5_42"] + 1) * 25
        self.input_dataframe["Momentum_63"] = np.log(self.input_dataframe["Momentum_63"] + 1) * 25
        self.input_dataframe["Momentum_5_63"] = np.log(self.input_dataframe["Momentum_5_63"] + 1) * 25

    def Preprocess(self) :
        self.pre_input1 = []
        self.pre_input2 = []
        self.pre_output = []
        for i in range(self.__len__()) :
            input1,input2,output = self.__getitem__(i)
            self.pre_input1.append(input1)
            self.pre_input2.append(input2)
            self.pre_output.append(output)
        self.is_preprocessed = True

    def __len__(self) :
        return len(self.input_dataframe) - self.input_size + 1
        
    def __getitem__(self, idx) :
        if(self.is_preprocessed) :
            return self.pre_input1[idx],self.pre_input2[idx],self.pre_output[idx]
    
        sub_input_frame = self.input_dataframe.iloc[idx:idx + self.input_size]
        input_RSI = torch.tensor(sub_input_frame["RSI"].values, dtype=torch.float32)
        input_MACD = torch.tensor(sub_input_frame["MACD"].values, dtype=torch.float32)
        input_Return_1d = torch.tensor(sub_input_frame["Return_1d"], dtype=torch.float32)
        input_Return_5d = torch.tensor(sub_input_frame["Return_5d"], dtype=torch.float32)
        input_Return_10d = torch.tensor(sub_input_frame["Return_10d"], dtype=torch.float32)
        input_Return_Momentum_5 = torch.tensor(sub_input_frame["Momentum_5"], dtype=torch.float32)
        input_Return_Momentum_10 = torch.tensor(sub_input_frame["Momentum_10"], dtype=torch.float32)
        input_Return_Momentum_5_10 = torch.tensor(sub_input_frame["Momentum_5_10"], dtype=torch.float32)
        input_Return_Momentum_21 = torch.tensor(sub_input_frame["Momentum_21"], dtype=torch.float32)
        input_Return_Momentum_5_21 = torch.tensor(sub_input_frame["Momentum_5_21"], dtype=torch.float32)

        input_Month = torch.nn.functional.one_hot(torch.tensor([sub_input_frame.iloc[-1]["Month"] - 1], dtype=torch.int64),num_classes=12).float()

        output = torch.log(torch.tensor(self.output_dataframe.iloc[idx + self.input_size - 1], dtype=torch.float32) + 1) * 25

        return torch.stack((input_RSI
                            ,input_MACD
                            ,input_Return_1d
                            ,input_Return_5d
                            ,input_Return_10d
                            ,input_Return_Momentum_5
                            ,input_Return_Momentum_10
                            ,input_Return_Momentum_5_10
                            ,input_Return_Momentum_21
                            ,input_Return_Momentum_5_21
                            )),input_Month,output

# model
class GRUBlock(nn.Module) :
    def __init__(self, input_size, output_size) :
        super().__init__()
        self.GRU = nn.GRU(input_size,output_size,1,batch_first=True,bias=True)
        self.Dropout = nn.Dropout(p=0.2)
        self.LayerNorm = nn.LayerNorm(output_size)
    
    def forward(self, input) :
        h1,_ = self.GRU(input)
        h2 = self.Dropout(h1)
        output = self.LayerNorm(h2)
        return output

class GRU(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.hidden_size = 64
        self.batch_size = 32
        self.layer1 = GRUBlock(10,self.hidden_size)
        self.layer2 = GRUBlock(self.hidden_size,self.hidden_size)
        self.layer3 = nn.Linear(self.hidden_size,16,bias=True)

    def forward(self, input) :
        h1 = self.layer1(input)
        h2 = self.layer2(h1)
        output = self.layer3(h2[:,-1,:])
        return output
    
class Encoder(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.batch_size = 32
        self.layer1 = nn.Linear(12,32)
        self.layer2 = nn.Linear(32,8)

    def forward(self, input):
        h1 = self.layer1(input)
        output = self.layer2(h1)
        return output

class StockGRU2(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.batch_size = 32
        self.gru = GRU()
        self.month_encoder = Encoder()
        self.layer1 = nn.Linear(24,16)
        self.layer2 = nn.Linear(16,1)

    def forward(self, input1, input2) :
        h00 = self.gru(input1)
        h01 = self.month_encoder(input2)
        h0 = torch.concat((h00,h01),dim=1)
        h1 = self.layer1(h0)
        output = self.layer2(h1)
        return output

def GRU2_stock_predict(model, ticker,  input_dataframe, output_dataframe) :

    X_dataframe = input_dataframe.loc[ticker]
    y_dataframe = output_dataframe.loc[ticker]

    dataset = GRUdataset2(X_dataframe, y_dataframe, 8)
    dataset.Preprocess()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # input size = 8, the prediction is discard
    predict_res = y_dataframe.iloc[0:7].values.tolist()
    target_res = y_dataframe.iloc[0:7].values.tolist()
    
    for batch_X1,bacth_X2, batch_y in dataloader:
        batch_X1 = torch.transpose(batch_X1[:,:,:],1,2).to(device)
        bacth_X2 = bacth_X2[:,-1].to(device)
        batch_y = batch_y[:,None].to(device) / 25 # This is due to the model is scale by 25
        
        y_pred = model(batch_X1,bacth_X2) / 25 # This is due to the model is scale by 25

        predict_res += y_pred.to(torch.device("cpu")).reshape(-1).tolist()
        target_res += batch_y.to(torch.device("cpu")).reshape(-1).tolist()

    return np.array(predict_res),np.array(target_res)

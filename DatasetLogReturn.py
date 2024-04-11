import torch
from torch.utils.data import Dataset

from tqdm.notebook import tqdm

import pandas as pd
import numpy as np
import bisect as bs
import random as rd
from datetime import date

# date scaler : scale day count
date_scaler = 30000

# return number of day count fomr 1 Jan 1962
# d : date string of form yyyy-mm-dd
def DateConverter(d) :
    year = int(d[0:4])
    month = int(d[5:7])
    day = int(d[8:10])
    d0 = date(1962,1,1)
    d1 = date(year,month,day)
    return (d1 - d0).days

# return mean and sd
# a : numpy array
def GetStatProperty(a) :
    return (np.mean(a), np.std(a))

# return normalized data
# a : numpy array
# stat_property : tuple of (mean, std)
def Normalize(a, stat_property) :
    a -= stat_property[0]
    a /= stat_property[1]
    return a

# retrun log-return of a[1:]
# a : numpy array
def LogReturn(a) :
    eps = 1e-10
    min_log = -15

    if np.any(np.abs(a[:-1]) < eps) :
        raise Exception("Devided by zero")
    
    b = a[1:] / a[:-1]
    c = np.log(b)

    if np.any(c < min_log) :
        raise Exception("Log underflow")

    return np.log(b)


# Log return dataset
class StockDatasetLogReturn(Dataset) :
    # The dataset read csv files, return two consecutive data sequence with lenght input_size and output_size
    # root_dirs : list of csv file path, ex.["./data/AAA.csv","./data/BBB.csv,"./data/CCC.csv"]
    # input_size : size of list of input part
    # output_size : size of list of output part
    def __init__(self, root_dirs, input_size = 1, output_size = 1) :
        self.root_dirs = root_dirs
        self.is_preprocessed = False
        self.dataframes = []
        self.stat_prop = []
        self.input_size = input_size
        self.output_size = output_size
        self.start_idx = [0]
        for filepath in tqdm(root_dirs) :
            self.dataframes.append(pd.read_csv(f"{filepath}"))
            # incase we have dataframe shorter than {input_size + output_size} 
            if(self.start_idx[-1] + len(self.dataframes[-1]) - input_size - output_size - 1 < 0) :
                self.dataframes.pop()
                continue
            self.start_idx.append(self.start_idx[-1] + len(self.dataframes[-1]) - input_size - output_size - 1)
            self.stat_prop.append(GetStatProperty(self.dataframes[-1].loc()[:,["Volume"]].to_numpy()))

    # Preprocess all data into pre_input, pre_output
    def Preprocess(self) :
        self.pre_input = []
        self.pre_output = []
        for i in tqdm(range(self.__len__())) :
            input,output = self.__getitem__(i)
            self.pre_input.append(input)
            self.pre_output.append(output)
        self.is_preprocessed = True

    def __len__(self) :
        return self.start_idx[-1]
        
    # return tuple of tensors (input,output)
    # input is tensor of shape (6,input_size) where
    #       input[0] : days count from 1 Jan 1962
    #       input[1] : Log-return of open value
    #       input[2] : Log-return of high value
    #       input[3] : Log-return of low value
    #       input[4] : Log-return of close value
    #       input[5] : normalized volume
    # output is tensor of shape(6,output_size) where
    # just like input but output
    def __getitem__(self, idx) :
        if(self.is_preprocessed) :
            return self.pre_input[idx],self.pre_output[idx]
        dataframe_idx = bs.bisect_right(self.start_idx,idx) - 1
        input_start_entry = idx - self.start_idx[dataframe_idx] + 1
        output_start_entry = input_start_entry + self.input_size

        # in case divided by zero
        try :
            input_date = self.dataframes[dataframe_idx].loc[input_start_entry:input_start_entry + self.input_size - 1,["Date"]].values.reshape(-1).tolist()
            input_date = np.array(list(map(DateConverter,input_date))).astype('float32') / date_scaler
            input_open = self.dataframes[dataframe_idx].loc[input_start_entry - 1:input_start_entry + self.input_size - 1,["Open"]].to_numpy().reshape(-1).astype('float32')
            input_open = LogReturn(input_open)
            input_high = self.dataframes[dataframe_idx].loc[input_start_entry - 1:input_start_entry + self.input_size - 1,["High"]].to_numpy().reshape(-1).astype('float32')
            input_high = LogReturn(input_high)
            input_low = self.dataframes[dataframe_idx].loc[input_start_entry - 1:input_start_entry + self.input_size - 1,["Low"]].to_numpy().reshape(-1).astype('float32')
            input_low = LogReturn(input_low)
            input_close = self.dataframes[dataframe_idx].loc[input_start_entry - 1:input_start_entry + self.input_size - 1,["Close"]].to_numpy().reshape(-1).astype('float32')
            input_close = LogReturn(input_close)
            input_volume = self.dataframes[dataframe_idx].loc[input_start_entry:input_start_entry + self.input_size - 1,["Volume"]].to_numpy().reshape(-1).astype('float32')
            input_volume = Normalize(input_volume,self.stat_prop[dataframe_idx])

            output_date = self.dataframes[dataframe_idx].loc[output_start_entry:output_start_entry + self.output_size - 1,["Date"]].values.reshape(-1).tolist()
            output_date = np.array(list(map(DateConverter,output_date))).astype('float32') / date_scaler
            output_open = self.dataframes[dataframe_idx].loc[output_start_entry - 1:output_start_entry + self.output_size - 1,["Open"]].to_numpy().reshape(-1).astype('float32')
            output_open = LogReturn(output_open)
            output_high = self.dataframes[dataframe_idx].loc[output_start_entry - 1:output_start_entry + self.output_size - 1,["High"]].to_numpy().reshape(-1).astype('float32')
            output_high = LogReturn(output_high)
            output_low = self.dataframes[dataframe_idx].loc[output_start_entry - 1:output_start_entry + self.output_size - 1,["Low"]].to_numpy().reshape(-1).astype('float32')
            output_low = LogReturn(output_low)
            output_close = self.dataframes[dataframe_idx].loc[output_start_entry - 1:output_start_entry + self.output_size - 1,["Close"]].to_numpy().reshape(-1).astype('float32')
            output_close = LogReturn(output_close)
            output_volume = self.dataframes[dataframe_idx].loc[output_start_entry:output_start_entry + self.output_size - 1,["Volume"]].to_numpy().reshape(-1).astype('float32')
            output_volume = Normalize(output_volume,self.stat_prop[dataframe_idx])
        except :
            # return new getiten as randon idx
            return self.__getitem__(rd.randrange(0,self.__len__() - 1))

        return torch.from_numpy(np.vstack((input_date,input_open,input_high,input_low,input_close,input_volume))), torch.from_numpy(np.vstack((output_date,output_open,output_high,output_low,output_close,output_volume)))


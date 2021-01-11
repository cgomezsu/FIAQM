"""
v.e.s.

To make FPC predictions


MIT License

Copyright (c) 2020 Cesar A. Gomez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""                                                                            

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from collections import OrderedDict
import os, time
import pickle

torch.manual_seed(3)
random.seed(3)												        # For reproducibility

def pdata(drate, window=1):
        dfinal = []
        seq = window+1
        for index in range(len(drate)-seq+1):
                dfinal.append(drate[index:index+seq])
        dtrain = np.array(dfinal)
        X_train = dtrain[:,:-1]                             # Select all columns, but the last one
        y_train = dtrain[:,-1]                              # Select last column
                
        # Normalizing data:        
        scaler = MinMaxScaler(feature_range=(0, 1))                        
        Xn_train = scaler.fit_transform(X_train.reshape(-1, window))
        yn_train = scaler.fit_transform(y_train.reshape(-1, 1))
        
        # Converting datasets into tensors:
        X_train = torch.from_numpy(Xn_train).type(torch.Tensor)
        y_train = torch.from_numpy(yn_train).type(torch.Tensor).view(-1)
        
        return [X_train, y_train]

# Define LSTM model
class LSTM(nn.Module):

    def __init__(self, input_size=1, hidden_size=30, output_size=1, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0.2)

        # Define the output layer
        self.linear = nn.Linear(hidden_size, output_size)

        # Hidden state initialization
        self.hidden_cell = (torch.zeros(self.num_layers,1,self.hidden_size),
                            torch.zeros(self.num_layers,1,self.hidden_size))   # Batch size = 1

    def forward(self, input_seq):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden_cell: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        
        # Only take the output from the final timestep
        y_pred = self.linear(lstm_out.view(len(input_seq), -1))
        return y_pred[-1]

lo_model = LSTM()

num_pred = 100
window = 10
samples = num_pred+window

preds_file = '/home/ubuntu/LO/Predictions/dr_est.npy'

while True:

    # Load file with last model parameters
    wo_last = '/home/ubuntu/LO/wo_last.pt'
    while not os.path.isfile(wo_last):
        time.sleep(0)
    while not os.path.getsize(wo_last) >= 79641:                # To make sure that the file is not empty and still being copied (w file is 79641 B in size)
        time.sleep(0)
    wo_dict = torch.load(wo_last)

    with torch.no_grad():                                    
        lo_model.load_state_dict(wo_dict, strict=True)
        
    # Prediction making    
    os.system('tail -n {} ~/ixp_stat/drp_stat_ixp.csv > ~/ixp_stat/drp_tmp_ixp.csv'.format(samples))
    drops = pd.read_csv('~/ixp_stat/drp_tmp_ixp.csv', header=None)
    drop_d = drops.diff()                        
    drop_d = drop_d.abs()                                       # To correct some negative values due to wrong measurements
    os.system('tail -n {} ~/ixp_stat/pkt_stat_ixp.csv > ~/ixp_stat/pkt_tmp_ixp.csv'.format(samples))
    pkts = pd.read_csv('~/ixp_stat/pkt_tmp_ixp.csv', header=None)
    pkt_d = pkts.diff()
    pkt_d = pkt_d.abs()
    dr = (drop_d/pkt_d).fillna(0)

    X, y = pdata(dr[0], window)
    
    lo_model.eval()
    dr_est = np.zeros(num_pred)

    for seq in range(num_pred):
        with torch.no_grad():
            lo_model.hidden_cell = (torch.zeros(lo_model.num_layers, 1, lo_model.hidden_size),
                                    torch.zeros(lo_model.num_layers, 1, lo_model.hidden_size))
            pred = lo_model(X[seq])
            dr_est[seq] = pred.detach().numpy()
    
    # Crop prediction outliers
    dr_est[dr_est < 0] = 0					
    dr_est[dr_est > 1] = 1	
    
    # Send predictions to the other Local Learner
    np.save(preds_file,dr_est)
    os.system('sftp -q -o StrictHostKeyChecking=no -P 54321 -i /home/ubuntu/emula-vm.pem ubuntu@10.10.10.10:{} /home/ubuntu/LLB'.format(preds_file))

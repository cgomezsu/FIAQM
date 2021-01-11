"""
v.e.s.

Local Learner for the FPC training. Its execution is synchronized with lo_train


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
import pickle
import os, time

torch.manual_seed(3)    # For reproducibility
random.seed(3)

# Function to convert data into matrix [X=t, y=t+1] and normalize it for training

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
    

# Define model class
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

# Create local LSTM models:
model_a = LSTM()
model_b = LSTM()
loss_fn_a = nn.MSELoss()
loss_fn_b = nn.MSELoss()

## Training federated model:

n_queue = 20             # Number of queues to be consiedered for the FCP
dr = [None]*n_queue      # Initialization for the list of series corresponding to the drop rate data of each queue
nzs = [None]*n_queue     # List with the non-zero samples of each queue

window = 10                           # Window size for the sequence to be considered   
rounds = 10                           # Max rounds to be run          
Z = 1000                              # Number of local training iterations
learning_rate = 1e-3
samples = Z+window

for r in range(rounds):
    
    nz_queues = 0
    
    while nz_queues < 2:            # If all the queues have no drop rate data, wait until at least two have it
        
        for q in range(n_queue):    # Load all queues data        
            
            os.system('tail -n {} ~/bra_stat/drp_stat_{}.csv > ~/bra_stat/drp_tmp_{}.csv'.format(samples,q+2,q+2))
            drops = pd.read_csv('~/bra_stat/drp_tmp_%s.csv' %(q+2), header=None)
            drop_d = drops.diff()
            os.system('tail -n {} ~/bra_stat/pkt_stat_{}.csv > ~/bra_stat/pkt_tmp_{}.csv'.format(samples,q+2,q+2))
            pkts = pd.read_csv('~/bra_stat/pkt_tmp_%s.csv' %(q+2), header=None)
            pkt_d = pkts.diff()
            dr[q] = (drop_d/pkt_d).fillna(0)
            
            # Get the number of non-zero samples to determine the relative impact of each queue later
            nzs[q] = np.count_nonzero(dr[q])
        
        queues = [q for q in nzs if q != 0]   # Queues with non-zero values are selected only
        nz_queues = len(queues)                    
    
    # Queue selection for the federated training
    q_sel = queues.copy()
    q1 = q_sel.pop(random.randrange(len(q_sel)))            # Choose the first queue randomly and remove it from the list
    q2 = q_sel.pop(random.randrange(len(q_sel)))            # Choose the second queue randomly and remove it from the list
    q1 = nzs.index(q1)                                      # Index of value in the original list
    q2 = nzs.index(q2)

    # Getting tensors of normalized subsets data of selected queues:
    Xtrain_a, ytrain_a = pdata(dr[q1][0], window)       # dr is a list of series. The column 0 of each series has the corresponding queue data
    Xtrain_b, ytrain_b = pdata(dr[q2][0], window)

    # Getting the parameters of the Learning Orchestrator's model
    wo_file = '/home/ubuntu/LLA/wo_%s.pt' %r
    while not os.path.isfile(wo_file):
        time.sleep(0)
    while not os.path.getsize(wo_file) >= 79641:             # To make sure that the file is not empty and still being transferred (w file is 79641 B in size)
        time.sleep(0)
    wo_dict = torch.load(wo_file)
    
    # Load LO parameters to local models
    model_a.load_state_dict(wo_dict)
    model_b.load_state_dict(wo_dict)
     
    optim_a = torch.optim.Adam(model_a.parameters(), lr=learning_rate)
    optim_b = torch.optim.Adam(model_b.parameters(), lr=learning_rate)
    
    for seq in range(Z):  
        
        # Train model a
        optim_a.zero_grad()
        model_a.hidden_cell = (torch.zeros(model_a.num_layers, 1, model_a.hidden_size),
                               torch.zeros(model_a.num_layers, 1, model_a.hidden_size))
        ypred_a = model_a(Xtrain_a[seq])
        loss_a = loss_fn_a(ypred_a, ytrain_a[seq].view(-1))
        loss_a.backward()
        optim_a.step()
        
        # Train model b
        optim_b.zero_grad()
        model_b.hidden_cell = (torch.zeros(model_b.num_layers, 1, model_b.hidden_size),
                               torch.zeros(model_b.num_layers, 1, model_b.hidden_size))
        ypred_b = model_b(Xtrain_b[seq])
        loss_b = loss_fn_b(ypred_b, ytrain_b[seq].view(-1))
        loss_b.backward()
        optim_b.step()
        
            
    # Create file with dk values and send it to the Learning Orchestrator
    dk = [nzs[q1], nzs[q2]]
    d_file = '/home/ubuntu/LLA/dk_%s.data' %r
    
    with open(d_file, 'wb') as filehandle:               # Binary writing to save the list with dk values
        pickle.dump(dk, filehandle)
    os.system('sftp -q -o StrictHostKeyChecking=no -P 54321 -i /home/ubuntu/emula-vm.pem ubuntu@10.10.10.10:{} /home/ubuntu/LO'.format(d_file))
    
    # Create files with the models' parameters and send them to the Learning Orchestrator:        
    wa_file = '/home/ubuntu/LLA/wa_%s.pt' %r
    torch.save(model_a.state_dict(), wa_file)
    os.system('sftp -q -o StrictHostKeyChecking=no -P 54321 -i /home/ubuntu/emula-vm.pem ubuntu@10.10.10.10:{} /home/ubuntu/LO'.format(wa_file))
    
    wb_file = '/home/ubuntu/LLA/wb_%s.pt' %r
    torch.save(model_b.state_dict(), wb_file)
    os.system('sftp -q -o StrictHostKeyChecking=no -P 54321 -i /home/ubuntu/emula-vm.pem ubuntu@10.10.10.10:{} /home/ubuntu/LO'.format(wb_file))


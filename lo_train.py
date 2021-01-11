"""
v.e.s.

Learning Orchestrator for the FPC training


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
import random
from collections import OrderedDict
import os, time
import pickle

torch.manual_seed(3)
random.seed(3)												        # For reproducibility

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
rounds = 10                           # Max rounds to be run    

# Load dictionary with the parameters of the model pre-trained in a previous experiment.
# This is to enable the predictor to make approximate predictions before the first round of the FL process is completed
wo_init = '/home/ubuntu/wo_init.pt'
with torch.no_grad():                            
    lo_model.load_state_dict(torch.load(wo_init), strict=True)

for r in range(rounds):

    # Create file with the model parameters
    wo_file = '/home/ubuntu/LO/wo_%s.pt' %r
    torch.save(lo_model.state_dict(), wo_file)
    
    # Create a copy of the last model parameters file to make predictions
    os.system('cp {} /home/ubuntu/LO/wo_last.pt'.format(wo_file))
    
    # Send file to the Local Learner
    os.system('sftp -q -o StrictHostKeyChecking=no -P 54321 -i /home/ubuntu/emula-vm.pem ubuntu@10.10.10.10:{} /home/ubuntu/LLA &'.format(wo_file))
    
    # Load received file with dk values
    d_file = '/home/ubuntu/LO/dk_%s.data' %r
    while not os.path.isfile(d_file):
        time.sleep(0)
    while not os.path.getsize(d_file) >= 12:                # To make sure that the file is not empty and still being transferred (dk file is 12 B in size)
        time.sleep(0)
    with open(d_file, 'rb') as filehandle:                  # Binary reading to load the received list with dk values
        dk = pickle.load(filehandle)
    
    # Load received file with Model A parameters
    wa_file = '/home/ubuntu/LO/wa_%s.pt' %r
    while not os.path.isfile(wa_file):
        time.sleep(0)
    while not os.path.getsize(wa_file) >= 79641:            # To make sure that the file is not empty and still being transferred (w file is 79641 B in size)
        time.sleep(0)
    wa_dict = torch.load(wa_file)

    # Load received file with Model B parameters
    wb_file = '/home/ubuntu/LO/wb_%s.pt' %r
    while not os.path.isfile(wb_file):
        time.sleep(0)
    while not os.path.getsize(wb_file) >= 79641:             # To make sure that the file is not empty and still being transferred (w file is 79641 B in size)
        time.sleep(0)
    wb_dict = torch.load(wb_file)
    
    # Calculate models' parameters average and load it to the LO model
    sum_dk = dk[0]+dk[1]
    p_a = dk[0]/sum_dk                               # Relative impact of queue A
    p_b = dk[1]/sum_dk                               # Relative impact of queue B
    
    with torch.no_grad():                            # Avoid the autograd when operating parameter tensors
        avg_state_dict = OrderedDict({key : wa_dict[key]*p_a + wb_dict[key]*p_b for key in wa_dict})
        lo_model.load_state_dict(avg_state_dict, strict=True)

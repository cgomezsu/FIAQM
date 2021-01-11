"""
v.e.s.

Federated Congestion Predictor + Intelligent AQM


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

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.cli import CLI
import numpy as np
import pandas as pd
import time, random, os
import substring as ss
import tuner as tnr

random.seed(7)												        # For reproducibility

class CreateTopo(Topo):
    def build(self, n):
        bra = self.addSwitch('bra1')                                # Border Router A
        brb = self.addSwitch('brb1')                                # Border Router B
        ixp = self.addSwitch('ixp1')                                # IXP switch
        vs = self.addSwitch('vs1')                                  # Virtual switch to emulate a tunnel connection between Learning Orchestrator and Local Learners
        
        self.addLink(bra, ixp, bw=1000, delay='2ms')			    # Link between bra-eth1 and ixp-eth1 
        self.addLink(ixp, brb, bw=1000, delay='2ms')			    # Link between ixp-eth2 and brb-eth1 
                                                                   
        
        # Creation of hosts connected to Border Routers in each domain
        for j in range(n):
            BW = random.randint(250,500)
            d = str(random.randint(2,10))+'ms'				        # Each host has a random propagation delay (between 2 and 10 ms) on its link connected to the corresponding router
            ha = self.addHost('a%s' % (j+1), ip='10.10.0.%s' % (j+1))
            self.addLink(ha, bra, bw=BW, delay=d)	                        	
            
            BW = random.randint(250,500)							# Random BW to limit rate on each interface
            d = str(random.randint(2,10))+'ms'				        
            hb = self.addHost('b%s' % (j+1), ip='10.11.0.%s' % (j+1))
            self.addLink(hb, brb, bw=BW, delay=d)
            
        hlla = self.addHost('lla1', ip='10.10.11.11')               # Host acting as the Local Learner A	
        self.addLink(hlla, vs, bw=100)#, delay='2ms')	        
        
        hllb = self.addHost('llb1', ip='10.10.11.12')               # Host acting as the Local Learner B	
        self.addLink(hllb, vs, bw=100)#, delay='2ms')
        
        hlo = self.addHost('lo1', ip='10.10.10.10')                 # Host acting as the Learning Orchestrator
        self.addLink(hlo, vs, bw=100)#, delay='2ms')
        
        ma = self.addHost('ma1', ip='10.0.0.10')			        # There are two monitor hosts for probing
        self.addLink(ma, bra, bw=1000)                              # The BW of the monitor hosts are the same as the inter-domain links
        mb = self.addHost('mb1', ip='10.0.0.11')
        self.addLink(mb, brb, bw=1000)

setLogLevel('info')											    # To show info messages

n = 20                                                          # Number of network elements connected per border router

topo = CreateTopo(n)
net = Mininet(topo, link=TCLink, autoSetMacs=True)			    # We use Traffic Control links

info('\n*** Starting network\n')
net.start()

# Creating network devices from topology
lo1 = net['lo1']
lla1 = net['lla1']
llb1 = net['llb1']
bra1 = net['bra1']
ixp1 = net['ixp1']
brb1 = net['brb1']
ma1 = net['ma1']
mb1 = net['mb1']

# AQM configuration for link between IXP switch and Border Router A
ixp1.cmd('tc qdisc del dev ixp1-eth1 root')                                                                                 # Clear current qdisc
ixp1.cmd('tc qdisc add dev ixp1-eth1 root handle 1:0 htb default 1')                		                                # Set the name of the root as 1:, for future references. The default class is 1
ixp1.cmd('tc class add dev ixp1-eth1 classid 1:1 htb rate 1000mbit')                        	                            # Create class 1:1 as direct descendant of root (the parent is 1:)
ixp1.cmd('tc qdisc add dev ixp1-eth1 parent 1:1 handle 10:1 fq_codel limit 1000 target 50ms interval 1000ms noecn')         # Create qdisc with ID (handle) 10 of class 1. Its parent class is 1:1. Queue size limited to 1000 pkts

# AQM configuration for link between Border Router B and IXP switch. This will be the IAQM
aqm_target = 5000
aqm_interval = 100                                                                                                           # Initial parameters
brb1.cmd('tc qdisc del dev brb1-eth1 root')													
brb1.cmd('tc qdisc add dev brb1-eth1 root handle 1:0 htb default 1')						
brb1.cmd('tc class add dev brb1-eth1 classid 1:1 htb rate 1000mbit')						
brb1.cmd('tc qdisc add dev brb1-eth1 parent 1:1 handle 10:1 fq_codel limit 1000 target {}us interval {}ms noecn'.format(aqm_target,aqm_interval))		# Both target and interval of this queue will be set dynamically as the emulation runs

info('\n*** Setting up AQM for intra-domain link buffers at Border Router\n')

a = [0 for j in range(n)]       # List initialization for hosts A
b = [0 for j in range(n)]       # List initialization for hosts B

for j in range(n):

    # Changing the queue discipline on interfaces connected to Border Router A
    BW = random.randint(250,500)								# Random BW to limit rate on each interface

    bra1.cmd('tc qdisc del dev bra1-eth{} root'.format(j+2))                                                                                
    bra1.cmd('tc qdisc add dev bra1-eth{} root handle 1:0 htb default 1'.format(j+2))                		                                
    bra1.cmd('tc class add dev bra1-eth{} classid 1:1 htb rate {}mbit'.format(j+2,BW))                        	                            
    bra1.cmd('tc qdisc add dev bra1-eth{} parent 1:1 handle 10:1 fq_codel limit 1000 target 2000us interval 40ms noecn'.format(j+2))        
    
    time.sleep(3)                                           # Wait a moment while the AQM is configured
    
    a[j] = net['a%s' % (j+1)]                               # Creating net devices from topology
    b[j] = net['b%s' % (j+1)]
        
time.sleep(5)           

info('\n*** Testing connectivity...\n')

for j in range(n):
    net.ping(hosts=[a[j],b[j]])

net.ping(hosts=[lla1,lo1])
net.ping(hosts=[llb1,lo1])
net.ping(hosts=[ma1,mb1])

info('\n*** Starting AQM stat captures...\n')

bra1.cmd('bash ~/stat_bra.sh')                              # Bash script to capture AQM stats at Border Router A
ixp1.cmd('bash ~/stat_ixp.sh &')                            # Bash script to capture AQM stats at the IXP switch
brb1.cmd('bash ~/stat_brb.sh &')                            # Bash script to capture AQM stats at Border Router B

info('\n*** Starting RRUL traffic between pairs...\n')

for j in range(n):
    a[j].cmd('netserver &')                                           # Start server on domain-A hosts for RRUL tests 
    l = random.randint(300,900)                                       # Random length for each RRUL test
    b[j].cmd('bash rrul.sh 10.10.0.{} {} &'.format(j+1,l))          # Start RRUL tests on domain-B hosts

time.sleep(20)                                   # Waiting time while all RRUL tests start

info('\n*** Capturing AQM stats for initial training...\n')

time.sleep(150)                                  # Waiting time for the getting initial training samples
                                                 # If emulation gets stuck in the first period, it's porbably because there's not enough traffic samples to train the model. Leave a longer time

info('\n*** Starting Federated Congestion Predictor process...\n')

lo1.cmd('mkdir ~/LO')                           # Directory to store parameter files of Learning Orchestrator
lo1.cmd('mkdir ~/LO/Predictions')               # Directory to store predictions of Learning Orchestrator
lla1.cmd('mkdir ~/LLA')                         # Directory to store files of Local Learner A
llb1.cmd('mkdir ~/LLB')                         # Directory to store files of Local Learner B

# Start SCP server on Learning Orchestrator
lo1.cmd('/usr/sbin/sshd -p 54321')

# Start learning process on Learning Orchestrator
lo1.cmd('source PyTorch/bin/activate')              # Activate virtual environment where PyTorch is installed        
lo1.cmd('python LO-Train_v3.py &')
lo1.cmd('python LO-Predict_v4.py &')

# Start learning process on Local Learner A
lla1.cmd('source PyTorch/bin/activate')                      
lla1.cmd('python LL-Train_v2.py &')

# Receive predictions

periods = 300                                       # Number of periods (of aprox 2 secs) to run the emulation
factor = 1000

S = 100												# Number of states: discrete levels of congestion [0, 100] in a period of 2 s			
A = np.arange(1100, 11100, 100)						# Set of actions: set value of target parameter in us
epsilon = 0.5
s_curr = random.randint(0,S)                        # Random initialization of the first observed state for tuning
ind_action = len(A)-1								# Initial action for tuning: max FQ-CoDel target considered (7.5 ms)

cong_pred_max = 1e-6
hist_pred = np.zeros(periods)
hist_local = np.zeros(periods)
hist_r = np.zeros(periods)
hist_rtt = np.ones(periods)*8
hist_tput = np.ones(periods)*100                           
    
ma1.cmd('iperf -s &')								# Iperf server on Monitor A to measure throughput

t0 = time.time()                                    # To take time of all periods
preds_file = '/home/ubuntu/LLB/dr_est.npy'

for i in range(periods):
    
    # Measure RTT and throughput in 1 sec		
    mb1.cmd('iperf -c 10.0.0.10 -i 0.1 -t 1 | tail -1 > ~/LLB/tput.out &')
    mb1.cmd('ping 10.0.0.10 -i 0.1 -w 1 -q | tail -1 > ~/LLB/ping.out')

    print("*** Period",i)
    print("+++ Configured target and interval parameters:",aqm_target,"us",aqm_interval,"ms")
            
            
    # Load received file with predictions, sent by the Learning Orchestrator
    while not os.path.isfile(preds_file):
        time.sleep(0)
    while not os.path.getsize(preds_file) >= 928:           # To make sure that the file is not empty and still being transferred (file size with 100 predictions)
        time.sleep(0)
    
    dr_est = np.load(preds_file)
    
    # Some signal rearrangements of the predictions
    dr_est = dr_est*factor
    dr_est = dr_est-dr_est.mean()
    dr_est = np.abs(dr_est)
    dr_est[dr_est > 1] = 1                                    # Avoid any possible outlier after rearranging

    hist_pred[i] = dr_est.mean()
        
    # Discretize values of predictions
    if hist_pred[i] > cong_pred_max:
        cong_pred_max = hist_pred[i]						# Stores the max value of predicted congestion
    
    s_next = int((hist_pred[i]/cong_pred_max)*S-1)    
    
    print("+++ Predicted level of congestion ahead:",s_next+1)    
    
    time.sleep(1)                                           # The duration of each period is this waiting time + ping time
    
    statinfo = os.stat('/home/ubuntu/LLB/ping.out')
    if statinfo.st_size < 10:
        mRTT = hist_rtt.max()                               # If no ping response was gotten, take the maximum in the records (worst case)
        print ('>>> No mRTT response. Taking maximum known: %.3f' % mRTT, 'ms')
    else:
        din = open('/home/ubuntu/LLB/ping.out').readlines()
        slice = ss.substringByInd(din[0],26,39)
        text = (slice.split('/'))
        mRTT = float(text[1])
        print ('>>> mRTT: %.3f' % mRTT, 'ms')
    
    hist_rtt[i] = mRTT
        
    statinfo = os.stat('/home/ubuntu/LLB/tput.out')
    if statinfo.st_size < 10:
        tput = hist_tput.min()                                   # If no tput response was gotten, take the minimum in the records (worst case)
        print ('>>> No Tput response. Taking minimum known: %.3f' % tput, 'Mbps')
               
    else:
        din = open('/home/ubuntu/LLB/tput.out').readlines()
        tput = float(ss.substringByInd(din[0],34,37))
        unit = ss.substringByInd(din[0],39,39)
        if unit == 'K':
            tput = tput*0.001
        print ('>>> Tput: %.3f' % tput, 'Mbps')
    
    hist_tput[i] = tput
        
    hist_r[i] = tput/mRTT	
    R = hist_r[i]								                # Reward is based on power function
        
    print ('>>> Power: %.2f' % R)
    
    if i >= 75:                                                 # Start the AQM tuning process after this period
        
        # Update Q-values    
        Q = tnr.update(s_curr, ind_action, R, s_next)
        
        s_curr = s_next
        
        # Select action for next iteration
        ind_action = tnr.action(s_curr, epsilon)
        
        aqm_target = A[ind_action]                              # Select a FQ-CoDel target
        aqm_interval = int(aqm_target/(0.05*1000))				# Select a FQ-CoDel interval. Tipycally, target is 5% of interval
        
        brb1.cmd('tc qdisc change dev brb1-eth1 parent 1:1 handle 10:1 fq_codel limit 1000 target {}us interval {}ms noecn'.format(aqm_target,aqm_interval))  # Change the AQM parameters at Border Router B

print("*** Total wall time: ", time.time() - t0, " seconds")
    
np.save('/home/ubuntu/hist_pred.npy',hist_pred)
np.save('/home/ubuntu/hist_power.npy',hist_r)
np.save('/home/ubuntu/q-table.npy',Q)

#CLI(net)                                                       # Uncomment this line to explore the temporary files before exiting Mininet

info('*** Deleting temporary files...\n')

lo1.cmd('rm -r ~/LO') 
lla1.cmd('rm -r ~/LLA') 
llb1.cmd('rm -r ~/LLB')   
  
info('*** Experiment finished!\n')

net.stop()
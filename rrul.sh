#!/bin/bash

: ' 
v.e.s.

Script to run RRUL tests.


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
'

					
SERVER=$1
l=$2									# Random length
s=1

# TCP streams in both directions:
netperf -v 0 -P 0 -s $s -l $l -H $SERVER -t TCP_STREAM >/dev/null &	
netperf -v 0 -P 0 -s $s -l $l -H $SERVER -t TCP_MAERTS >/dev/null &

sleep $s	

# TCP and UDP RTT while under load:
netperf -v 0 -P 0 -s $s -l $l -H $SERVER -t TCP_RR >/dev/null &
netperf -v 0 -P 0 -s $s -l $l -H $SERVER -t UDP_RR >/dev/null &

sleep $s
ping $SERVER -s 272 -i 0.1 -w $l -q >/dev/null &	# To emulate VoIP-like traffic (typical packet 280B; 8B of ICMP header)

wait

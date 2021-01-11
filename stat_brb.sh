#!/bin/bash

: ' 
v.e.s.

Script to capture the queue stats of the interface at Border Router B.


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

mkdir ~/brb_stat
START=$(date +%s%N)

for (( c=0; c<7000; c++ ))
do 
	aqm_stat=$(tc -s qdisc show dev brb1-eth1) 
	echo $aqm_stat | grep -oP -m 1 '(?<=bytes ).*' | cut -d' ' -f1 >> ~/brb_stat/pkt_stat_brb.csv &
	echo $aqm_stat | grep -oP -m 1 '(?<=dropped ).*' | cut -d, -f1 >> ~/brb_stat/drp_stat_brb.csv &
	sleep 55e-3 				# This time + above instructions execution takes about 100 ms in our setting				
done 	

END=$(date +%s%N)
DIFF=$(( $END - $START ))
echo $DIFF >> ~/brb_stat/clock.time					

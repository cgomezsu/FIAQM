#!/bin/bash

: ' 
v.e.s.

Script to create and execute the script files for the capture of each interface at Border Router A.


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

mkdir ~/bra_stat
mkdir ~/bra_stat/scripts

for i in {2..21}									# Create the script files according to the ethernet interfaces at border router
do
	echo '#!/bin/bash' > ~/bra_stat/scripts/eth$i.sh
	echo 'START=$(date +%s%N)' >> ~/bra_stat/scripts/eth$i.sh
	echo 'for (( c=0; c<7000; c++ ))' >> ~/bra_stat/scripts/eth$i.sh
	echo 'do' >> ~/bra_stat/scripts/eth$i.sh
	echo '	aqm_stat=$(tc -s qdisc show dev bra1-eth'$i')' >> ~/bra_stat/scripts/eth$i.sh
	echo '	echo $aqm_stat | grep -oP -m 1 '"'(?<=bytes ).*' | cut -d' ' -f1 >> ~/bra_stat/pkt_stat_"$i".csv" >> ~/bra_stat/scripts/eth$i.sh
	echo '	echo $aqm_stat | grep -oP -m 1 '"'(?<=dropped ).*' | cut -d, -f1 >> ~/bra_stat/drp_stat_"$i".csv" >> ~/bra_stat/scripts/eth$i.sh
	echo '	sleep 45e-3' >> ~/bra_stat/scripts/eth$i.sh				# This time + above instructions execution takes about 100 ms in our setting				
	echo 'done' >> ~/bra_stat/scripts/eth$i.sh						
	echo 'END=$(date +%s%N)' >> ~/bra_stat/scripts/eth$i.sh
	echo 'DIFF=$(( $END - $START ))' >> ~/bra_stat/scripts/eth$i.sh
	echo 'echo $DIFF >> ~/bra_stat/clock.time' >> ~/bra_stat/scripts/eth$i.sh 
	bash ~/bra_stat/scripts/eth$i.sh &								# Execute the script just created
done

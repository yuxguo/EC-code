#!/bin/bash

# seed, gener_level, external_message_mode, src_vr, dst_vr

for ((i=$1;i<$2;i++));do
python main.py $i "l2_inpo" $3 $4 $5 0 &
sleep 2s
done
wait

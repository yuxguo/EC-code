#!/bin/bash

# [start_seed, end_seed) gener_level symbol_onehot_dim dump_message
for ((i=$1;i<$2;i++));do
python main.py $i $3 $4 $5 &
sleep 5s
done
wait

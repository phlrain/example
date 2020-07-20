#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python main.py \
         --cuda \
         --emsize 200 \
         --nhid 200 \
         --dropout 0.0 \
         --epochs 2 \
         --bptt 20 \
         --data data/simple-examples/data/

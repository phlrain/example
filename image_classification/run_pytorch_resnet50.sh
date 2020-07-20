#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python3 train_pytorch.py \
          --workers=8 \
          --batch_size=128 \
          --data_dir=./data/ILSVRC2012 \
          --model=ResNet50

#!/bin/bash
python train.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina \
 --lr 0.00015 --epoch_count 0 --n_epochs 15 \
 --pretrained_name task2a


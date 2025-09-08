#!/bin/bash
python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input B --results_dir ./results/case_1

python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BC --results_dir ./results/case_2

python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BD --haze_intensity 0 --results_dir ./results/case_3

python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BD --haze_intensity 2 --results_dir ./results/case_4

python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BDE --haze_intensity 1 --results_dir ./results/case_5

python test.py --dataroot ./datasets/raina --name task2a --model cbd --dataset_mode raina --test_input BCDE --haze_intensity 1 --results_dir ./results/case_6

python psnr.py 
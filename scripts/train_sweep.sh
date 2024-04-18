#!/bin/bash
nohup python train_sweep.py > train_sweep.log 2>&1 &
tail -f train_sweep.log

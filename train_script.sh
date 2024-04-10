#!/bin/bash
source .venv/bin/activate
nohup python train_script.py > train_script.log 2>&1 &
tail -f train_script.log

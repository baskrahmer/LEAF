#!/bin/bash
source .venv/bin/activate
nohup python mlm_sweep.py > mlm_sweep.log 2>&1 &
tail -f mlm_sweep.log

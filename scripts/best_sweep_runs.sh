#!/bin/bash
nohup ../.venv/bin/python best_sweep_runs.py > best_sweep_runs.log 2>&1 &
tail -f best_sweep_runs.log

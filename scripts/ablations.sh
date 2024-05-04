#!/bin/bash
nohup ../.venv/bin/python ablations.py > ablations.log 2>&1 &
tail -f ablations.log

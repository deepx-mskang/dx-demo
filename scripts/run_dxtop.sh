#!/bin/bash

#terminator -e "bash -c 'dxtop; exec bash'"
cd ~/dx-demos/perf-monitor

source ../.venv/bin/activate

python3 perf_monitor_design.py

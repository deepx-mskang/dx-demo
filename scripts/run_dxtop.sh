#!/bin/bash

#terminator -e "bash -c 'dxtop; exec bash'"
cd ~/demos/perf_monitor

source .venv-perf/bin/activate

python3 perf_monitor_design.py

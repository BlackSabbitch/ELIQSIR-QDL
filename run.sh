#!/bin/bash

LOG_PATH=${LOG_PATH:-run.log}
python run.py "$@" 2>&1 | tee "$LOG_PATH"

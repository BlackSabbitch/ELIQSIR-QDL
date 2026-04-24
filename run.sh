# #!/bin/bash

python run.py "$@" 2>&1 | tee "$LOG_PATH"
